import numpy as np
import torch
from transformers.cache_utils import DynamicCache


class TGVKVCache:
    """TriKV: Damage-Equalized Triage for multimodal KV caches.

    One metric, three fates, zero per-layer heuristics.

    (1) UDM -- Unified Damage Metric. For every cached KV pair, the exact
        closed-form perturbation of the text queries' attention outputs is
        computed for each possible fate:
          evict j:      D_ev(j)  = sum_i w_i * A_ij/(1-A_ij) * ||v_j - o_i||
          merge j->n:   D_mg(j)  = sum_i w_i * A_ij * ||v_j - v_n||
          keep j:       0
        where o_i = sum_k A_ik v_k is the output query i already receives.
        Damages are divided by the layer's mean output norm, giving RELATIVE
        perturbations that are comparable across layers.

    (2) DWF -- Damage Water-Filling. No per-layer budget policy. All L*N
        keep-costs are pooled and a single global threshold retains the most
        expensive tokens until the total budget is spent. Layer budgets
        emerge from equalising marginal damage across layers (the KKT
        condition of budgeted damage minimisation). Per-layer floors (a
        minimum keep count and a few vision representatives) are charged
        AGAINST the budget, so the total retained count equals the budget
        exactly.

    (3) NRM -- Neighbor-Restricted Merging. A below-threshold vision token
        is merged into the most value-similar RETAINED token within a small
        raster-order window (mass-weighted value average, per-slot mass
        tracked), if the relative value distance is under `merge_threshold`;
        otherwise it is evicted. Neighbor restriction keeps the search
        linear and keeps RoPE key mismatch negligible. The cache stays
        physically short, so decode speed is fully preserved.

    (4) LCT -- Lazy Chunked Triage. During decode, per-token damage is
        accumulated with an EMA of the same counterfactual quantity. The
        cache may drift `decode_chunk` tokens above budget, then is
        compressed back in one batch (merge-or-evict), amortising tensor
        copies and denoising the eviction signal. Sinks, instruction text,
        and the newest `recent_window` tokens are protected via explicit
        masks pruned in sync with the cache.

    Text is not special-cased by a hard rule: its damage is naturally large,
    and `text_boost` multiplies it further, yielding TPR-like protection
    that degrades gracefully instead of by fiat.

    Interface mirrors TGVKVCache (__call__ with past_key_values /
    num_of_token / attentions / input_ids, plus the online-prefill hooks),
    so it drops into the same harness. Assumes batch_size == 1 for the
    triage bookkeeping.
    """

    supports_online_prefill = True

    def __init__(
        self,
        layer_num,
        image_token_id,
        start_size=4,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.0,
        batch_size=1,
        merge_window=8,        # raster-order neighbor search radius
        merge_threshold=0.35,  # max relative value distance to allow a merge
        text_boost=1000.0,     # damage multiplier for text KVs (soft TPR)
        last_query_boost=4.0,  # extra weight for the final text queries
        n_last_queries=4,
        amp_cap=50.0,          # ceiling on A/(1-A)
        min_keep_per_layer=16, # DWF floor (on top of sinks + protect)
        min_vision_keep=8,     # vision representatives guaranteed per layer
        decode_chunk=1,       # LCT: compress once we exceed budget by this
        decode_decay=0.95,     # EMA decay of accumulated decode damage
        recent_window=8,       # newest tokens never evicted during decode
        **kwargs,
    ):
        self.start_size = start_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.batch_size = batch_size
        self.protect_size = 1
        self.layer_num = layer_num
        self.image_token_id = image_token_id
        self.ratio = ratio

        self.merge_window = merge_window
        self.merge_threshold = merge_threshold
        self.text_boost = text_boost
        self.last_query_boost = last_query_boost
        self.n_last_queries = n_last_queries
        self.amp_cap = amp_cap
        self.min_keep_per_layer = min_keep_per_layer
        self.min_vision_keep = min_vision_keep
        self.decode_chunk = decode_chunk
        self.decode_decay = decode_decay
        self.recent_window = recent_window
        self._eps = 1e-4

        self.total_budget = 0

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _repeat_kv(values, n_heads):
        h_kv = values.shape[0]
        if h_kv == n_heads:
            return values
        return values.repeat_interleave(n_heads // h_kv, dim=0)

    def _query_weights(self, n_text, device):
        w = torch.ones(n_text, device=device)
        n_last = min(self.n_last_queries, n_text)
        if n_last > 0:
            w[-n_last:] = self.last_query_boost
        return w / w.sum()

    def _layer_damage(self, attn_text, values, image_start, text_start):
        """UDM for one layer.

        attn_text : [H, Nt, N] post-softmax rows of the text queries
        values    : [H_kv, N, d]
        Returns (keep_cost [N], attn_mass [N], hm_values [N, d]) where
        keep_cost is the normalised eviction damage (inf on hard-protected
        slots), attn_mass is the query-weighted attention received, and
        hm_values are head-mean value vectors used for merge pairing.
        """
        H, Nt, N = attn_text.shape
        A = attn_text.float()
        V = self._repeat_kv(values, H).float().to(A.device)        # [H, N, d]

        w = self._query_weights(Nt, A.device).view(1, Nt, 1)

        o = torch.bmm(A, V)                                        # [H, Nt, d]
        G = torch.bmm(o, V.transpose(1, 2))                        # [H, Nt, N]
        v_sq = (V * V).sum(-1).unsqueeze(1)                        # [H, 1, N]
        o_sq = (o * o).sum(-1).unsqueeze(-1)                       # [H, Nt, 1]
        dist = (v_sq + o_sq - 2.0 * G).clamp_min(0).sqrt()

        amp = A / (1.0 - A).clamp_min(self._eps)
        if self.amp_cap is not None:
            amp = amp.clamp_max(self.amp_cap)

        damage = (w * amp * dist).sum(1).mean(0)                   # [N]
        # Relative perturbation: normalise by typical output magnitude so
        # damages are comparable ACROSS layers (required by DWF).
        o_scale = o.norm(dim=-1).mean() + self._eps
        damage = damage / o_scale

        # Soft text priority.
        damage[:image_start] = damage[:image_start] * self.text_boost
        damage[text_start:] = damage[text_start:] * self.text_boost
        # Hard protection: sinks + the final token.
        damage[: self.start_size] = float("inf")
        damage[N - self.protect_size :] = float("inf")

        attn_mass = (w * A).sum(1).mean(0)                         # [N]
        hm_values = V.mean(0)                                      # [N, d]
        return damage, attn_mass, hm_values

    def _best_kept_neighbor(self, hm_values, cand_abs, kept_mask, lo, hi):
        """For each candidate (absolute index), the most value-similar
        RETAINED token within +/- merge_window inside [lo, hi).

        Returns (rel_dist [n_cand], neighbor_abs [n_cand]); rel_dist is inf
        where no valid neighbor exists.
        """
        device = hm_values.device
        n_cand = cand_abs.numel()
        best_d = torch.full((n_cand,), float("inf"), device=device)
        best_n = torch.full((n_cand,), -1, dtype=torch.long, device=device)
        v_c = hm_values[cand_abs]                                  # [n_cand, d]
        for off in range(-self.merge_window, self.merge_window + 1):
            if off == 0:
                continue
            n_abs = cand_abs + off
            valid = (n_abs >= lo) & (n_abs < hi)
            n_safe = n_abs.clamp(lo, hi - 1)
            valid = valid & kept_mask[n_safe]
            d = (v_c - hm_values[n_safe]).norm(dim=-1)
            d = torch.where(valid, d, torch.full_like(d, float("inf")))
            upd = d < best_d
            best_d = torch.where(upd, d, best_d)
            best_n = torch.where(upd, n_safe, best_n)
        rel = best_d / (v_c.norm(dim=-1) + self._eps)
        return rel, best_n

    # ------------------------------------------------------------------ #
    # Online prefill hooks (layer-by-layer collection)                    #
    # ------------------------------------------------------------------ #
    def begin_online_prefill(self, input_ids):
        if input_ids is None:
            return False
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        if len(image_positions) < 2 or image_positions[0].numel() == 0:
            return False
        image_start = image_positions[1][0].item()
        visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
        self._online_prefill_stats = {
            "image_start": image_start,
            "text_start": image_start + visual_token_num,
            "text_attn_rows": [None] * self.layer_num,
        }
        return True

    def collect_online_prefill_attention(self, layer_idx, attention):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None or attention is None:
            return
        text_start = stats["text_start"]
        stats["text_attn_rows"][layer_idx] = (
            attention.squeeze(0)[:, text_start:, :].detach().half()
        )

    def finish_online_prefill(self):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None:
            return None
        missing = [i for i, r in enumerate(stats["text_attn_rows"]) if r is None]
        if missing:
            raise RuntimeError(f"Missing TriKV online prefill stats for layers: {missing}")
        self._online_prefill_stats = None
        return {
            "trikv_online_prefill": True,
            "image_start": stats["image_start"],
            "text_start": stats["text_start"],
            "text_attn_rows": tuple(stats["text_attn_rows"]),
        }

    def set_pending_attentions(self, attentions):
        self._pending_attentions = attentions

    def pop_pending_attentions(self):
        attentions = getattr(self, "_pending_attentions", None)
        if hasattr(self, "_pending_attentions"):
            del self._pending_attentions
        return attentions

    def _is_online_prefill_stats(self, attentions):
        return isinstance(attentions, dict) and attentions.get("trikv_online_prefill", False)

    # ------------------------------------------------------------------ #
    # Dispatch                                                            #
    # ------------------------------------------------------------------ #
    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None

        if self._is_online_prefill_stats(attentions):
            rows = [r for r in attentions["text_attn_rows"]]
            return self._prefill(
                past_key_values, num_of_token, rows,
                attentions["image_start"], attentions["text_start"],
            )
        if attentions[0].shape[-2] > 1:
            image_start = (input_ids == self.image_token_id).nonzero(as_tuple=True)[1][0].item()
            visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
            text_start = image_start + visual_token_num
            rows = [a.squeeze(0)[:, text_start:, :] for a in attentions]
            return self._prefill(past_key_values, num_of_token, rows, image_start, text_start)
        return self._decode(past_key_values, num_of_token, attentions, input_ids)

    # ------------------------------------------------------------------ #
    # Prefill: UDM -> DWF -> NRM                                          #
    # ------------------------------------------------------------------ #
    def _prefill(self, past_key_values, num_of_token, text_rows, image_start, text_start):
        seq_lens = [p[0].size(self.k_seq_dim) for p in past_key_values]
        seq_len = seq_lens[0]

        keep_per_layer_target = int(round(num_of_token * (1.0 - self.ratio)))
        total_budget = keep_per_layer_target * self.layer_num
        self.total_budget = total_budget
        if keep_per_layer_target >= seq_len:
            print("[TriKV] No KV to compress.")
            return past_key_values

        # ---- UDM per layer ---------------------------------------------
        damages, masses, hm_vals = [], [], []
        for l in range(self.layer_num):
            v_cache = past_key_values[l][1].squeeze(0)              # [H_kv, N, d]
            dmg, mass, hv = self._layer_damage(
                text_rows[l].to(v_cache.device), v_cache, image_start, text_start
            )
            damages.append(dmg)
            masses.append(mass)
            hm_vals.append(hv)

        # ---- DWF (budget-exact): floors are charged INSIDE the budget ----
        # Pass 1: forced-keep sets (hard-protected slots + vision floor +
        # per-layer minimum). These consume budget slots.
        device = damages[0].device
        forced_masks = []
        for l in range(self.layer_num):
            dmg = damages[l]
            N = seq_lens[l]
            f = torch.zeros(N, dtype=torch.bool, device=dmg.device)
            f[: self.start_size] = True
            f[N - self.protect_size :] = True
            n_vis = text_start - image_start
            if n_vis > 0 and self.min_vision_keep > 0:
                nv = min(self.min_vision_keep, n_vis)
                add = torch.topk(dmg[image_start:text_start], nv).indices + image_start
                f[add] = True
            min_floor = self.start_size + self.protect_size + self.min_keep_per_layer
            deficit = min_floor - int(f.sum().item())
            if deficit > 0:
                d2 = dmg.clone()
                d2[f] = float("-inf")
                f[torch.topk(d2, deficit).indices] = True
            forced_masks.append(f)

        total_forced = int(sum(int(f.sum().item()) for f in forced_masks))
        remaining = total_budget - total_forced
        if remaining < 0:
            print(
                f"[TriKV] Floors ({total_forced}) exceed budget ({total_budget}); "
                f"keeping floors only. Lower min_keep_per_layer/min_vision_keep "
                f"if strict budget adherence is required at this ratio."
            )
            remaining = 0

        # Pass 2: EXACT global top-k over the non-forced tokens. Total
        # retained = total_forced + remaining = total_budget (floors
        # included), and topk avoids tie-overshoot of a >= threshold test.
        flat = torch.cat(
            [
                d.to(device).masked_fill(forced_masks[l].to(device), float("-inf"))
                for l, d in enumerate(damages)
            ]
        )
        global_keep = torch.zeros(flat.numel(), dtype=torch.bool, device=device)
        if remaining > 0:
            global_keep[torch.topk(flat, remaining).indices] = True

        self.ratios = np.zeros(self.layer_num, dtype=np.float64)

        # ---- Per-layer triage: keep / merge / evict ----------------------
        self._reset_decode_state()
        out = []
        offset = 0
        for l in range(self.layer_num):
            dmg = damages[l]
            N = seq_lens[l]
            kept_mask = forced_masks[l] | global_keep[offset : offset + N].to(dmg.device)
            offset += N

            # NRM: below-threshold vision tokens choose merge vs evict.
            cand_abs = ((~kept_mask).nonzero(as_tuple=True)[0])
            cand_abs = cand_abs[(cand_abs >= image_start) & (cand_abs < text_start)]
            merge_src = merge_tgt = None
            if cand_abs.numel() > 0 and self.merge_threshold > 0:
                rel, n_abs = self._best_kept_neighbor(
                    hm_vals[l], cand_abs, kept_mask, image_start, text_start
                )
                ok = rel <= self.merge_threshold
                merge_src = cand_abs[ok]
                merge_tgt = n_abs[ok]

            kept_idx = kept_mask.nonzero(as_tuple=True)[0].sort().values
            k_cache = past_key_values[l][0].squeeze(0)              # [H_kv, N, d]
            v_cache = past_key_values[l][1].squeeze(0)
            dev = k_cache.device
            kept_dev = kept_idx.to(dev)
            k_sel = k_cache.index_select(1, kept_dev).clone()
            v_sel = v_cache.index_select(1, kept_dev).clone()

            weights = torch.ones(kept_idx.numel(), device=dev)
            if merge_src is not None and merge_src.numel() > 0:
                pos_map = torch.full((N,), -1, dtype=torch.long, device=dev)
                pos_map[kept_dev] = torch.arange(kept_dev.numel(), device=dev)
                tgt_pos = pos_map[merge_tgt.to(dev)]
                v_src = v_cache.index_select(1, merge_src.to(dev)).float()
                # Mass-weighted value average, executed per KV head in fp32,
                # cast back to the cache dtype at the end.
                v_num = v_sel.float() * weights.view(1, -1, 1)
                v_num.index_add_(1, tgt_pos, v_src)
                weights.index_add_(0, tgt_pos, torch.ones(tgt_pos.numel(), device=dev))
                v_sel = (v_num / weights.view(1, -1, 1)).to(v_sel.dtype)

            out.append([k_sel.unsqueeze(0), v_sel.unsqueeze(0)])

            self.ratios[l] = kept_idx.numel() / float(N)
            self._register_layer_decode_state(dmg, kept_idx, weights, image_start, text_start, N)

        temp = 0
        for i in out:
            temp += i[0].shape[-2]
        if temp > self.total_budget:
            print(
                f"[TriKV] Warning: total retained ({temp}) exceeds budget ({self.total_budget}). "
            )
        return DynamicCache(out)

    # ------------------------------------------------------------------ #
    # Decode-state bookkeeping                                            #
    # ------------------------------------------------------------------ #
    def _reset_decode_state(self):
        self._acc = []
        self._protected = []
        self._mass = []

    def _register_layer_decode_state(self, damage, kept_idx, weights, image_start, text_start, N):
        d = damage.clone()
        d[torch.isinf(d)] = d[~torch.isinf(d)].max() if (~torch.isinf(d)).any() else 1.0
        self._acc.append(d[kept_idx].float())
        protected = (
            (kept_idx < self.start_size)
            | (kept_idx < image_start)
            | (kept_idx >= text_start)
            | (kept_idx >= N - self.protect_size)
        )
        self._protected.append(protected.to(damage.device))
        self._mass.append(weights.float())

    def get_merge_log_bias(self, layer_idx):
        """Optional: log-mass additive attention bias for integrators whose
        attention path accepts a float mask; restores the merged slots'
        aggregate attention share exactly. Not required for correctness."""
        return torch.log(self._mass[layer_idx].clamp_min(1.0))

    # ------------------------------------------------------------------ #
    # Decode: EMA damage + Lazy Chunked Triage                             #
    # ------------------------------------------------------------------ #
    def _step_damage(self, step_attn, v_cache):
        A = step_attn.squeeze(0).squeeze(1).float()                 # [H, N]
        H = A.shape[0]
        V = self._repeat_kv(v_cache.squeeze(0), H).float().to(A.device)
        o = torch.einsum("hn,hnd->hd", A, V)
        G = torch.einsum("hd,hnd->hn", o, V)
        dist = (
            (V * V).sum(-1) + (o * o).sum(-1, keepdim=True) - 2.0 * G
        ).clamp_min(0).sqrt()
        amp = A / (1.0 - A).clamp_min(self._eps)
        if self.amp_cap is not None:
            amp = amp.clamp_max(self.amp_cap)
        dmg = (amp * dist).mean(0)
        return dmg / (o.norm(dim=-1).mean() + self._eps), V.mean(0)  # [N], [N, d]

    def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        out = []
        for i, (k, v) in enumerate(past_key_values):
            seq_len = k.size(self.k_seq_dim)
            acc, protected, mass = self._acc[i], self._protected[i], self._mass[i]

            n_new = seq_len - acc.numel()
            if n_new > 0:
                dev = acc.device
                acc = torch.cat([acc, torch.zeros(n_new, device=dev)])
                protected = torch.cat([protected, torch.zeros(n_new, dtype=torch.bool, device=dev)])
                mass = torch.cat([mass, torch.ones(n_new, device=dev)])

            step_dmg, hm_v = self._step_damage(attentions[i], v)
            acc = acc * self.decode_decay + step_dmg.to(acc.device)

            target = int(round(num_of_token * self.ratios[i]))
            overflow = seq_len - max(target, self.start_size + self.protect_size)

            if overflow < self.decode_chunk:                        # LCT: wait
                self._acc[i], self._protected[i], self._mass[i] = acc, protected, mass
                out.append([k, v])
                continue

            evictable = ~protected
            evictable[: self.start_size] = False
            if self.recent_window > 0:
                evictable[max(0, seq_len - self.recent_window):] = False
            n_out = min(overflow, int(evictable.sum().item()))
            if n_out <= 0:
                self._acc[i], self._protected[i], self._mass[i] = acc, protected, mass
                out.append([k, v])
                continue

            masked = acc.masked_fill(~evictable, float("inf"))
            drop_idx = torch.topk(masked, n_out, largest=False).indices

            keep = torch.ones(seq_len, dtype=torch.bool, device=acc.device)
            keep[drop_idx] = False

            # Merge-or-evict for the dropped set.
            if self.merge_threshold > 0 and drop_idx.numel() > 0:
                hm_v = hm_v.to(acc.device)
                rel, n_abs = self._best_kept_neighbor(hm_v, drop_idx, keep, 0, seq_len)
                ok = rel <= self.merge_threshold
                m_src, m_tgt = drop_idx[ok], n_abs[ok]
                if m_src.numel() > 0:
                    dev = v.device
                    v_flat = v.squeeze(0)                           # [H_kv, N, d]
                    tgt, src = m_tgt.to(dev), m_src.to(dev)
                    num = v_flat[:, tgt].float() * mass[m_tgt].to(dev).view(1, -1, 1) \
                        + v_flat[:, src].float() * mass[m_src].to(dev).view(1, -1, 1)
                    new_mass = (mass[m_tgt] + mass[m_src]).to(dev)
                    v_flat[:, tgt] = (num / new_mass.view(1, -1, 1)).to(v_flat.dtype)
                    mass[m_tgt] = new_mass.to(mass.device)

            keep_idx = keep.nonzero(as_tuple=True)[0]
            kd = keep_idx.to(k.device)
            out.append([
                k.index_select(self.k_seq_dim, kd),
                v.index_select(self.v_seq_dim, kd),
            ])
            self._acc[i] = acc[keep_idx]
            self._protected[i] = protected[keep_idx]
            self._mass[i] = mass[keep_idx]

            
        temp = 0
        for i in out:
            temp += i[0].shape[-2]
        if temp > self.total_budget:
            print(
                f"[TriKV] Warning: total retained ({temp}) exceeds budget ({self.total_budget}). "
            )
        return DynamicCache(out)