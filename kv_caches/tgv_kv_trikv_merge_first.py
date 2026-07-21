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
        keep-costs are pooled and a single global selection retains the most
        expensive tokens until the total budget is spent. Layer budgets
        emerge from equalising marginal damage across layers (the KKT
        condition of budgeted damage minimisation).

        The budget is a HARD constraint: total retained <= total_budget,
        always. Floors are tiered preferences granted in priority order
        only while budget remains:
          tier 0 (structural): sinks + final token  -- trimmed only if the
                  budget is below the structural minimum (loud warning);
          tier 1: min_vision_keep vision representatives per layer;
          tier 2: min_keep_per_layer top-ups.
        Tier-1/2 grants are filled globally by damage, so under tight
        budgets the floors degrade gracefully instead of overshooting.

    (3) NRM -- Neighbor-Restricted Merging. A below-threshold vision token
        is merged into the most value-similar RETAINED token within a small
        raster-order window (mass-weighted value average, per-slot mass
        tracked), if the relative value distance is under `merge_threshold`;
        otherwise it is evicted. Neighbor restriction keeps the search
        linear and keeps RoPE key mismatch negligible. The cache stays
        physically short, so decode speed is fully preserved.

    (4) LCT -- Lazy Chunked Triage. During decode, per-token damage is
        accumulated with an EMA of the same counterfactual quantity. The
        cache may drift `decode_chunk` tokens above its target, then is
        compressed back in one batch (merge-or-evict). With
        `fixed_budget=True` (default) the per-layer target is pinned to the
        prefill allocation, so the total cache never exceeds total_budget
        during generation; with `fixed_budget=False` the target grows
        proportionally with generated length (TGV-KV-style).

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
        text_boost=1.0,        # damage multiplier for text KVs (soft TPR)
        last_query_boost=10.0,  # extra weight for the final text queries
        n_last_queries=4,
        amp_cap=50.0,          # ceiling on A/(1-A)
        min_keep_per_layer=8,  # DWF tier-2 floor (on top of sinks + protect)
        min_vision_keep=4,     # DWF tier-1 floor: vision reps per layer
        decode_chunk=1,        # LCT: compress once we exceed target by this
        decode_decay=0.95,     # EMA decay of accumulated decode damage
        recent_window=8,       # newest tokens never evicted during decode
        fixed_budget=True,     # pin decode target to the prefill allocation
        merge_first=True,      # consolidate duplicates BEFORE selection
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
        self.fixed_budget = fixed_budget
        self.merge_first = merge_first
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

    def _check_budget(self, out, phase):
        temp = 0
        for i in out:
            temp += i[0].shape[-2]
        if temp > self.total_budget:
            print(
                f"[TriKV] Warning ({phase}): total retained ({temp}) exceeds "
                f"budget ({self.total_budget})."
            )
        return temp

    def _layer_damage(self, attn_text, values, image_start, text_start):
        """UDM for one layer.

        attn_text : [H, Nt, N] post-softmax rows of the text queries
        values    : [H_kv, N, d]
        Returns (keep_cost [N], hm_values [N, d], o [H, Nt, d], o_scale)
        where keep_cost is the normalised eviction damage (inf on
        hard-protected slots), hm_values are head-mean value vectors used
        for clustering / merge pairing, and (o, o_scale) are the text-query
        blends and their mean norm, reused by cluster valuation.
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

        hm_values = V.mean(0)                                      # [N, d]
        return damage, hm_values, o, o_scale

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

    def _consolidate_vision(self, dmg, attn_text, values, hm, image_start, text_start, o, o_scale):
        """Merge-first consolidation (runs BEFORE selection).

        Chains raster-adjacent vision tokens whose head-mean values differ
        by less than `merge_threshold` (relative) into clusters. Each
        cluster is re-valued with its AGGREGATED counterfactual damage:

            D_c = sum_i w_i * Ac/(1-Ac) * ||v_bar_c - o_i||,
            Ac = sum_{j in c} A_ij,   v_bar_c = mean value of the cluster,

        so a group of duplicates bids for a keep slot with its collective
        importance instead of each member individually looking cheap
        (the group-blindness of leave-one-out scoring). Members other than
        the representative are barred from selection (-inf), so a cluster
        can never occupy more than one budget slot.

        Returns (adjusted_damage [N], info) where info carries the cluster
        assignment for value fusion after selection, or (dmg, None) when
        nothing clusters.
        """
        n_vis = text_start - image_start
        if n_vis <= 1 or self.merge_threshold <= 0:
            return dmg, None
        device = dmg.device

        hm_vis = hm[image_start:text_start].to(device)             # [n_vis, d]
        # adjacent-pair chaining: token j joins its left neighbor's cluster
        # if their values are near-identical (relative distance <= delta)
        rel = (hm_vis[1:] - hm_vis[:-1]).norm(dim=-1) / (hm_vis[1:].norm(dim=-1) + self._eps)
        new_cluster = torch.ones(n_vis, dtype=torch.bool, device=device)
        new_cluster[1:] = rel > self.merge_threshold
        cluster_id = torch.cumsum(new_cluster.long(), 0) - 1       # [n_vis]
        C = int(cluster_id[-1].item()) + 1
        sizes = torch.bincount(cluster_id, minlength=C).to(device) # [C]
        if int((sizes > 1).sum().item()) == 0:
            return dmg, None                                        # all singletons

        # representative of each cluster = member with highest per-token
        # damage (tie -> earliest), so the retained key sits on the most
        # attended member and RoPE mismatch of the fused value is minimal.
        idx = torch.arange(n_vis, device=device)
        score = dmg[image_start:text_start].float() - idx.float() * 1e-9
        max_s = torch.full((C,), float("-inf"), device=device)
        max_s.scatter_reduce_(0, cluster_id, score, reduce="amax", include_self=True)
        is_rep = score >= max_s[cluster_id]
        rep_local = torch.zeros(C, dtype=torch.long, device=device)
        rep_local[cluster_id[is_rep]] = idx[is_rep]
        rep_pos = rep_local + image_start                           # [C] global

        # aggregated cluster damage
        A = attn_text.float().to(device)                            # [H, Nt, N]
        H, Nt, _ = A.shape
        Vh = self._repeat_kv(values, H).float().to(device)          # [H, N, d]
        w = self._query_weights(Nt, device).view(1, Nt, 1)

        A_v = A[:, :, image_start:text_start]                       # [H, Nt, n_vis]
        A_c = torch.zeros(H, Nt, C, device=device)
        A_c.index_add_(2, cluster_id, A_v)                          # summed attention
        Vv = Vh[:, image_start:text_start, :]                       # [H, n_vis, d]
        v_sum = torch.zeros(H, C, Vv.shape[-1], device=device)
        v_sum.index_add_(1, cluster_id, Vv)
        v_bar = v_sum / sizes.view(1, -1, 1).float()                # [H, C, d]

        G = torch.bmm(o.to(device), v_bar.transpose(1, 2))          # [H, Nt, C]
        v_sq = (v_bar * v_bar).sum(-1).unsqueeze(1)                 # [H, 1, C]
        o_sq = (o.to(device) * o.to(device)).sum(-1).unsqueeze(-1)  # [H, Nt, 1]
        dist = (v_sq + o_sq - 2.0 * G).clamp_min(0).sqrt()
        amp = A_c / (1.0 - A_c).clamp_min(self._eps)
        if self.amp_cap is not None:
            amp = amp.clamp_max(self.amp_cap)
        dmg_c = (w * amp * dist).sum(1).mean(0) / o_scale           # [C]

        adjusted = dmg.clone()
        adjusted[image_start:text_start] = float("-inf")            # members barred
        adjusted[rep_pos] = dmg_c.to(adjusted.dtype)                # reps carry cluster value

        info = {"cluster_id": cluster_id, "rep_pos": rep_pos,
                "sizes": sizes, "image_start": image_start}
        return adjusted, info

    def _cluster_merge_lists(self, info, kept_mask):
        """After selection: members of KEPT clusters fuse into their
        representative; members of dropped clusters are simply evicted."""
        cluster_id, rep_pos = info["cluster_id"], info["rep_pos"]
        image_start = info["image_start"]
        device = kept_mask.device
        vis_pos = torch.arange(cluster_id.numel(), device=device) + image_start
        kept_rep = kept_mask[rep_pos.to(device)]                    # [C]
        cid = cluster_id.to(device)
        src_mask = kept_rep[cid] & (vis_pos != rep_pos.to(device)[cid])
        merge_src = vis_pos[src_mask]
        merge_tgt = rep_pos.to(device)[cid[src_mask]]
        return merge_src, merge_tgt

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
    # Budget-respecting tiered floors                                     #
    # ------------------------------------------------------------------ #
    def _allocate_forced(self, damages, seq_lens, image_start, text_start, total_budget):
        """Build per-layer forced-keep masks such that
        sum(forced) <= total_budget ALWAYS.

        Tier 0: sinks + final token (structural; trimmed only if the budget
                is below the structural minimum, with a loud warning).
        Tier 1: vision representatives (min_vision_keep per layer).
        Tier 2: min_keep_per_layer top-ups.
        Tiers 1 and 2 are granted globally by descending damage while
        budget remains.

        Returns (forced_masks, remaining) with
        remaining = total_budget - sum(forced) >= 0.
        """
        L = self.layer_num
        n_vis = text_start - image_start
        structural_per_layer = self.start_size + self.protect_size
        n_struct = structural_per_layer * L

        # ---- pathological case: budget below the structural minimum -----
        if n_struct > total_budget:
            print(
                f"[TriKV] Warning: budget ({total_budget}) is below the "
                f"structural minimum ({n_struct} = (sinks+final) x layers). "
                f"Trimming structural protections to fit; quality will be "
                f"severely degraded. Increase the budget or reduce start_size."
            )
            quota = total_budget
            forced_masks = []
            for l in range(L):
                N = seq_lens[l]
                f = torch.zeros(N, dtype=torch.bool, device=damages[l].device)
                slots = list(range(self.start_size)) + list(range(N - self.protect_size, N))
                for s in slots[: max(quota, 0)]:
                    f[s] = True
                quota -= int(f.sum().item())
                forced_masks.append(f)
            return forced_masks, 0

        # ---- tier 0 ------------------------------------------------------
        forced_masks = []
        tier1_lay, tier1_idx, tier1_dmg = [], [], []
        tier2_lay, tier2_idx, tier2_dmg = [], [], []
        for l in range(L):
            dmg = damages[l]
            N = seq_lens[l]
            f = torch.zeros(N, dtype=torch.bool, device=dmg.device)
            f[: self.start_size] = True
            f[N - self.protect_size :] = True
            forced_masks.append(f)

            sel = f.clone()
            # tier-1 candidates: top vision tokens by damage
            if n_vis > 0 and self.min_vision_keep > 0:
                nv = min(self.min_vision_keep, n_vis)
                cand = torch.topk(dmg[image_start:text_start], nv).indices + image_start
                tier1_lay.append(torch.full((nv,), l, dtype=torch.long))
                tier1_idx.append(cand.cpu())
                tier1_dmg.append(dmg[cand].detach().float().cpu())
                sel[cand] = True
            # tier-2 candidates: top-up to the per-layer minimum
            min_floor = structural_per_layer + self.min_keep_per_layer
            deficit = min_floor - int(sel.sum().item())
            if deficit > 0:
                d2 = dmg.clone()
                d2[sel] = float("-inf")
                cand2 = torch.topk(d2, deficit).indices
                tier2_lay.append(torch.full((deficit,), l, dtype=torch.long))
                tier2_idx.append(cand2.cpu())
                tier2_dmg.append(dmg[cand2].detach().float().cpu())

        remaining = total_budget - n_struct

        # ---- grant tiers 1 then 2, globally by damage, within budget -----
        for lays, idxs, dmgs_t in (
            (tier1_lay, tier1_idx, tier1_dmg),
            (tier2_lay, tier2_idx, tier2_dmg),
        ):
            if remaining <= 0 or len(lays) == 0:
                continue
            lay = torch.cat(lays)
            idx = torch.cat(idxs)
            dmg_t = torch.cat(dmgs_t)
            take = min(int(dmg_t.numel()), remaining)
            if take <= 0:
                continue
            granted = torch.topk(dmg_t, take).indices
            g_lay, g_idx = lay[granted], idx[granted]
            for l in range(L):
                sel_l = g_idx[g_lay == l]
                if sel_l.numel() > 0:
                    forced_masks[l][sel_l.to(forced_masks[l].device)] = True
            remaining -= take

        return forced_masks, max(remaining, 0)

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

        # ---- UDM per layer (+ merge-first consolidation) -----------------
        damages, hm_vals, cluster_infos = [], [], []
        for l in range(self.layer_num):
            v_cache = past_key_values[l][1].squeeze(0)              # [H_kv, N, d]
            rows_l = text_rows[l].to(v_cache.device)
            dmg, hv, o, o_scale = self._layer_damage(
                rows_l, v_cache, image_start, text_start
            )
            cinfo = None
            if self.merge_first:
                dmg, cinfo = self._consolidate_vision(
                    dmg, rows_l, v_cache, hv, image_start, text_start, o, o_scale
                )
            damages.append(dmg)
            hm_vals.append(hv)
            cluster_infos.append(cinfo)

        # ---- DWF with hard budget: tiered, budget-respecting floors ------
        device = damages[0].device
        forced_masks, remaining = self._allocate_forced(
            damages, seq_lens, image_start, text_start, total_budget
        )

        # ---- EXACT global top-k over the non-forced tokens. Total
        # retained = sum(forced) + remaining <= total_budget always. -------
        flat = torch.cat(
            [
                d.to(device).masked_fill(forced_masks[l].to(device), float("-inf"))
                for l, d in enumerate(damages)
            ]
        )
        global_keep = torch.zeros(flat.numel(), dtype=torch.bool, device=device)
        # never let top-k spill into -inf entries (barred cluster members /
        # already-forced slots) when the budget exceeds the finite pool
        remaining = min(remaining, int(torch.isfinite(flat).sum().item()))
        if remaining > 0:
            global_keep[torch.topk(flat, remaining).indices] = True

        self.ratios = np.zeros(self.layer_num, dtype=np.float64)
        self._prefill_kept = np.zeros(self.layer_num, dtype=np.int64)

        # ---- Per-layer triage: keep / merge / evict ----------------------
        self._reset_decode_state()
        out = []
        offset = 0
        for l in range(self.layer_num):
            dmg = damages[l]
            N = seq_lens[l]
            kept_mask = forced_masks[l] | global_keep[offset : offset + N].to(dmg.device)
            offset += N

            # Merge lists.
            # merge_first: members of KEPT clusters fuse into their
            #              representative; dropped clusters are evicted whole.
            # legacy:      below-threshold vision tokens seek a retained twin
            #              AFTER selection (original NRM).
            merge_src = merge_tgt = None
            if cluster_infos[l] is not None:
                merge_src, merge_tgt = self._cluster_merge_lists(cluster_infos[l], kept_mask)
            elif not self.merge_first:
                cand_abs = ((~kept_mask).nonzero(as_tuple=True)[0])
                cand_abs = cand_abs[(cand_abs >= image_start) & (cand_abs < text_start)]
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
            self._prefill_kept[l] = kept_idx.numel()
            self._register_layer_decode_state(dmg, kept_idx, weights, image_start, text_start, N)

        self._check_budget(out, "prefill")
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

            if self.fixed_budget and hasattr(self, "_prefill_kept"):
                # Hard budget: the cache never grows past its prefill
                # allocation, so sum over layers stays <= total_budget.
                target = int(self._prefill_kept[i])
            else:
                target = int(round(num_of_token * self.ratios[i]))
            overflow = seq_len - max(target, self.start_size + self.protect_size)

            if overflow < self.decode_chunk:                        # LCT: wait
                self._acc[i], self._protected[i], self._mass[i] = acc, protected, mass
                out.append([k, v])
                continue

            # Tiered eviction priority (hard budget): prefer normal tokens,
            # then soft-protected (instruction text / vision floor), then the
            # recent window. Only sinks and the current final token are
            # absolutely untouchable.
            structural = torch.zeros_like(protected)
            structural[: self.start_size] = True
            structural[seq_len - self.protect_size :] = True

            prio = acc.clone()
            prio = prio + protected.float() * 1e6          # tier B: soft-protected
            if self.recent_window > 0:
                recent = torch.zeros_like(protected)
                recent[max(0, seq_len - self.recent_window):] = True
                prio = prio + recent.float() * 1e12        # tier C: newest tokens
            prio = prio.masked_fill(structural, float("inf"))

            n_out = min(overflow, int((~structural).sum().item()))
            if n_out <= 0:
                self._acc[i], self._protected[i], self._mass[i] = acc, protected, mass
                out.append([k, v])
                continue

            drop_idx = torch.topk(prio, n_out, largest=False).indices

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

        self._check_budget(out, "decode")
        return DynamicCache(out)