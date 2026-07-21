import numpy as np
import torch
from colorama import Fore
from transformers.cache_utils import DynamicCache


class TGVKVCache:
    """TGV-KV with CPR: Counterfactual Perturbation Ranking.

    CPR replaces TWR's attention-sum vision score with the EXACT change in
    the attention output of the text queries caused by evicting a KV pair.

    Removing key j renormalises the softmax over the survivors, so the new
    output of text query i is  o'_i = (o_i - A_ij * v_j) / (1 - A_ij),
    which gives a closed-form, per-token counterfactual perturbation:

        || Delta_ij || = A_ij / (1 - A_ij) * || v_j - o_i ||

    The vision importance of KV j is the dominant-text-weighted sum of this
    perturbation over all text queries (weights w_i follow the paper's
    Eq. 7, so the score stays text-grounded):

        s_j = sum_i  w_i * A_ij / (1 - A_ij) * || v_j - o_i ||

    Compared to attention-sum ranking, CPR is
      * value-aware:      a token whose value barely differs from what the
                          query already aggregates (o_i) scores ~0 even at
                          high attention;
      * redundancy-aware: near-duplicate vision patches have values close to
                          the aggregate output, so their marginal removal is
                          correctly judged harmless -- no explicit similarity
                          or diversity machinery needed;
      * renormalisation-aware: the 1/(1-A) factor accounts for softmax mass
                          redistribution after eviction.

    Because leave-one-out underestimates GROUPS of duplicates (each looks
    individually harmless), eviction proceeds in `cascade_rounds` rounds:
    after round 1 evicts set E, outputs are updated in closed form
        o'_i = (o_i - sum_{j in E} A_ij v_j) / (1 - m_i),   m_i = sum_E A_ij
    and scores are recomputed against o'. Once one duplicate is evicted, the
    surviving duplicate's value deviates strongly from the updated aggregate
    and it becomes protected -- diversity emerges from the counterfactual
    itself, with no extra forward pass.

    The decode phase applies the SAME principle: each step's single-query
    attention row and the cached V give an instantaneous perturbation score,
    accumulated with an EMA; eviction removes the KV whose removal would
    damage the output least. Protection of sink + instruction-text KVs is
    tracked with explicit index masks pruned in sync with the cache (fixing
    the suffix-drift bug of the original release, where the protected suffix
    slid onto generated tokens as decoding progressed).

    TVB (layer budgets, Eq. 5-6) and TPR (text-first retention via the +100
    offset) are kept exactly as in the paper, so CPR is an isolated,
    ablatable replacement of the ranking component only.

    Note: eviction bookkeeping assumes batch_size == 1 (as the original
    `_decode` implicitly did via `.squeeze(0)`).
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
        cascade_rounds=2,     # counterfactual refinement rounds (1 = plain LOO)
        cascade_frac=0.5,     # fraction of the eviction quota removed in round 1
        amp_cap=50.0,         # ceiling on A/(1-A); None = uncapped
        dist_power=1.0,       # exponent on ||v_j - o_i||; 0 = attention-only
        cpr_blend=0.7,        # 1 = pure CPR, 0 = pure attention-sum (TWR-like)
        decode_decay=0.95,    # EMA decay of accumulated decode perturbation
        recent_window=8,      # newest tokens never evicted during decode
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

        self.cascade_rounds = cascade_rounds
        self.cascade_frac = cascade_frac
        self.amp_cap = amp_cap
        self.dist_power = dist_power
        self.cpr_blend = cpr_blend
        self.decode_decay = decode_decay
        self.recent_window = recent_window
        self._eps = 1e-4

    # ------------------------------------------------------------------ #
    # CPR core                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _repeat_kv(values, n_heads):
        """[H_kv, N, d] -> [H, N, d] for GQA models."""
        h_kv = values.shape[0]
        if h_kv == n_heads:
            return values
        return values.repeat_interleave(n_heads // h_kv, dim=0)

    def _cpr_vision_scores(self, attn_text, values, image_start, text_start, text_w, evict_v):
        """Counterfactual Perturbation Ranking for one layer.

        attn_text : [H, Nt, N]  post-softmax attention rows of text queries
        values    : [H_kv, N, d] value cache of this layer
        text_w    : [Nt]        dominant-text weights (paper Eq. 7), any scale
        evict_v   : int         planned number of vision evictions (cascade)

        Returns vision scores [Nv] in [0, 1]; higher = more important.
        """
        H, Nt, _ = attn_text.shape
        A = attn_text.float()
        V = self._repeat_kv(values, H).float().to(A.device)       # [H, N, d]
        Nv = text_start - image_start

        o = torch.bmm(A, V)                                        # [H, Nt, d]
        Vv = V[:, image_start:text_start, :]                       # [H, Nv, d]
        A_tv = A[:, :, image_start:text_start]                     # [H, Nt, Nv]
        w = text_w.float().to(A.device)
        w = (w / (w.sum() + self._eps)).view(1, Nt, 1)

        v_sq = (Vv * Vv).sum(-1).unsqueeze(1)                      # [H, 1, Nv]

        def perturbation_scores(o_cur, removed_mass=None):
            # ||v_j - o_i||  via  ||v||^2 + ||o||^2 - 2 <o, v>
            G = torch.bmm(o_cur, Vv.transpose(1, 2))               # [H, Nt, Nv]
            o_sq = (o_cur * o_cur).sum(-1).unsqueeze(-1)           # [H, Nt, 1]
            dist = (v_sq + o_sq - 2.0 * G).clamp_min(0).sqrt()
            if self.dist_power != 1.0:
                dist = dist.clamp_min(self._eps) ** self.dist_power
            A_eff = A_tv
            if removed_mass is not None:                           # renormalised
                A_eff = A_tv / (1.0 - removed_mass).clamp_min(self._eps).unsqueeze(-1)
            amp = A_eff / (1.0 - A_eff).clamp_min(self._eps)
            if self.amp_cap is not None:
                amp = amp.clamp_max(self.amp_cap)
            return (w * amp * dist).sum(1).mean(0)                 # [Nv]

        def norm01(s):
            s = s - s.min()
            return s / (s.max() + self._eps)

        att_score = None
        if self.cpr_blend < 1.0:
            # TWR-style dominant-text-weighted attention sum, for blending.
            att_score = norm01((w * A_tv).sum(1).mean(0))

        def blended(s):
            s = norm01(s)
            if att_score is None:
                return s
            return self.cpr_blend * s + (1.0 - self.cpr_blend) * att_score

        score = blended(perturbation_scores(o))

        # ---- cascaded counterfactual refinement -----------------------
        n_rounds = max(int(self.cascade_rounds), 1)
        evict_v = int(min(max(evict_v, 0), Nv - 1))
        if n_rounds > 1 and evict_v > 1:
            r1 = int(evict_v * self.cascade_frac)
            if r1 > 0:
                evict1 = torch.topk(score, r1, largest=False).indices
                idx_abs = evict1 + image_start
                A_e = A[:, :, idx_abs]                             # [H, Nt, r1]
                mass = A_e.sum(-1)                                 # [H, Nt]
                o_upd = (o - torch.bmm(A_e, V[:, idx_abs, :])) / (
                    1.0 - mass
                ).clamp_min(self._eps).unsqueeze(-1)
                refreshed = blended(perturbation_scores(o_upd, removed_mass=mass))
                # Round-1 evictees are pinned strictly below every survivor,
                # preserving their relative order from the base score.
                floor = refreshed.min() - 1.0
                b = score[evict1]
                b = (b - b.min()) / (b.max() - b.min() + self._eps)
                refreshed[evict1] = floor + 1e-3 * b
                score = refreshed

        # Normalise to [0, 1] so the +100 text offset (TPR) always dominates.
        score = score - score.min()
        score = score / (score.max() + self._eps)
        return score

    def _cpr_step_scores(self, step_attn, k_cache, v_cache):
        """Per-step perturbation score during decode.

        step_attn : [B, H, 1, N] attention of the single new query
        v_cache   : [B, H_kv, N, d]
        Returns [N] scores; higher = more important to the current query.
        """
        A = step_attn.squeeze(0).squeeze(1).float()                # [H, N]
        H = A.shape[0]
        V = self._repeat_kv(v_cache.squeeze(0), H).float().to(A.device)  # [H, N, d]
        o = torch.einsum("hn,hnd->hd", A, V)                       # [H, d]
        G = torch.einsum("hd,hnd->hn", o, V)                       # [H, N]
        dist = (
            (V * V).sum(-1) + (o * o).sum(-1, keepdim=True) - 2.0 * G
        ).clamp_min(0).sqrt()                                      # [H, N]
        if self.dist_power != 1.0:
            dist = dist.clamp_min(self._eps) ** self.dist_power
        amp = A / (1.0 - A).clamp_min(self._eps)
        if self.amp_cap is not None:
            amp = amp.clamp_max(self.amp_cap)
        return (amp * dist).mean(0)                                # [N]

    # ------------------------------------------------------------------ #
    # Decode-state bookkeeping                                            #
    # ------------------------------------------------------------------ #
    def _reset_decode_state(self):
        self._acc_scores = []       # per-layer float tensor [cache_len]
        self._protected_masks = []  # per-layer bool tensor  [cache_len]
        self.initial_text_len_list = []  # kept for backward compatibility

    def _register_layer_decode_state(self, score_row, kept_idx, text_start, seq_len):
        kept_idx = kept_idx.to(score_row.device)
        acc = score_row.gather(0, kept_idx).float().clone()
        protected = (
            (kept_idx < self.start_size)
            | (kept_idx >= text_start)
            | (kept_idx >= seq_len - self.protect_size)
        )
        self._acc_scores.append(acc)
        self._protected_masks.append(protected)

    # ------------------------------------------------------------------ #
    # Online prefill hooks                                                #
    # ------------------------------------------------------------------ #
    def begin_online_prefill(self, input_ids):
        if input_ids is None:
            return False

        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
        if len(image_positions) < 2 or image_positions[0].numel() == 0:
            return False

        image_start = image_positions[1][0].item()
        visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
        text_start = image_start + visual_token_num

        self._online_prefill_stats = {
            "image_start": image_start,
            "visual_token_num": visual_token_num,
            "text_start": text_start,
            # Per-head text-query attention rows, stored in fp16.
            # Memory: H * Nt * N per layer -- cheap since Nt is small.
            "text_attn_rows": [None] * self.layer_num,
            "text_weights": [None] * self.layer_num,
            "pre_scores": [None] * self.layer_num,
            "post_scores": [None] * self.layer_num,
            "text_image_attn_sums": [None] * self.layer_num,
        }
        return True

    def collect_online_prefill_attention(self, layer_idx, attention):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None or attention is None:
            return

        image_start = stats["image_start"]
        visual_token_num = stats["visual_token_num"]
        text_start = stats["text_start"]

        attn = attention.squeeze(0)                     # [H, N, N]
        all_attn = attn.mean(0)                         # head-mean (TVB / TPR)

        # Dominant-text weights (paper Eq. 7) from head-mean TT attention.
        text_text_attns = all_attn[text_start:, text_start:]
        tt = text_text_attns.sum(0)
        b = torch.arange(1, tt.shape[-1] + 1).flip([0]).to(tt.device)
        stats["text_weights"][layer_idx] = (tt / b).detach()

        # Per-head text rows for exact CPR at finish time.
        stats["text_attn_rows"][layer_idx] = attn[:, text_start:, :].detach().half()

        text_image_attns = all_attn[text_start:, image_start:text_start]
        stats["text_image_attn_sums"][layer_idx] = text_image_attns.reshape(-1).sum()

        stats["pre_scores"][layer_idx] = all_attn[:, :image_start].sum(0, keepdim=True) + 100
        stats["post_scores"][layer_idx] = (
            all_attn[:, image_start + visual_token_num :].sum(0, keepdim=True) + 100
        )

    def finish_online_prefill(self):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None:
            return None

        missing_layers = [idx for idx, r in enumerate(stats["text_attn_rows"]) if r is None]
        if missing_layers:
            raise RuntimeError(f"Missing TGV-KV online prefill stats for layers: {missing_layers}")

        self._online_prefill_stats = None
        return {
            "tgv_kv_online_prefill": True,
            "image_start": stats["image_start"],
            "text_start": stats["text_start"],
            "text_attn_rows": tuple(stats["text_attn_rows"]),
            "text_weights": tuple(stats["text_weights"]),
            "pre_scores": tuple(stats["pre_scores"]),
            "post_scores": tuple(stats["post_scores"]),
            "text_image_attn_sums": tuple(stats["text_image_attn_sums"]),
        }

    def set_pending_attentions(self, attentions):
        self._pending_attentions = attentions

    def pop_pending_attentions(self):
        attentions = getattr(self, "_pending_attentions", None)
        if hasattr(self, "_pending_attentions"):
            del self._pending_attentions
        return attentions

    def _is_online_prefill_stats(self, attentions):
        return isinstance(attentions, dict) and attentions.get("tgv_kv_online_prefill", False)

    # ------------------------------------------------------------------ #
    # Dispatch                                                            #
    # ------------------------------------------------------------------ #
    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None

        if self._is_online_prefill_stats(attentions):
            self._reset_decode_state()
            return self._prefill_from_online_stats(past_key_values, num_of_token, attentions)
        if attentions[0].shape[-2] > 1:
            self._reset_decode_state()
            return self._prefill(past_key_values, num_of_token, attentions, input_ids)
        return self._decode(past_key_values, num_of_token, attentions, input_ids)

    # ------------------------------------------------------------------ #
    # Prefill                                                             #
    # ------------------------------------------------------------------ #
    def _prefill(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        image_start = (input_ids == self.image_token_id).nonzero(as_tuple=True)[1][0].item()
        visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
        text_start = image_start + visual_token_num

        # ---- TVB layer budgets (paper Eq. 5-6), unchanged --------------
        all_attns = torch.stack([x.squeeze(0).mean(0) for x in attentions])  # [L,N,N]
        text_image_attns = all_attns[:, text_start:, image_start:text_start]
        text_image_attn_sum = text_image_attns.reshape(text_image_attns.size(0), -1).sum(dim=1)
        normalized_layer_ratio = text_image_attn_sum / text_image_attn_sum.sum()
        layer_ratio = (
            seq_len
            - (len(normalized_layer_ratio) * seq_len * (1 - self.ratio) * normalized_layer_ratio)
        ) / seq_len
        self.ratios = layer_ratio.float().cpu().numpy()
        forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        # ---- dominant-text weights (paper Eq. 7), per layer -------------
        text_text_attns = all_attns[:, text_start:, text_start:]
        tt = text_text_attns.sum(1)                                          # [L, Nt]
        b = torch.arange(1, tt.shape[-1] + 1).flip([0]).to(tt.device).unsqueeze(0)
        text_weights = tt / b                                                # [L, Nt]

        # ---- CPR vision scores + TPR offsets, per layer -----------------
        Nv = text_start - image_start
        score_rows = []
        for idx in range(self.layer_num):
            attn_l = attentions[idx].squeeze(0)                              # [H, N, N]
            v_cache = past_key_values[idx][1].squeeze(0)                     # [H_kv, N, d]
            evict_v = int(min(forget_nums[idx], Nv - 1))
            vision_score = self._cpr_vision_scores(
                attn_l[:, text_start:, :],
                v_cache,
                image_start,
                text_start,
                text_weights[idx],
                evict_v,
            )                                                                # [Nv], [0,1]
            pre_score = all_attns[idx, :, :image_start].sum(0) + 100
            post_score = all_attns[idx, :, image_start + visual_token_num :].sum(0) + 100
            score_rows.append(
                torch.cat(
                    [
                        pre_score,
                        vision_score.to(pre_score.device, pre_score.dtype),
                        post_score,
                    ]
                )
            )
        score_sum = torch.stack(score_rows).unsqueeze(1)                     # [L, 1, N]

        return self._evict_prefill(past_key_values, score_sum, forget_nums, seq_lens, text_start)

    def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        image_start = attentions["image_start"]
        text_start = attentions["text_start"]

        text_image_attn_sum = torch.stack(list(attentions["text_image_attn_sums"]))
        normalized_layer_ratio = text_image_attn_sum / text_image_attn_sum.sum()
        layer_ratio = (
            seq_len
            - (len(normalized_layer_ratio) * seq_len * (1 - self.ratio) * normalized_layer_ratio)
        ) / seq_len
        self.ratios = layer_ratio.float().cpu().numpy()
        forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        Nv = text_start - image_start
        score_rows = []
        for idx in range(self.layer_num):
            v_cache = past_key_values[idx][1].squeeze(0)                     # [H_kv, N, d]
            evict_v = int(min(forget_nums[idx], Nv - 1))
            vision_score = self._cpr_vision_scores(
                attentions["text_attn_rows"][idx].to(v_cache.device),
                v_cache,
                image_start,
                text_start,
                attentions["text_weights"][idx],
                evict_v,
            )
            pre_score = attentions["pre_scores"][idx].squeeze(0)
            post_score = attentions["post_scores"][idx].squeeze(0)
            score_rows.append(
                torch.cat(
                    [
                        pre_score,
                        vision_score.to(pre_score.device, pre_score.dtype),
                        post_score,
                    ]
                )
            )
        score_sum = torch.stack(score_rows).unsqueeze(1)                     # [L, 1, N]

        return self._evict_prefill(past_key_values, score_sum, forget_nums, seq_lens, text_start)

    def _evict_prefill(self, past_key_values, score_sum, forget_nums, seq_lens, text_start):
        """Shared TPR eviction loop (identical to the paper's policy)."""
        past_key_values_return = []

        for idx in range(self.layer_num):
            forget_num = forget_nums[idx]
            seq_len = int(seq_lens[idx])
            selected_idx = (
                torch.argsort(score_sum[idx, :, self.start_size : (seq_len - self.protect_size)])[
                    :, forget_num:
                ]
                + self.start_size
            )
            selected_idx = selected_idx.sort().values

            device = selected_idx.device
            pre = torch.arange(self.start_size, device=device).unsqueeze(0).expand(self.batch_size, -1)
            post = (
                torch.tensor([seq_len - self.protect_size], device=device)
                .unsqueeze(0)
                .expand(self.batch_size, -1)
            )
            selected_idx = torch.cat([pre, selected_idx, post], dim=-1)
            self.initial_text_len_list.append(
                max((selected_idx[0] >= text_start).sum().item(), self.protect_size)
            )

            self._register_layer_decode_state(score_sum[idx, 0], selected_idx[0], text_start, seq_len)

            k, v = past_key_values[idx]
            selected_idx = selected_idx.to(k.device)

            k_select = k.gather(
                dim=-2,
                index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, k.shape[1], -1, k.shape[-1]),
            )
            v_select = v.gather(
                dim=-2,
                index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, v.shape[1], -1, v.shape[-1]),
            )

            past_key_values_return.append([k_select, v_select])

        return DynamicCache(past_key_values_return)

    # ------------------------------------------------------------------ #
    # Decode                                                              #
    # ------------------------------------------------------------------ #
    def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        # Lazy init if decode is called without a preceding prefill.
        if not hasattr(self, "_acc_scores") or len(self._acc_scores) != self.layer_num:
            self._acc_scores = [
                torch.zeros(int(n), device=past_key_values[i][0].device)
                for i, n in enumerate(seq_lens)
            ]
            self._protected_masks = []
            for i, n in enumerate(seq_lens):
                m = torch.zeros(int(n), dtype=torch.bool, device=past_key_values[i][0].device)
                m[: self.start_size] = True
                self._protected_masks.append(m)

        past_key_values_return = []
        for i, (k, v) in enumerate(past_key_values):
            seq_len = int(seq_lens[i])
            acc = self._acc_scores[i]
            protected = self._protected_masks[i]

            n_new = seq_len - acc.numel()
            if n_new > 0:
                acc = torch.cat([acc, torch.zeros(n_new, device=acc.device, dtype=acc.dtype)])
                protected = torch.cat(
                    [protected, torch.zeros(n_new, dtype=torch.bool, device=protected.device)]
                )

            # Same counterfactual principle at decode time, EMA-accumulated.
            step_score = self._cpr_step_scores(attentions[i], k, v).to(acc.device)
            acc = acc * self.decode_decay + step_score

            if forget_nums[i] <= 0:
                self._acc_scores[i] = acc
                self._protected_masks[i] = protected
                past_key_values_return.append([k, v])
                continue

            evictable = ~protected
            evictable[: self.start_size] = False
            if self.recent_window > 0:
                evictable[max(0, seq_len - self.recent_window) :] = False

            n_evict = min(int(forget_nums[i]), int(evictable.sum().item()))
            if n_evict <= 0:
                self._acc_scores[i] = acc
                self._protected_masks[i] = protected
                past_key_values_return.append([k, v])
                continue

            masked = acc.masked_fill(~evictable, float("inf"))
            evict_idx = torch.topk(masked, n_evict, largest=False).indices
            keep = torch.ones(seq_len, dtype=torch.bool, device=acc.device)
            keep[evict_idx] = False
            keep_idx = keep.nonzero(as_tuple=True)[0]

            keep_idx_kv = keep_idx.to(k.device)
            past_key_values_return.append(
                [
                    k.index_select(self.k_seq_dim, keep_idx_kv),
                    v.index_select(self.v_seq_dim, keep_idx_kv),
                ]
            )
            self._acc_scores[i] = acc[keep_idx]
            self._protected_masks[i] = protected[keep_idx]

        return DynamicCache(past_key_values_return)