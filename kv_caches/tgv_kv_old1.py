import numpy as np
import torch
from colorama import Fore
from transformers.cache_utils import DynamicCache


class TGVKVCache:
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

        self.enable_merge = True
        self.merge_thresh = 0.95
    # 1
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
            "score_sums": [None] * self.layer_num,
            "all_attn" : [None] * self.layer_num,
            "vision_token_similarity" : [None] * self.layer_num, 
            "text_image_attn_sums": [None] * self.layer_num,
        }
        return True

    def _similarity(self, A, sim_source, image_feats, grid_shape, spatial_mix, eps):
        """Pairwise cosine similarity among image tokens, mapped to [0,1]. [N,N]."""
        N = A.shape[1]
        parts = []
        if "attn" in sim_source:
            D = A.t()                                # [N, T] attention profile per image token
            D = D / D.norm(dim=1, keepdim=True).clamp_min(eps)
            parts.append((D @ D.t()).clamp(0, 1))
        if "feat" in sim_source:
            assert image_feats is not None, "pass image_feats for sim_source containing 'feat'"
            Df = image_feats.float()
            Df = Df / Df.norm(dim=1, keepdim=True).clamp_min(eps)
            parts.append(((Df @ Df.t()) * 0.5 + 0.5).clamp(0, 1))    # cosine [-1,1] -> [0,1]
        if "spatial" in sim_source:
            R, C = grid_shape
            idx = torch.arange(N, device=A.device)
            rc = torch.stack([idx // C, idx % C], dim=1).float()
            d = torch.cdist(rc, rc)
            sig = (R + C) / 4.0
            parts.append(torch.exp(-(d ** 2) / (2 * sig ** 2)))
    
        if not parts:
            raise ValueError(f"bad sim_source={sim_source!r}")
        if len(parts) == 1:
            return parts[0]
        return (1 - spatial_mix) * parts[0] + spatial_mix * parts[-1]



    @torch.no_grad()
    def image_token_scores(self,
        cross_attn: torch.Tensor,          # [T, N] already text-weighted
        beta: float = 1.0,                 # diversity strength: 0 = plain sum(0), higher = more suppression
        mode: str = "density",            # "soft_nms" (keeps representatives) | "density"
        sim_source: str = "attn",          # "attn" | "feat" | "spatial" | "attn+spatial" | "feat+spatial"
        image_feats: torch.Tensor | None = None,   # [N, D] for sim_source containing "feat"
        grid_shape: tuple[int, int] = (24, 24),
        spatial_mix: float = 0.5,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        A = cross_attn.float()                       # fp16 in -> fp32 for stable argmax/exp
        T, N = A.shape
    
        importance = A.sum(0)                        # [N] the current (baseline) score
    
    
        sim = self._similarity(A, sim_source, image_feats, grid_shape, spatial_mix, eps)  # [N,N] in [0,1]
        sim.fill_diagonal_(0.0)

        if beta == 0:
            return importance,sim
    
        if mode == "soft_nms":
            more = importance[:, None] > importance[None, :]          # more[i,j]: i more important than j
            supp = sim.masked_fill(~more, 0.0).max(dim=0).values      # [N] sim to closest more-important token
            score = importance * (1.0 - supp).clamp(0.0, 1.0) ** beta
        elif mode == "density":
            # importance-weighted local density; representatives are penalized too (softer)
            dens = sim @ importance                                   # [N]
            dens = dens / dens.max().clamp_min(eps)
            score = importance * torch.exp(- beta * dens)
        else:
            raise ValueError(mode)
    
        return score,sim
    
 
    def med_norm(self,x):
        eps = 1e-6
        m = x.median()
        return x / m.clamp_min(eps) if (m > 0) else x
    
    def text_vision_score(self,all_attn, image_start, visual_token_num, vis_score,
                      window=16, pre_w=1.0, vis_w=1.0, post_w=1.5,
                      bridge_w=0.5, protect=1e4,text_token_num = 16):
        """Return score_sum [1, S] where text & vision are on a comparable scale,
        the +100 blanket is replaced by targeted protection."""
        S = all_attn.shape[0]
        text_start = image_start + visual_token_num
        W = min(window, S)
        # W = int(text_token_num/1)
    
        # observation window: last W query rows -> same queries score every token type
        win = all_attn[S - W:, :]                       # [W, S]
        prospective = win.sum(0)                         # [S]
    
        pre  = prospective[:image_start]                 # system prompt
        post = prospective[text_start:]                  # question
        vis  = vis_score.reshape(-1)                     # their diversity score
    
        score_pre  = pre_w  * self.med_norm(pre)
        score_vis  = vis_w  * self.med_norm(vis)
        score_post = post_w * self.med_norm(post)
    
        # cross-modal bridge: question tokens that pull on the image are load-bearing
        ti = all_attn[text_start:, image_start:text_start].sum(1)   # [post_len]
        score_post = score_post + bridge_w * self.med_norm(ti)
    
        # recency protection: last Wp question tokens are always kept
        Wp = min(W, post.shape[0])
        score_post[-Wp:] = score_post[-Wp:] + protect
    
        return torch.cat([score_pre, score_vis, score_post]).unsqueeze(0)
    
    # 2 for each layer
    def collect_online_prefill_attention(self, layer_idx, attention):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None or attention is None:
            return

        image_start = stats["image_start"]
        visual_token_num = stats["visual_token_num"]
        text_start = stats["text_start"]

        all_attn = attention.squeeze(0).mean(0)
        text_image_attns = all_attn[text_start:, image_start:text_start]
        text_text_attns = all_attn[text_start:, text_start:]
        image_image_attns = all_attn[image_start : (image_start + visual_token_num), image_start : (image_start + visual_token_num)]
        text_text_score = text_text_attns.sum(0, keepdim=True)
        b = torch.arange(1, text_text_score.shape[-1] + 1).flip([0]).to(text_text_score.device).unsqueeze(0)
        text_text_score = text_text_score / b
        text_image_score = (text_image_attns * text_text_score.transpose(-1, -2)).sum(0, keepdim=True)

        text_image_s = (text_image_attns * text_text_score.transpose(-1, -2))
        res,similarity = self.image_token_scores(cross_attn=text_image_s,beta=0, sim_source="attn+spatial",spatial_mix=0.4)
        text_tokens_num = text_text_score.shape[-1]

        pre_score = all_attn[:, :image_start].sum(0, keepdim=True) + 100
        post_score = all_attn[:, image_start + visual_token_num :].sum(0, keepdim=True) + 100

        #stats["score_sums"][layer_idx] = torch.cat([pre_score, text_image_score, post_score], dim=1)
        stats["score_sums"][layer_idx] = torch.cat([pre_score, res.unsqueeze(0), post_score], dim=1)
        #stats["score_sums"][layer_idx] = self.text_vision_score(all_attn, image_start, visual_token_num, vis_score=res,text_token_num=text_tokens_num)

        stats["text_image_attn_sums"][layer_idx] = text_image_attns.reshape(-1).sum()
        stats["all_attn"][layer_idx] = all_attn
        stats["vision_token_similarity"][layer_idx] = similarity 
    # 3
    def finish_online_prefill(self):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None:
            return None

        missing_layers = [idx for idx, score in enumerate(stats["score_sums"]) if score is None]
        if missing_layers:
            raise RuntimeError(f"Missing TGV-KV online prefill stats for layers: {missing_layers}")

        self._online_prefill_stats = None
        return {
            "tgv_kv_online_prefill": True,
            "text_start": stats["text_start"],
            "image_start" : stats['image_start'],
            "visual_token_num" : stats["visual_token_num"],
            "score_sums": tuple(stats["score_sums"]),
            "text_image_attn_sums": tuple(stats["text_image_attn_sums"]),
            "all_attn" : stats["all_attn"],
            "vision_token_similarity" : stats["vision_token_similarity"]
        }

    @torch.no_grad()
    def _pre_eviction_merge(self, k, v, scores, similarity, image_start, text_start, eps=1e-6):
        """
        Fully vectorized, loop-free merge of vision tokens using cluster centers.
        """
        dev = k.device
        N = text_start - image_start
        if N <= 0:
            return k, v, scores, 0

        vis_scores = scores[image_start:text_start].clone()
        sim = similarity.to(dev) # [N, N]

        # 1. Break exact ties in scores using indices to ensure a strict DAG (Directed Acyclic Graph)
        rank_offset = torch.linspace(0, 1e-6, N, device=dev)
        rank_scores = vis_scores + rank_offset

        # 2. Identify tokens dominated by a more important, similar token
        # is_better_and_similar[i, j] is True if j is similar to i AND j has a strictly higher score
        is_better_and_similar = (sim >= self.merge_thresh) & (rank_scores.unsqueeze(0) < rank_scores.unsqueeze(1))

        # 3. Find Cluster Centers: Tokens not dominated by ANY other token
        is_center = ~is_better_and_similar.any(dim=1) # [N]

        # 4. Mask the similarity matrix so tokens can ONLY merge into cluster centers
        valid_targets = (sim >= self.merge_thresh) & is_center.unsqueeze(0)

        # 5. For each token, find the best valid cluster center
        target_sims = sim.masked_fill(~valid_targets, -1.0)
        best_sims, best_targets = target_sims.max(dim=1) # [N]

        # 6. A token is redundant if it found a valid center AND is not a center itself
        is_redundant = (best_sims >= self.merge_thresh) & ~is_center
        
        merged_count = is_redundant.sum().item()
        if merged_count == 0:
            return k, v, scores, 0

        # 7. Create the assignment map (centers/unmerged map to themselves, redundant map to centers)
        assign = torch.arange(N, device=dev)
        assign = torch.where(is_redundant, best_targets, assign)

        # 8. Vectorized mass-weighted K and V accumulation
        k_vis = k[:, :, image_start:text_start, :]
        v_vis = v[:, :, image_start:text_start, :]
        w = vis_scores.float().clamp_min(eps)

        # Pre-weight the tensors
        k_w = k_vis * w.view(1, 1, N, 1)
        v_w = v_vis * w.view(1, 1, N, 1)

        # Initialize accumulation buffers
        k_acc = torch.zeros_like(k_vis, dtype=torch.float32)
        v_acc = torch.zeros_like(v_vis, dtype=torch.float32)
        w_acc = torch.zeros_like(w, dtype=torch.float32)

        # Scatter add the weights and values into their assigned cluster centers
        k_acc.index_add_(2, assign, k_w)
        v_acc.index_add_(2, assign, v_w)
        w_acc.index_add_(0, assign, w)

        # Normalize K/V back to their proper scale
        k_vis_merged = (k_acc / w_acc.view(1, 1, N, 1)).to(k.dtype)
        v_vis_merged = (v_acc / w_acc.view(1, 1, N, 1)).to(v.dtype)

        # 9. Accumulate the scores into the cluster centers
        scores_acc = torch.zeros_like(vis_scores)
        scores_acc.index_add_(0, assign, vis_scores)
        
        # Mark redundant tokens with a massive negative penalty to guarantee eviction downstream
        scores_acc[is_redundant] = -1e9

        # 10. Replace the vision chunk in the global tensors
        k_new = k.clone()
        v_new = v.clone()
        scores_new = scores.clone()

        k_new[:, :, image_start:text_start, :] = k_vis_merged
        v_new[:, :, image_start:text_start, :] = v_vis_merged
        scores_new[image_start:text_start] = scores_acc

        return k_new, v_new, scores_new, merged_count

    @torch.no_grad()
    def _merge_vision(self, k, v, kept_positions, image_start, text_start,
                      weight_full=None, eps=1e-6):
        """Fix A: merge each EVICTED vision token into its most-similar KEPT
        vision token, via mass-weighted averaging of K and V, BEFORE the caller
        gathers `kept_positions`. Only near-duplicates (sim >= merge_thresh) are
        fused; the rest are dropped (baseline behavior). Batch size 1.
        k, v: [1, H, S, D]. Returns modified (k, v); positions are unchanged."""
        H, S, D = k.shape[1], k.shape[2], k.shape[3]
        dev = k.device
        kept_positions = kept_positions.to(dev)

        pos = torch.arange(S, device=dev)
        kept_mask = torch.zeros(S, dtype=torch.bool, device=dev)
        kept_mask[kept_positions] = True
        is_vis = (pos >= image_start) & (pos < text_start)

        kept_vis  = torch.nonzero(kept_mask & is_vis,  as_tuple=True)[0]   # [K]
        evict_vis = torch.nonzero(~kept_mask & is_vis, as_tuple=True)[0]   # [E]
        if kept_vis.numel() == 0 or evict_vis.numel() == 0:
            return k, v

        # similarity feature: "key" (post-RoPE) or "value" (RoPE-clean); head-averaged, fp32
        feat = v if getattr(self, "merge_sim", "key") == "value" else k
        fbar = feat[0].float().mean(0)                                     # [S, D]
        kk = torch.nn.functional.normalize(fbar[kept_vis],  dim=-1)        # [K, D]
        ke = torch.nn.functional.normalize(fbar[evict_vis], dim=-1)        # [E, D]
        sim_max, tgt = (ke @ kk.t()).max(dim=1)                            # [E], [E]

        # --- only fuse genuine near-duplicates; drop the rest ---
        thresh = self.merge_thresh
        keep_merge = sim_max >= thresh
        if getattr(self, "merge_debug", False):
            print(f"[merge] evict={evict_vis.numel()} "
                  f"sim_max(mean/max)={sim_max.mean():.3f}/{sim_max.max():.3f} "
                  f"fused={int(keep_merge.sum())} ({100*keep_merge.float().mean():.1f}%)")
        evict_vis = evict_vis[keep_merge]
        tgt = tgt[keep_merge]
        if evict_vis.numel() == 0:
            return k, v                                                    # nothing duplicate -> = baseline

        # weights (importance mass), computed on the FILTERED evicted set
        if weight_full is not None:
            w = weight_full.to(dev).float().clamp_min(eps)
            w_kept, w_evict = w[kept_vis], w[evict_vis]
        else:
            w_kept  = torch.ones(kept_vis.numel(),  device=dev)
            w_evict = torch.ones(evict_vis.numel(), device=dev)

        k = k.clone(); v = v.clone()
        for cache in (k, v):
            x = cache[0]                                                   # [H, S, D], cache dtype
            acc = x[:, kept_vis, :].float() * w_kept[None, :, None]        # [H, K, D] fp32
            acc.index_add_(1, tgt, x[:, evict_vis, :].float() * w_evict[None, :, None])
            denom = w_kept.clone()
            denom.index_add_(0, tgt, w_evict)                             # [K]
            x[:, kept_vis, :] = (acc / denom[None, :, None].clamp_min(eps)).to(x.dtype)
        return k, v



    def set_pending_attentions(self, attentions):
        self._pending_attentions = attentions
    # 4
    def pop_pending_attentions(self):
        attentions = getattr(self, "_pending_attentions", None)
        if hasattr(self, "_pending_attentions"):
            del self._pending_attentions
        return attentions
    # 6
    def _is_online_prefill_stats(self, attentions):
        return isinstance(attentions, dict) and attentions.get("tgv_kv_online_prefill", False)
    # 5
    def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        if past_key_values is None:
            return None

        if self._is_online_prefill_stats(attentions):
            self.initial_text_len_list = []
            return self._prefill_from_online_stats(past_key_values, num_of_token, attentions)
        if attentions[0].shape[-2] > 1:
            self.initial_text_len_list = []
            return self._prefill(past_key_values, num_of_token, attentions, input_ids)
        return self._decode(past_key_values, num_of_token, attentions, input_ids)
    # 8
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

        all_attns = [x.squeeze(0).mean(0) for x in attentions]
        all_attns = torch.stack(all_attns)
        text_image_attns = all_attns[:, text_start:, image_start:text_start]
        text_text_attns = all_attns[:, text_start:, text_start:]
        text_text_score = text_text_attns.sum(1, keepdim=True)
        b = torch.arange(1, text_text_score.shape[-1] + 1).flip([0]).to(text_text_score.device).unsqueeze(0).unsqueeze(0)
        text_text_score = text_text_score / b
        text_image_score = (text_image_attns * text_text_score.transpose(-1, -2)).sum(1, keepdim=True)
        pre_score = all_attns[..., :image_start].sum(1, keepdim=True) + 100
        post_score = all_attns[..., image_start + visual_token_num :].sum(1, keepdim=True) + 100
        score_sum = torch.cat([pre_score, text_image_score, post_score], dim=2)

        text_image_attn_sum = text_image_attns.reshape(text_image_attns.size(0), -1).sum(dim=1)
        normalized_layer_ratio = text_image_attn_sum / text_image_attn_sum.sum()
        layer_ratio = (seq_len - (len(normalized_layer_ratio) * seq_len * (1 - self.ratio) * normalized_layer_ratio)) / seq_len
        self.ratios = layer_ratio.float().cpu().numpy()
        forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        past_key_values_return = []

        for idx in range(self.layer_num):
            forget_num = forget_nums[idx]
            seq_len = seq_lens[idx]
            selected_idx = torch.argsort(score_sum[idx, :, self.start_size : (seq_len - self.protect_size)])[:, forget_num:] + self.start_size
            selected_idx = selected_idx.sort().values

            device = selected_idx.device
            pre = torch.arange(self.start_size, device=device).unsqueeze(0).expand(self.batch_size, -1)
            post = torch.tensor([seq_len - self.protect_size], device=device).unsqueeze(0).expand(self.batch_size, -1)
            selected_idx = torch.cat([pre, selected_idx, post], dim=-1)
            self.initial_text_len_list.append(max((selected_idx[0] >= text_start).sum().item(), self.protect_size))

            k, v = past_key_values[idx]
            selected_idx = selected_idx.to(k.device)

            k_select = k.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, k.shape[1], -1, k.shape[-1]))
            v_select = v.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, v.shape[1], -1, v.shape[-1]))

            past_key_values_return.append([k_select, v_select])

        return DynamicCache(past_key_values_return)
    # 7
    def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
        #remove all of these
        score_sums = list(attentions["score_sums"])
        text_image_attn_sums = list(attentions["text_image_attn_sums"])
        
        similaritys = list(attentions["vision_token_similarity"])
        for idx in range (self.layer_num):
            # this part should be removed
            V_value = past_key_values[idx][1] 
            K_value = past_key_values[idx][0] 

            image_start = attentions["image_start"]
            visual_token_num = attentions["visual_token_num"]
            text_start = attentions["text_start"]

            all_attn = attentions['all_attn'][idx]
            text_image_attns = all_attn[text_start:, image_start:text_start]
            text_text_attns = all_attn[text_start:, text_start:]
            image_image_attns = all_attn[image_start : (image_start + visual_token_num), image_start : (image_start + visual_token_num)]
            text_text_score = text_text_attns.sum(0, keepdim=True)
            b = torch.arange(1, text_text_score.shape[-1] + 1).flip([0]).to(text_text_score.device).unsqueeze(0)
            text_text_score = text_text_score / b
            text_image_score = (text_image_attns * text_text_score.transpose(-1, -2)).sum(0, keepdim=True)

            text_image_s = (text_image_attns * text_text_score.transpose(-1, -2))
            # this could be K or V
            image_feats = K_value.squeeze(0).mean(0)[image_start : (image_start + visual_token_num), :]
            res,similaritys[idx] = self.image_token_scores(cross_attn=text_image_s,image_feats=image_feats,beta=0,mode='density', sim_source="feat",spatial_mix=0.2)
            text_tokens_num = text_text_score.shape[-1]

            pre_score = all_attn[:, :image_start].sum(0, keepdim=True) + 100
            post_score = all_attn[:, image_start + visual_token_num :].sum(0, keepdim=True) + 100

            #stats["score_sums"][layer_idx] = torch.cat([pre_score, text_image_score, post_score], dim=1)
            score_sums[idx] = torch.cat([pre_score, res.unsqueeze(0), post_score], dim=1)
            #stats["score_sums"][layer_idx] = self.text_vision_score(all_attn, image_start, visual_token_num, vis_score=res,text_token_num=text_tokens_num)

            # attentions["text_image_attn_sums"][idx] = text_image_attns.reshape(-1).sum()
            
            #sim = self._similarity(K_value.squeeze(0).mean(0),image_feats =K_value.squeeze(0).mean(0) ,sim_source = 'feat',grid_shape = (24,24),spatial_mix = 0.5,eps = 1e-8)

            ##########
        # remove this and uncomment score_sum blow
        score_sum = torch.stack(score_sums)




        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        text_start = attentions["text_start"]
        image_start = attentions["image_start"]
        #score_sum = torch.stack(list(attentions["score_sums"]))
        text_image_attn_sum = torch.stack(list(attentions["text_image_attn_sums"]))

        normalized_layer_ratio = text_image_attn_sum / text_image_attn_sum.sum()
        layer_ratio = (seq_len - (len(normalized_layer_ratio) * seq_len * (1 - self.ratio) * normalized_layer_ratio)) / seq_len
        self.ratios = layer_ratio.float().cpu().numpy()
        forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        past_key_values_return = []

        for idx in range(self.layer_num):






            original_forget_num = forget_nums[idx]
            seq_len = seq_lens[idx]
            
            k, v = past_key_values[idx]
            score = score_sum[idx, 0].clone()
            #similarity = attentions["vision_token_similarity"][idx]
            similarity = similaritys[idx]

            # 1. Merge similar tokens and combine their scores BEFORE eviction
            if self.enable_merge:
                k, v, score, merged_count = self._pre_eviction_merge(
                    k, v, score, similarity, image_start, text_start
                )
                
                # 2. Compute new forget_num
                # We already "lost" merged_count tokens (they are flagged for guaranteed eviction).
                # If original_forget_num is higher, we need to evict extra. If it's lower, we still 
                # MUST evict the merged tokens since their information was already absorbed.
                total_to_evict = max(original_forget_num, merged_count)
            else:
                total_to_evict = original_forget_num

            # 3. Sort by the UPDATED scores to find which tokens to keep
            # The redundant merged tokens (score = -1e9) will cleanly fall into the evicted bucket here.
            selected_idx = torch.argsort(score[self.start_size : (seq_len - self.protect_size)])[total_to_evict:] + self.start_size
            selected_idx = selected_idx.sort().values

            device = selected_idx.device
            pre = torch.arange(self.start_size, device=device).unsqueeze(0).expand(self.batch_size, -1)
            post = torch.tensor([seq_len - self.protect_size], device=device).unsqueeze(0).expand(self.batch_size, -1)
            selected_idx = torch.cat([pre, selected_idx.unsqueeze(0), post], dim=-1)
            
            self.initial_text_len_list.append(max((selected_idx[0] >= text_start).sum().item(), self.protect_size))

            selected_idx = selected_idx.to(k.device)

            # 4. Standard Gather (The merged tokens are naturally dropped here)
            k_select = k.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, k.shape[1], -1, k.shape[-1]))
            v_select = v.gather(dim=-2, index=selected_idx.view(self.batch_size, 1, -1, 1).expand(-1, v.shape[1], -1, v.shape[-1]))

            past_key_values_return.append([k_select, v_select])

        return DynamicCache(past_key_values_return)




    # 9
    def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            return past_key_values

        past_key_values_return = []
        for i, (k, v) in enumerate(past_key_values):
            if forget_nums[i] == 0:
                past_key_values_return.append([k, v])
                continue
            seq_len = seq_lens[i]
            protected_suffix_len = self.initial_text_len_list[i] if hasattr(self, "initial_text_len_list") else self.protect_size
            evict_start = self.start_size
            evict_end = seq_len - protected_suffix_len
            if evict_start >= evict_end:
                past_key_values_return.append([k, v])
                continue
            decode_score = attentions[i].mean(1).squeeze(0).sum(0)
            pruned_idx = decode_score[evict_start:evict_end].argmin().item() + evict_start
            past_key_values_return.append(
                [
                    torch.cat([k[:, :, 0:pruned_idx], k[:, :, (pruned_idx + 1) : seq_len]], dim=self.k_seq_dim),
                    torch.cat([v[:, :, 0:pruned_idx], v[:, :, (pruned_idx + 1) : seq_len]], dim=self.v_seq_dim),
                ]
            )
        return DynamicCache(past_key_values_return)
