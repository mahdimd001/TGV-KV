import torch
import numpy as np
from colorama import Fore
from transformers.cache_utils import DynamicCache

from temp import *


from my_forman_ricci2 import FormanRicciTensorGPU

class RicciKVCache:
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
        protect_size=1,
        sparsification=20.0,
        **kwargs,
    ):
        self.start_size = start_size
        self.protect_size = protect_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

        self.batch_size = batch_size
        self.layer_num = layer_num
        self.image_token_id = image_token_id

        self.ratio = ratio
        self.sparsification = 20.0

    # ---------------------------------------------------------
    # 1. ONLINE PREFILL HOOKS (Matches TGV signatures)
    # ---------------------------------------------------------
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
            "all_attns": [None] * self.layer_num,
            "all_attns_sparsified": [None] * self.layer_num,
        }
        return True


    def block_threshold_sparsify(self, adj_mat: torch.Tensor, sparsification: float):
        """
        Sparsifies a matrix by keeping only values above a dynamic statistical threshold:
        threshold = mean + (sparsification * std)
        """
        # Safety check for empty tensors
        if adj_mat.numel() == 0:
            return adj_mat

        # 1. Calculate the block's specific statistics
        mean_val = adj_mat.mean()
        std_val = adj_mat.std()
        
        # 2. Determine the cutoff threshold
        threshold = mean_val + (sparsification * std_val)
        
        # 3. Apply the mask: keep original values if >= threshold, else set to 0
        sparse_mat = torch.where(adj_mat >= threshold, adj_mat, torch.zeros_like(adj_mat))
        
        return sparse_mat

    def collect_online_prefill_attention(self, layer_idx, attention):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None or attention is None:
            return

        # Store the average attention map across heads for Ricci graph construction
        all_attn = attention.squeeze(0).mean(0)
        stats["all_attns"][layer_idx] = all_attn




        image_start = stats["image_start"]
        visual_token_num = stats["visual_token_num"]
        text_start = stats["text_start"]


        text_image_attns = all_attn[text_start:, image_start:text_start]
        text_text_attns = all_attn[text_start:, text_start:]
        image_image_attns = all_attn[image_start : (image_start + visual_token_num), image_start : (image_start + visual_token_num)]




        text_image_attns = self.block_threshold_sparsify(text_image_attns, 1.0)
        text_text_attns = self.block_threshold_sparsify(text_text_attns, 1.0)
        image_image_attns = self.block_threshold_sparsify(image_image_attns, 3.5)




        # #count non-zero edges in each block for debugging
        # num_edges_text_image = text_image_attns.numel()
        # num_nonzero_text_image = (text_image_attns > 0).sum().item()
        # print(f"Text-Image Block: Non-zero edges: {num_nonzero_text_image}/{num_edges_text_image} ({100 * num_nonzero_text_image / num_edges_text_image:.2f}%)")

        # num_edges_text_text = text_text_attns.numel()
        # num_nonzero_text_text = (text_text_attns > 0).sum().item()
        # print(f"Text-Text Block: Non-zero edges: {num_nonzero_text_text}/{num_edges_text_text} ({100 * num_nonzero_text_text / num_edges_text_text:.2f}%)")

        # num_edges_image_image = image_image_attns.numel()
        # num_nonzero_image_image = (image_image_attns > 0).sum().item()
        # print(f"Image-Image Block: Non-zero edges: {num_nonzero_image_image}/{num_edges_image_image} ({100 * num_nonzero_image_image / num_edges_image_image:.2f}%)")
        comms, info = attention_to_communities(image_image_attns, k=12, flow_iterations=15)
        
        
        # create the all_attn from the 3 parts, filling the rest with zeros
        all_attn_full = torch.zeros_like(all_attn)
        all_attn_full[text_start:, image_start:text_start] = text_image_attns
        all_attn_full[text_start:, text_start:] = text_text_attns
        all_attn_full[image_start : (image_start + visual_token_num), image_start : (image_start + visual_token_num)] = image_image_attns

        stats["all_attns_sparsified"][layer_idx] = all_attn_full
        

    def finish_online_prefill(self):
        stats = getattr(self, "_online_prefill_stats", None)
        if stats is None:
            return None

        missing_layers = [idx for idx, attn in enumerate(stats["all_attns"]) if attn is None]
        if missing_layers:
            raise RuntimeError(f"Missing Ricci online prefill stats for layers: {missing_layers}")

        self._online_prefill_stats = None
        return {
            "ricci_online_prefill": True,
            "text_start": stats["text_start"],
            "all_attns": tuple(stats["all_attns"]),
            "all_attns_sparsified": tuple(stats["all_attns_sparsified"]),
        }

    # ---------------------------------------------------------
    # 2. STATE MANAGEMENT
    # ---------------------------------------------------------
    def set_pending_attentions(self, attentions):
        self._pending_attentions = attentions

    def pop_pending_attentions(self):
        attentions = getattr(self, "_pending_attentions", None)
        if hasattr(self, "_pending_attentions"):
            del self._pending_attentions
        return attentions

    def _is_online_prefill_stats(self, attentions):
        return isinstance(attentions, dict) and attentions.get("ricci_online_prefill", False)


    def get_ricci_indices_for_layer(
        self,
        layer_idx, 
        attn_score_all_layers, 
        past_key_values, 
        forget_num, 
        start_size, 
        protect_size, 
        sparsification=20.0,
        ):
        """
        Constructs a graph for a specific layer, computes Ricci curvature, 
        and returns the selected (anchor) and throw (merge) indices.
        """
        # 1. Extract variables for the current layer
        layer_attn = attn_score_all_layers[layer_idx]       # Shape: [624, 624]
        seq_len = layer_attn.shape[-1]                      # 624
        

        # OPTION 1: Attention Scores
        # Symmetrize the causal matrix: [624, 624]
        adj_mat = layer_attn + layer_attn.transpose(0, 1)

        # # Remove the self-loops (diagonal) so Ricci doesn't get confused
        adj_mat.fill_diagonal_(0.0)

        
        #convert dtype to float16 to speed up the following processing
        adj_mat = adj_mat.to(torch.float)

        mean_val = adj_mat.mean()
        std_val = adj_mat.std()
        threshold = mean_val + (sparsification * std_val)
        
        # Sparsify: Keep only the top 5% of edges so the optimal transport math doesn't hang
        #adj_mat[adj_mat < threshold] = 0.0
        
        
        num_edges = adj_mat.numel()
        num_nonzero = (adj_mat > 0).sum().item()
        #print(f"Layer {layer_idx}: Adjacency matrix size: {adj_mat.shape}, Non-zero edges: {num_nonzero}/{num_edges} ({100 * num_nonzero / num_edges:.2f}%)")


        
        orc = FormanRicciTensorGPU(adj_mat)
        a,b,node_ricci_scores = orc.compute_ricci_curvature()

        
        
        # 5. Filter out protected tokens (start_size and protect_size)
        # We only want to evaluate tokens in the "middle" for compression.
        middle_nodes = list(range(start_size, seq_len - protect_size))
        middle_ricci_scores = node_ricci_scores[middle_nodes]


        
        # Sort the middle nodes by Ricci Score (Ascending: Lowest to Highest)
        sorted_middle_indices = torch.argsort(middle_ricci_scores)
        
        # Map the sorted indices back to their absolute positions in the [0-623] sequence
        sorted_absolute_indices = torch.tensor(middle_nodes, device=layer_attn.device)[sorted_middle_indices]
        
        # Anchors (Keep): Lowest Ricci scores (Negative = Structural Bridges) and protected tokens
        selected_idx = sorted_absolute_indices[:-forget_num]
        selected_idx = torch.cat([torch.arange(start_size).cuda(), selected_idx, torch.arange(seq_len - protect_size,seq_len).cuda()], dim=0) # the last token is always kept

        
        # Redundant (Throw/Merge): Highest Ricci scores (Positive = Dense overlaps)
        throw_idx = sorted_absolute_indices[-forget_num:]

        
        return selected_idx, throw_idx

    # ---------------------------------------------------------
    # 3. ROUTING & CORE CACHE LOGIC
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. PREFILL LOGIC (Merging via Curvature)
    # ---------------------------------------------------------
    def _prefill(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))

        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return DynamicCache(past_key_values)

        # Fallback if hooks missed: compute avg attention map now
        all_attns = [x.squeeze(0).mean(0).detach().cpu() for x in attentions]
        return self._apply_ricci_pruning(past_key_values, all_attns, forget_num, seq_lens)

    def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio))

        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return DynamicCache(past_key_values)

        all_attns = attentions["all_attns"]
        all_attns_sparsified = attentions["all_attns_sparsified"]
        return self._apply_ricci_pruning(past_key_values, all_attns, forget_num, seq_lens, all_attns_sparsified=all_attns_sparsified)

    def _apply_ricci_pruning(self, past_key_values, all_attns, forget_num, seq_lens, all_attns_sparsified=None):
        past_key_values_return = []

        

        for idx in range(len(past_key_values)):
            k, v = past_key_values[idx]

            # 1. Fetch indices from your Ricci geometry calculation
            selected_idx, throw_idx = self.get_ricci_indices_for_layer(
                layer_idx=idx,
                attn_score_all_layers=all_attns_sparsified,
                past_key_values=past_key_values,
                forget_num=forget_num,
                start_size=self.start_size,
                protect_size=self.protect_size,
                sparsification=self.sparsification
            )
            # sort selected_idx and throw_idx to maintain original order in the cache
            selected_idx, _ = torch.sort(selected_idx)
            throw_idx, _ = torch.sort(throw_idx)


            merge_idx = []
            for i in range(len(throw_idx)):
                merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0))
            merge_idx = torch.cat(merge_idx)

            

            device = k.device
            selected_idx = selected_idx.to(device)
            throw_idx = throw_idx.to(device)

            self.initial_text_len_list.append(self.protect_size)

            # 2. Extract tokens targeted for deletion (throw_idx)
            k_forget = k.gather(dim=-2, index=throw_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]))
            v_forget = v.gather(dim=-2, index=throw_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]))

            # 3. Merge thrown tokens into their geometrically nearest neighbors via scatter_reduce
            k = k.scatter_reduce(-2, merge_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]), k_forget, 'mean', include_self=True)
            v = v.scatter_reduce(-2, merge_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]), v_forget, 'mean', include_self=True)

            # 4. Filter the cache to strictly retain only the selected nodes
            k_new = k.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1]))
            v_new = v.gather(dim=-2, index=selected_idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1]))

            past_key_values_return.append([k_new, v_new])

        return DynamicCache(past_key_values_return)

    # ---------------------------------------------------------
    # 5. AUTO-REGRESSIVE DECODE
    # ---------------------------------------------------------
    def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        forget_nums = (seq_lens - num_of_token * (1 - self.ratio)).astype(np.int32)
        forget_nums[forget_nums < 0] = 0

        if np.all(forget_nums <= 0):
            return DynamicCache(past_key_values)

        past_key_values_return = []
        selected_idx = getattr(self, "selected_idx", 16)

        for i, (k, v) in enumerate(past_key_values):
            if forget_nums[i] == 0:
                past_key_values_return.append([k, v])
                continue
                
            seq_len = seq_lens[i]
            


            # Safety check: Prevent out-of-bounds slicing
            if selected_idx < 0 or selected_idx >= seq_len:
                past_key_values_return.append([k, v])
                continue

            k_new = torch.cat([k[:, :, 0:selected_idx], k[:, :, (selected_idx + 1):seq_len]], dim=self.k_seq_dim)
            v_new = torch.cat([v[:, :, 0:selected_idx], v[:, :, (selected_idx + 1):seq_len]], dim=self.v_seq_dim)
            
            past_key_values_return.append([k_new, v_new])
            
        return DynamicCache(past_key_values_return)




# vertions1

# import numpy as np
# import torch
# import torch.nn.functional as F
# from colorama import Fore
# from transformers.cache_utils import DynamicCache

# from my_forman_ricci2 import FormanRicciTensorGPU


# class RicciKVCache:
#     """
#     Forman-Ricci-curvature KV eviction for VLMs, upgraded with three ideas
#     transferred from TGV-KV:

#       * TVB (Text-Vision Budgeting): per-layer budget from text->vision attention.
#       * TWR (Text-Weighted Ranking): vision importance grounded in dominant text.
#       * TPR (Text-Prioritised Retention): protect text/system KV; evict vision first.

#     The Ricci-specific contribution is preserved: redundant (high-curvature) vision
#     tokens are *merged* into their most similar retained neighbour rather than just
#     dropped. Curvature now only ranks the vision block, which is both faster and
#     avoids the catastrophic case of merging away dominant text tokens.
#     """

#     supports_online_prefill = True

#     def __init__(
#         self,
#         layer_num,
#         image_token_id,
#         start_size=4,
#         k_seq_dim=2,
#         v_seq_dim=2,
#         ratio=0.0,
#         batch_size=1,
#         protect_size=1,
#         sparsification=2.0,
#         # ---- ablation / behaviour switches ----
#         use_tvb=True,          # per-layer budget from TV attention (Step 1)
#         use_twr=True,          # text-grounded vision scoring (Step 3)
#         use_tpr=True,          # protect text, evict vision first (Step 2)
#         twr_alpha=0.9,         # blend: alpha*text_score + (1-alpha)*curvature_score
#         merge_mode="evict",     # "mean" (merge) or "evict" (drop) thrown vision KV
#         **kwargs,
#     ):
#         self.start_size = start_size
#         self.protect_size = protect_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim

#         self.batch_size = batch_size
#         self.layer_num = layer_num
#         self.image_token_id = image_token_id

#         self.ratio = ratio
#         self.sparsification = sparsification

#         self.use_tvb = use_tvb
#         self.use_twr = use_twr
#         self.use_tpr = use_tpr
#         self.twr_alpha = twr_alpha
#         self.merge_mode = merge_mode

#         # filled during prefill, consumed during decode
#         self.ratios = None
#         self.initial_text_len_list = []

#     # =========================================================
#     # helpers
#     # =========================================================
#     def _sparsify(self, mat, k):
#         """Keep entries >= mean + k*std, zero the rest (statistical thresholding)."""
#         if mat.numel() == 0:
#             return mat
#         thr = mat.mean() + k * mat.std()
#         return torch.where(mat >= thr, mat, torch.zeros_like(mat))

#     @staticmethod
#     def _minmax(x):
#         """Min-max normalise a 1-D tensor to [0, 1]; flat tensors map to zeros."""
#         if x.numel() == 0:
#             return x
#         lo, hi = x.min(), x.max()
#         if (hi - lo) <= 0:
#             return torch.zeros_like(x)
#         return (x - lo) / (hi - lo)

#     def _layer_summary(self, all_attn, image_start, visual_token_num, text_start):
#         """
#         Build the compact per-layer summary that downstream pruning needs.

#         Returns a dict with:
#           vv_adj           : [Nv, Nv] sparsified vision-vision adjacency (for Ricci)
#           vision_twr       : [Nv]     text-weighted vision importance (TWR, Eq. 8)
#           text_importance  : [Nt]     text self-attention importance (Eq. 9)
#           tv_sum           : scalar   text->vision attention mass (TVB, Eq. 5)
#         """
#         v_end = image_start + visual_token_num

#         tv_block = all_attn[text_start:, image_start:text_start]        # [Nt, Nv]
#         tt_block = all_attn[text_start:, text_start:]                   # [Nt, Nt]
#         vv_block = all_attn[image_start:v_end, image_start:v_end]       # [Nv, Nv]

#         # --- TWR: causal-averaged column sum of text-text -> text weights (Eq. 7)
#         if tt_block.numel() > 0:
#             text_col = tt_block.sum(0, keepdim=True)                    # [1, Nt]
#             divisor = torch.arange(
#                 1, text_col.shape[-1] + 1, device=text_col.device
#             ).flip(0).unsqueeze(0)
#             text_weight = text_col / divisor                           # [1, Nt]
#             vision_twr = (tv_block * text_weight.transpose(0, 1)).sum(0)  # [Nv]
#             text_importance = tt_block.sum(0)                          # [Nt]
#         else:
#             vision_twr = tv_block.sum(0)
#             text_importance = all_attn.new_zeros(0)

#         return {
#             "vv_adj": self._sparsify(vv_block.float(), self.sparsification),
#             "vision_twr": vision_twr.float(),
#             "text_importance": text_importance.float(),
#             "tv_sum": tv_block.sum().float(),
#         }

#     # =========================================================
#     # 1. ONLINE PREFILL HOOKS (TGV-compatible signatures)
#     # =========================================================
#     def begin_online_prefill(self, input_ids):
#         if input_ids is None:
#             return False

#         image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
#         if len(image_positions) < 2 or image_positions[0].numel() == 0:
#             return False

#         image_start = image_positions[1][0].item()
#         visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
#         text_start = image_start + visual_token_num

#         self._online_prefill_stats = {
#             "image_start": image_start,
#             "visual_token_num": visual_token_num,
#             "text_start": text_start,
#             "summaries": [None] * self.layer_num,
#         }
#         return True

#     def collect_online_prefill_attention(self, layer_idx, attention):
#         stats = getattr(self, "_online_prefill_stats", None)
#         if stats is None or attention is None:
#             return

#         all_attn = attention.squeeze(0).mean(0)  # mean over heads -> [S, S]
#         stats["summaries"][layer_idx] = self._layer_summary(
#             all_attn,
#             stats["image_start"],
#             stats["visual_token_num"],
#             stats["text_start"],
#         )

#     def finish_online_prefill(self):
#         stats = getattr(self, "_online_prefill_stats", None)
#         if stats is None:
#             return None

#         missing = [i for i, s in enumerate(stats["summaries"]) if s is None]
#         if missing:
#             raise RuntimeError(f"Missing Ricci online prefill stats for layers: {missing}")

#         self._online_prefill_stats = None
#         return {
#             "ricci_online_prefill": True,
#             "image_start": stats["image_start"],
#             "visual_token_num": stats["visual_token_num"],
#             "text_start": stats["text_start"],
#             "summaries": tuple(stats["summaries"]),
#         }

#     # =========================================================
#     # 2. STATE MANAGEMENT
#     # =========================================================
#     def set_pending_attentions(self, attentions):
#         self._pending_attentions = attentions

#     def pop_pending_attentions(self):
#         attentions = getattr(self, "_pending_attentions", None)
#         if hasattr(self, "_pending_attentions"):
#             del self._pending_attentions
#         return attentions

#     def _is_online_prefill_stats(self, attentions):
#         return isinstance(attentions, dict) and attentions.get("ricci_online_prefill", False)

#     # =========================================================
#     # 3. ROUTING
#     # =========================================================
#     def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         if past_key_values is None:
#             return None

#         if self._is_online_prefill_stats(attentions):
#             self.initial_text_len_list = []
#             return self._prefill_from_online_stats(past_key_values, num_of_token, attentions)

#         if attentions[0].shape[-2] > 1:
#             self.initial_text_len_list = []
#             return self._prefill(past_key_values, num_of_token, attentions, input_ids)

#         return self._decode(past_key_values, num_of_token, attentions, input_ids)

#     # =========================================================
#     # 4. PREFILL
#     # =========================================================
#     def _build_summaries_from_raw(self, attentions, input_ids):
#         """Fallback path: derive the same per-layer summaries from raw attention."""
#         image_start = (input_ids == self.image_token_id).nonzero(as_tuple=True)[1][0].item()
#         visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
#         text_start = image_start + visual_token_num

#         summaries = []
#         for x in attentions:
#             all_attn = x.squeeze(0).mean(0)
#             summaries.append(
#                 self._layer_summary(all_attn, image_start, visual_token_num, text_start)
#             )
#         return image_start, visual_token_num, text_start, summaries

#     def _prefill(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         # Step 6: the non-online path now builds the same summaries instead of crashing.
#         image_start, visual_token_num, text_start, summaries = self._build_summaries_from_raw(
#             attentions, input_ids
#         )
#         meta = {
#             "image_start": image_start,
#             "visual_token_num": visual_token_num,
#             "text_start": text_start,
#             "summaries": summaries,
#         }
#         return self._apply_ricci_pruning(past_key_values, num_of_token, meta)

#     def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
#         return self._apply_ricci_pruning(past_key_values, num_of_token, attentions)

#     # -------- TVB: per-layer budget (Step 1) --------
#     def _layer_forget_nums(self, summaries, seq_lens, target_keep):
#         """
#         Distribute the average keep budget (target_keep) across layers proportionally
#         to each layer's text->vision attention mass (TGV Eq. 5). Layers with more
#         cross-modal interaction keep more KV.
#         """
#         L = len(summaries)
#         if self.use_tvb:
#             tv = torch.stack([s["tv_sum"] for s in summaries]).float()
#             tv = torch.clamp(tv, min=1e-9)
#             nlr = tv / tv.sum()                      # sums to 1 across layers
#             keep_per_layer = (L * target_keep) * nlr.cpu().numpy()
#         else:
#             keep_per_layer = np.full(L, float(target_keep))

#         forget = (seq_lens - keep_per_layer).round().astype(np.int64)
#         forget = np.clip(forget, 0, seq_lens - (self.start_size + self.protect_size))
#         return forget

#     def _vision_keep_scores(self, summary, vv_adj):
#         """
#         Combine geometry (Ricci) with text grounding (TWR) into a keep-score for
#         each vision token: higher = more important to keep.
#         """
#         orc = FormanRicciTensorGPU(vv_adj)
#         _, _, node_ricci = orc.compute_ricci_curvature()   # [Nv]
#         node_ricci = node_ricci.to(vv_adj.device).float()

#         # Low curvature == structural bridge == keep -> flip sign before normalising.
#         geo_keep = self._minmax(-node_ricci)

#         if self.use_twr:
#             twr = self._minmax(summary["vision_twr"].to(vv_adj.device))
#             return self.twr_alpha * twr + (1.0 - self.twr_alpha) * geo_keep

#         return geo_keep

#     def _apply_ricci_pruning(self, past_key_values, num_of_token, meta):
#         seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
#         seq_len0 = int(seq_lens[0])
#         target_keep = num_of_token * (1 - self.ratio)

#         if seq_len0 - target_keep <= 0:
#             print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
#             self.ratios = np.zeros(self.layer_num)
#             return DynamicCache([list(p) for p in past_key_values])

#         image_start = meta["image_start"]
#         visual_token_num = meta["visual_token_num"]
#         text_start = meta["text_start"]
#         summaries = meta["summaries"]
#         v_end = image_start + visual_token_num

#         forget_nums = self._layer_forget_nums(summaries, seq_lens, target_keep)
#         self.ratios = (forget_nums / np.maximum(seq_lens, 1)).astype(np.float64)

#         past_key_values_return = []
#         for idx in range(self.layer_num):
#             k, v = past_key_values[idx]
#             device = k.device
#             seq_len = int(seq_lens[idx])
#             forget_num = int(forget_nums[idx])
#             summary = summaries[idx]

#             if forget_num <= 0:
#                 past_key_values_return.append([k, v])
#                 self.initial_text_len_list.append(self.protect_size)
#                 continue

#             # Candidate pools (sinks + last protect_size are never touched).
#             vision_idx = torch.arange(
#                 max(image_start, self.start_size), min(v_end, seq_len - self.protect_size),
#                 device=device,
#             )
#             text_idx = torch.cat([
#                 torch.arange(self.start_size, image_start, device=device),               # system
#                 torch.arange(text_start, seq_len - self.protect_size, device=device),    # prompt
#             ])
#             keep_total = seq_len - forget_num
#             reserved = self.start_size + self.protect_size

#             vv_adj = summary["vv_adj"].to(device)
#             v_scores = self._vision_keep_scores(summary, vv_adj)  # [Nv], higher=keep

#             # ---- TPR: decide how many vision vs text to keep (Step 2) ----
#             middle_budget = max(keep_total - reserved, 0)
#             n_text = text_idx.numel()
#             n_vision = vision_idx.numel()

#             if self.use_tpr and middle_budget >= n_text:
#                 # Keep all text, fill remainder with the best vision.
#                 keep_text = text_idx
#                 vision_keep_n = min(middle_budget - n_text, n_vision)
#                 order = torch.argsort(v_scores, descending=True)
#                 keep_vis_local = order[:vision_keep_n]
#                 throw_vis_local = order[vision_keep_n:]
#             elif self.use_tpr:
#                 # Extreme budget: drop all vision, keep the most important text.
#                 ti = self._minmax(summary["text_importance"].to(device))
#                 # text_importance only covers prompt text; system tokens default to high.
#                 sys_n = (text_idx < image_start).sum().item()
#                 prompt_scores = ti if ti.numel() == n_text - sys_n else torch.zeros(
#                     n_text - sys_n, device=device
#                 )
#                 full_scores = torch.cat([
#                     torch.ones(sys_n, device=device), prompt_scores
#                 ])
#                 order = torch.argsort(full_scores, descending=True)
#                 keep_text = text_idx[order[:middle_budget]]
#                 keep_vis_local = torch.empty(0, dtype=torch.long, device=device)
#                 throw_vis_local = torch.arange(n_vision, device=device)
#             else:
#                 # No TPR: rank vision and text jointly, keep the top middle_budget.
#                 if summary["text_importance"].numel() == n_text:
#                     t_scores = self._minmax(summary["text_importance"].to(device))
#                 else:
#                     t_scores = torch.zeros(n_text, device=device)
#                 all_local = torch.cat([vision_idx, text_idx])
#                 all_scores = torch.cat([v_scores, t_scores])
#                 order = torch.argsort(all_scores, descending=True)
#                 keep_mid = all_local[order[:middle_budget]]
#                 selected = torch.cat([
#                     torch.arange(self.start_size, device=device),
#                     keep_mid,
#                     torch.arange(seq_len - self.protect_size, seq_len, device=device),
#                 ]).unique().sort().values
#                 k_new, v_new = self._gather(k, v, selected)
#                 past_key_values_return.append([k_new, v_new])
#                 self.initial_text_len_list.append(
#                     max(int((selected >= text_start).sum().item()), self.protect_size)
#                 )
#                 continue

#             keep_vis = vision_idx[keep_vis_local]
#             throw_vis = vision_idx[throw_vis_local]

#             selected = torch.cat([
#                 torch.arange(self.start_size, device=device),
#                 keep_text,
#                 keep_vis,
#                 torch.arange(seq_len - self.protect_size, seq_len, device=device),
#             ]).unique().sort().values

#             # ---- Merge thrown vision into key-similar retained vision (Step 4) ----
#             if self.merge_mode == "mean" and throw_vis.numel() > 0 and keep_vis.numel() > 0:
#                 k, v = self._merge(k, v, throw_vis, keep_vis)

#             k_new, v_new = self._gather(k, v, selected)
#             past_key_values_return.append([k_new, v_new])
#             self.initial_text_len_list.append(
#                 max(int((selected >= text_start).sum().item()), self.protect_size)
#             )

#         return DynamicCache(past_key_values_return)

#     # -------- merge helper (Step 4) --------
#     def _merge(self, k, v, throw_idx, anchor_idx):
#         """
#         Merge each thrown token into its most key-similar retained anchor using a
#         running mean (not a sum), so merged keys/values stay in-distribution.
#         Math is done in fp32 for stability, then cast back to the cache dtype
#         (otherwise the merged KV become fp32 and break the half-precision model).
#         """
#         orig_dtype = k.dtype
#         device = k.device

#         # Per-token key representation (mean over heads) for similarity, in fp32.
#         key_repr = F.normalize(k[0].float().mean(0), dim=-1)    # [S, D]
#         sim = key_repr[throw_idx] @ key_repr[anchor_idx].T      # [n_throw, n_anchor]
#         nearest = anchor_idx[sim.argmax(dim=-1)]                # [n_throw] absolute idx

#         counts = torch.ones(k.size(self.k_seq_dim), device=device, dtype=torch.float32)
#         counts.index_add_(0, nearest, torch.ones_like(nearest, dtype=torch.float32))

#         k_f = k.float()
#         v_f = v.float()
#         k_f.index_add_(self.k_seq_dim, nearest, k[:, :, throw_idx, :].float())
#         v_f.index_add_(self.v_seq_dim, nearest, v[:, :, throw_idx, :].float())

#         denom = counts.view(1, 1, -1, 1)
#         return (k_f / denom).to(orig_dtype), (v_f / denom).to(orig_dtype)

#     def _gather(self, k, v, idx):
#         idx = idx.to(k.device)
#         ke = idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
#         ve = idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1])
#         return k.gather(-2, ke), v.gather(-2, ve)

#     # =========================================================
#     # 5. DECODE  (Step 5: real attention-min eviction, TGV-style)
#     # =========================================================
#     def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])

#         if self.ratios is None:
#             self.ratios = np.zeros(len(past_key_values))
#         forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
#         forget_nums[forget_nums < 0] = 0

#         if np.all(forget_nums <= 0):
#             return DynamicCache([list(p) for p in past_key_values])

#         past_key_values_return = []
#         for i, (k, v) in enumerate(past_key_values):
#             if forget_nums[i] == 0:
#                 past_key_values_return.append([k, v])
#                 continue

#             seq_len = int(seq_lens[i])
#             protected_suffix = (
#                 self.initial_text_len_list[i]
#                 if i < len(self.initial_text_len_list) else self.protect_size
#             )
#             evict_start = self.start_size
#             evict_end = seq_len - protected_suffix
#             if evict_start >= evict_end:
#                 past_key_values_return.append([k, v])
#                 continue

#             # Lowest accumulated attention inside the eviction window is removed.
#             decode_score = attentions[i].mean(1).squeeze(0).sum(0)      # [seq_len]
#             pruned = decode_score[evict_start:evict_end].argmin().item() + evict_start

#             past_key_values_return.append([
#                 torch.cat([k[:, :, :pruned], k[:, :, pruned + 1:seq_len]], dim=self.k_seq_dim),
#                 torch.cat([v[:, :, :pruned], v[:, :, pruned + 1:seq_len]], dim=self.v_seq_dim),
#             ])

#         return DynamicCache(past_key_values_return)





#V2

# import numpy as np
# import torch
# import torch.nn.functional as F
# from colorama import Fore
# from transformers.cache_utils import DynamicCache

# from my_forman_ricci2 import FormanRicciTensorGPU


# class RicciKVCache:
#     """
#     Forman-Ricci-curvature KV eviction for VLMs, upgraded with three ideas
#     transferred from TGV-KV:

#       * TVB (Text-Vision Budgeting): per-layer budget from text->vision attention.
#       * TWR (Text-Weighted Ranking): vision importance grounded in dominant text.
#       * TPR (Text-Prioritised Retention): protect text/system KV; evict vision first.

#     The Ricci-specific contribution is preserved: redundant (high-curvature) vision
#     tokens are *merged* into their most similar retained neighbour rather than just
#     dropped. Curvature now only ranks the vision block, which is both faster and
#     avoids the catastrophic case of merging away dominant text tokens.
#     """

#     supports_online_prefill = True

#     def __init__(
#         self,
#         layer_num,
#         image_token_id,
#         start_size=4,
#         k_seq_dim=2,
#         v_seq_dim=2,
#         ratio=0.0,
#         batch_size=1,
#         protect_size=1,
#         sparsification=20.0,
#         # ---- ablation / behaviour switches ----
#         use_tvb=True,          # per-layer budget from TV attention (Step 1)
#         use_twr=True,          # text-grounded vision scoring (Step 3)
#         use_tpr=True,          # protect text, evict vision first (Step 2)
#         twr_alpha=0.5,         # (legacy) blend weight; unused by select_mode below
#         merge_mode="evict",    # "mean" (merge) or "evict" (drop) thrown vision KV
#         # ---- how curvature is used (this is the part that changed) ----
#         select_mode="mmr",    # "topk"=pure TWR (baseline); "mmr"=curvature-diversified; "penalty"=fast loop-free
#         mmr_lambda=0.5,        # redundancy strength in MMR; 0.0 reduces MMR back to pure TWR
#         graph_topk=64,          # kNN degree of the vision key-similarity graph
#         **kwargs,
#     ):
#         self.start_size = start_size
#         self.protect_size = protect_size
#         self.k_seq_dim = k_seq_dim
#         self.v_seq_dim = v_seq_dim

#         self.batch_size = batch_size
#         self.layer_num = layer_num
#         self.image_token_id = image_token_id

#         self.ratio = ratio
#         self.sparsification = sparsification

#         self.use_tvb = use_tvb
#         self.use_twr = use_twr
#         self.use_tpr = use_tpr
#         self.twr_alpha = twr_alpha
#         self.merge_mode = merge_mode
#         self.select_mode = select_mode
#         self.mmr_lambda = mmr_lambda
#         self.graph_topk = graph_topk

#         # filled during prefill, consumed during decode
#         self.ratios = None
#         self.initial_text_len_list = []

#     # =========================================================
#     # helpers
#     # =========================================================
#     def _sparsify(self, mat, k):
#         """Keep entries >= mean + k*std, zero the rest (statistical thresholding)."""
#         if mat.numel() == 0:
#             return mat
#         thr = mat.mean() + k * mat.std()
#         return torch.where(mat >= thr, mat, torch.zeros_like(mat))

#     @staticmethod
#     def _minmax(x):
#         """Min-max normalise a 1-D tensor to [0, 1]; flat tensors map to zeros."""
#         if x.numel() == 0:
#             return x
#         lo, hi = x.min(), x.max()
#         if (hi - lo) <= 0:
#             return torch.zeros_like(x)
#         return (x - lo) / (hi - lo)

#     @staticmethod
#     def _rank_norm(x):
#         """Rank-normalise to [0, 1] (robust to heavy tails, unlike min-max)."""
#         if x.numel() <= 1:
#             return torch.zeros_like(x)
#         r = torch.empty_like(x)
#         r[x.argsort()] = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
#         return r / (x.numel() - 1)

#     def _sparsify_topk(self, mat, k_neighbors):
#         """Keep each row's k strongest entries (kNN graph), zero the rest."""
#         if mat.numel() == 0:
#             return mat
#         k = min(int(k_neighbors), mat.size(-1))
#         thr = mat.topk(k, dim=-1).values[..., -1:].expand_as(mat)
#         return torch.where(mat >= thr, mat, torch.zeros_like(mat))

#     def _layer_summary(self, all_attn, image_start, visual_token_num, text_start):
#         """
#         Build the compact per-layer summary that downstream pruning needs.

#         Returns a dict with:
#           vision_twr       : [Nv]     text-weighted vision importance (TWR, Eq. 8)
#           text_importance  : [Nt]     text self-attention importance (Eq. 9)
#           tv_sum           : scalar   text->vision attention mass (TVB, Eq. 5)
#         Note: the vision redundancy graph is now built from KEYS at prune time
#         (see _vision_keysim_graph), not from the attention block.
#         """
#         tv_block = all_attn[text_start:, image_start:text_start]        # [Nt, Nv]
#         tt_block = all_attn[text_start:, text_start:]                   # [Nt, Nt]

#         # --- TWR: causal-averaged column sum of text-text -> text weights (Eq. 7)
#         if tt_block.numel() > 0:
#             text_col = tt_block.sum(0, keepdim=True)                    # [1, Nt]
#             divisor = torch.arange(
#                 1, text_col.shape[-1] + 1, device=text_col.device
#             ).flip(0).unsqueeze(0)
#             text_weight = text_col / divisor                           # [1, Nt]
#             vision_twr = (tv_block * text_weight.transpose(0, 1)).sum(0)  # [Nv]
#             text_importance = tt_block.sum(0)                          # [Nt]
#         else:
#             vision_twr = tv_block.sum(0)
#             text_importance = all_attn.new_zeros(0)

#         return {
#             "vision_twr": vision_twr.float(),
#             "text_importance": text_importance.float(),
#             "tv_sum": tv_block.sum().float(),
#         }

#     # =========================================================
#     # 1. ONLINE PREFILL HOOKS (TGV-compatible signatures)
#     # =========================================================
#     def begin_online_prefill(self, input_ids):
#         if input_ids is None:
#             return False

#         image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)
#         if len(image_positions) < 2 or image_positions[0].numel() == 0:
#             return False

#         image_start = image_positions[1][0].item()
#         visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
#         text_start = image_start + visual_token_num

#         self._online_prefill_stats = {
#             "image_start": image_start,
#             "visual_token_num": visual_token_num,
#             "text_start": text_start,
#             "summaries": [None] * self.layer_num,
#         }
#         return True

#     def collect_online_prefill_attention(self, layer_idx, attention):
#         stats = getattr(self, "_online_prefill_stats", None)
#         if stats is None or attention is None:
#             return

#         all_attn = attention.squeeze(0).mean(0)  # mean over heads -> [S, S]
#         stats["summaries"][layer_idx] = self._layer_summary(
#             all_attn,
#             stats["image_start"],
#             stats["visual_token_num"],
#             stats["text_start"],
#         )

#     def finish_online_prefill(self):
#         stats = getattr(self, "_online_prefill_stats", None)
#         if stats is None:
#             return None

#         missing = [i for i, s in enumerate(stats["summaries"]) if s is None]
#         if missing:
#             raise RuntimeError(f"Missing Ricci online prefill stats for layers: {missing}")

#         self._online_prefill_stats = None
#         return {
#             "ricci_online_prefill": True,
#             "image_start": stats["image_start"],
#             "visual_token_num": stats["visual_token_num"],
#             "text_start": stats["text_start"],
#             "summaries": tuple(stats["summaries"]),
#         }

#     # =========================================================
#     # 2. STATE MANAGEMENT
#     # =========================================================
#     def set_pending_attentions(self, attentions):
#         self._pending_attentions = attentions

#     def pop_pending_attentions(self):
#         attentions = getattr(self, "_pending_attentions", None)
#         if hasattr(self, "_pending_attentions"):
#             del self._pending_attentions
#         return attentions

#     def _is_online_prefill_stats(self, attentions):
#         return isinstance(attentions, dict) and attentions.get("ricci_online_prefill", False)

#     # =========================================================
#     # 3. ROUTING
#     # =========================================================
#     def __call__(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         if past_key_values is None:
#             return None

#         if self._is_online_prefill_stats(attentions):
#             self.initial_text_len_list = []
#             return self._prefill_from_online_stats(past_key_values, num_of_token, attentions)

#         if attentions[0].shape[-2] > 1:
#             self.initial_text_len_list = []
#             return self._prefill(past_key_values, num_of_token, attentions, input_ids)

#         return self._decode(past_key_values, num_of_token, attentions, input_ids)

#     # =========================================================
#     # 4. PREFILL
#     # =========================================================
#     def _build_summaries_from_raw(self, attentions, input_ids):
#         """Fallback path: derive the same per-layer summaries from raw attention."""
#         image_start = (input_ids == self.image_token_id).nonzero(as_tuple=True)[1][0].item()
#         visual_token_num = (input_ids == self.image_token_id).nonzero().shape[0]
#         text_start = image_start + visual_token_num

#         summaries = []
#         for x in attentions:
#             all_attn = x.squeeze(0).mean(0)
#             summaries.append(
#                 self._layer_summary(all_attn, image_start, visual_token_num, text_start)
#             )
#         return image_start, visual_token_num, text_start, summaries

#     def _prefill(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         # Step 6: the non-online path now builds the same summaries instead of crashing.
#         image_start, visual_token_num, text_start, summaries = self._build_summaries_from_raw(
#             attentions, input_ids
#         )
#         meta = {
#             "image_start": image_start,
#             "visual_token_num": visual_token_num,
#             "text_start": text_start,
#             "summaries": summaries,
#         }
#         return self._apply_ricci_pruning(past_key_values, num_of_token, meta)

#     def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
#         return self._apply_ricci_pruning(past_key_values, num_of_token, attentions)

#     # -------- TVB: per-layer budget (Step 1) --------
#     def _layer_forget_nums(self, summaries, seq_lens, target_keep):
#         """
#         Distribute the average keep budget (target_keep) across layers proportionally
#         to each layer's text->vision attention mass (TGV Eq. 5). Layers with more
#         cross-modal interaction keep more KV.
#         """
#         L = len(summaries)
#         if self.use_tvb:
#             tv = torch.stack([s["tv_sum"] for s in summaries]).float()
#             tv = torch.clamp(tv, min=1e-9)
#             nlr = tv / tv.sum()                      # sums to 1 across layers
#             keep_per_layer = (L * target_keep) * nlr.cpu().numpy()
#         else:
#             keep_per_layer = np.full(L, float(target_keep))

#         forget = (seq_lens - keep_per_layer).round().astype(np.int64)
#         forget = np.clip(forget, 0, seq_lens - (self.start_size + self.protect_size))
#         return forget

#     def _vision_keysim_graph(self, k, vision_idx):
#         """
#         Build a kNN graph over candidate vision tokens from KEY cosine similarity
#         (a direct redundancy signal), then compute Forman-Ricci curvature on it.

#         Returns:
#           sim  : [Nvc, Nvc] dense positive key-similarity (for the MMR penalty)
#           curv : [Nvc]      node curvature (high == dense/redundant region)
#         """
#         kr = k[0, :, vision_idx, :].float().mean(0)      # [Nvc, D], mean over heads
#         kr = F.normalize(kr, dim=-1)
#         sim = (kr @ kr.T).clamp(min=0.0)                 # cosine similarity, edges >= 0
#         sim.fill_diagonal_(0.0)

#         adj = self._sparsify_topk(sim, self.graph_topk)  # kNN graph
#         adj = torch.maximum(adj, adj.T)                  # undirected

#         orc = FormanRicciTensorGPU(adj)
#         _, _, node_ricci = orc.compute_ricci_curvature()
#         return sim, node_ricci.to(k.device).float()

#     def _select_vision_mmr(self, relevance, sim, curv, keep_n):
#         """
#         Greedy MMR. Relevance (TWR) decides importance; curvature decides how hard a
#         token suppresses its similar neighbours. A token in a dense, high-curvature
#         cluster is penalised more once a similar token is already kept, which spreads
#         the budget across distinct image regions instead of one hot blob.
#         """
#         device = relevance.device
#         n = relevance.numel()
#         rel = self._rank_norm(relevance)
#         # Augmented Forman: a near-clique has the penalty term vanish (shared
#         # neighbours) so its curvature is the LEAST negative, while a bridge keeps a
#         # large penalty and is the MOST negative. Least-negative == clustered ==
#         # redundant -> it should suppress its neighbours hardest.
#         rstrength = self._rank_norm(curv)                # least-negative (clustered) -> ~1.0 penalty
#         max_sim = torch.zeros(n, device=device)
#         chosen = torch.zeros(n, dtype=torch.bool, device=device)
#         out = []
#         for _ in range(keep_n):
#             score = rel - self.mmr_lambda * rstrength * max_sim
#             score[chosen] = float("-inf")
#             j = int(torch.argmax(score))
#             out.append(j)
#             chosen[j] = True
#             max_sim = torch.maximum(max_sim, sim[j])     # redundancy w.r.t. kept set
#         return torch.tensor(out, device=device, dtype=torch.long)

#     def _select_vision(self, k, vision_idx, image_start, summary, keep_n):
#         """
#         Choose which candidate vision tokens to keep. Returns (keep_local, throw_local)
#         as indices into vision_idx.
#           select_mode="topk" -> pure TWR top-k (your current best).
#           select_mode="mmr"  -> TWR relevance + curvature-driven diversity.
#         """
#         device = k.device
#         n = vision_idx.numel()
#         keep_n = int(max(min(keep_n, n), 0))
#         twr = summary["vision_twr"].to(device)
#         local_twr = twr[(vision_idx - image_start).clamp(0, twr.numel() - 1)]

#         if keep_n >= n:
#             return torch.arange(n, device=device), torch.empty(0, dtype=torch.long, device=device)
#         if keep_n == 0:
#             return torch.empty(0, dtype=torch.long, device=device), torch.arange(n, device=device)

#         if self.select_mode == "mmr":
#             sim, curv = self._vision_keysim_graph(k, vision_idx)
#             keep_local = self._select_vision_mmr(local_twr, sim, curv, keep_n)
#         elif self.select_mode == "penalty":
#             # Fast, loop-free: globally down-weight redundant (clustered) tokens.
#             # Cheaper than MMR but penalises cluster representatives too, so it
#             # under-keeps dense regions slightly. Best when keep_n is large.
#             _, curv = self._vision_keysim_graph(k, vision_idx)
#             redundancy = self._rank_norm(curv)           # clustered == redundant
#             score = self._rank_norm(local_twr) - self.mmr_lambda * redundancy
#             keep_local = torch.argsort(score, descending=True)[:keep_n]
#         else:  # "topk"
#             keep_local = torch.argsort(local_twr, descending=True)[:keep_n]

#         mask = torch.zeros(n, dtype=torch.bool, device=device)
#         mask[keep_local] = True
#         throw_local = torch.nonzero(~mask, as_tuple=True)[0]
#         return keep_local, throw_local

#     def _apply_ricci_pruning(self, past_key_values, num_of_token, meta):
#         seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
#         seq_len0 = int(seq_lens[0])
#         target_keep = num_of_token * (1 - self.ratio)

#         if seq_len0 - target_keep <= 0:
#             print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
#             self.ratios = np.zeros(self.layer_num)
#             return DynamicCache([list(p) for p in past_key_values])

#         image_start = meta["image_start"]
#         visual_token_num = meta["visual_token_num"]
#         text_start = meta["text_start"]
#         summaries = meta["summaries"]
#         v_end = image_start + visual_token_num

#         forget_nums = self._layer_forget_nums(summaries, seq_lens, target_keep)
#         self.ratios = (forget_nums / np.maximum(seq_lens, 1)).astype(np.float64)

#         past_key_values_return = []
#         for idx in range(self.layer_num):
#             k, v = past_key_values[idx]
#             device = k.device
#             seq_len = int(seq_lens[idx])
#             forget_num = int(forget_nums[idx])
#             summary = summaries[idx]

#             if forget_num <= 0:
#                 past_key_values_return.append([k, v])
#                 self.initial_text_len_list.append(self.protect_size)
#                 continue

#             # Candidate pools (sinks + last protect_size are never touched).
#             vision_idx = torch.arange(
#                 max(image_start, self.start_size), min(v_end, seq_len - self.protect_size),
#                 device=device,
#             )
#             text_idx = torch.cat([
#                 torch.arange(self.start_size, image_start, device=device),               # system
#                 torch.arange(text_start, seq_len - self.protect_size, device=device),    # prompt
#             ])
#             keep_total = seq_len - forget_num
#             reserved = self.start_size + self.protect_size

#             # ---- TPR: decide how many vision vs text to keep (Step 2) ----
#             middle_budget = max(keep_total - reserved, 0)
#             n_text = text_idx.numel()
#             n_vision = vision_idx.numel()

#             if self.use_tpr and middle_budget >= n_text:
#                 # Keep all text, fill remainder with the best vision.
#                 keep_text = text_idx
#                 vision_keep_n = min(middle_budget - n_text, n_vision)
#                 keep_vis_local, throw_vis_local = self._select_vision(
#                     k, vision_idx, image_start, summary, vision_keep_n
#                 )
#             elif self.use_tpr:
#                 # Extreme budget: drop all vision, keep the most important text.
#                 ti = self._minmax(summary["text_importance"].to(device))
#                 # text_importance only covers prompt text; system tokens default to high.
#                 sys_n = (text_idx < image_start).sum().item()
#                 prompt_scores = ti if ti.numel() == n_text - sys_n else torch.zeros(
#                     n_text - sys_n, device=device
#                 )
#                 full_scores = torch.cat([
#                     torch.ones(sys_n, device=device), prompt_scores
#                 ])
#                 order = torch.argsort(full_scores, descending=True)
#                 keep_text = text_idx[order[:middle_budget]]
#                 keep_vis_local = torch.empty(0, dtype=torch.long, device=device)
#                 throw_vis_local = torch.arange(n_vision, device=device)
#             else:
#                 # No TPR: rank vision (by TWR) and text (by self-attention) jointly.
#                 vtwr = summary["vision_twr"].to(device)
#                 v_scores = self._rank_norm(
#                     vtwr[(vision_idx - image_start).clamp(0, vtwr.numel() - 1)]
#                 )
#                 if summary["text_importance"].numel() == n_text:
#                     t_scores = self._rank_norm(summary["text_importance"].to(device))
#                 else:
#                     t_scores = torch.zeros(n_text, device=device)
#                 all_local = torch.cat([vision_idx, text_idx])
#                 all_scores = torch.cat([v_scores, t_scores])
#                 order = torch.argsort(all_scores, descending=True)
#                 keep_mid = all_local[order[:middle_budget]]
#                 selected = torch.cat([
#                     torch.arange(self.start_size, device=device),
#                     keep_mid,
#                     torch.arange(seq_len - self.protect_size, seq_len, device=device),
#                 ]).unique().sort().values
#                 k_new, v_new = self._gather(k, v, selected)
#                 past_key_values_return.append([k_new, v_new])
#                 self.initial_text_len_list.append(
#                     max(int((selected >= text_start).sum().item()), self.protect_size)
#                 )
#                 continue

#             keep_vis = vision_idx[keep_vis_local]
#             throw_vis = vision_idx[throw_vis_local]

#             selected = torch.cat([
#                 torch.arange(self.start_size, device=device),
#                 keep_text,
#                 keep_vis,
#                 torch.arange(seq_len - self.protect_size, seq_len, device=device),
#             ]).unique().sort().values

#             # ---- Merge thrown vision into key-similar retained vision (Step 4) ----
#             if self.merge_mode == "mean" and throw_vis.numel() > 0 and keep_vis.numel() > 0:
#                 k, v = self._merge(k, v, throw_vis, keep_vis)

#             k_new, v_new = self._gather(k, v, selected)
#             past_key_values_return.append([k_new, v_new])
#             self.initial_text_len_list.append(
#                 max(int((selected >= text_start).sum().item()), self.protect_size)
#             )

#         return DynamicCache(past_key_values_return)

#     # -------- merge helper (Step 4) --------
#     def _merge(self, k, v, throw_idx, anchor_idx):
#         """
#         Merge each thrown token into its most key-similar retained anchor using a
#         running mean (not a sum), so merged keys/values stay in-distribution.
#         Math is done in fp32 for stability, then cast back to the cache dtype
#         (otherwise the merged KV become fp32 and break the half-precision model).
#         """
#         orig_dtype = k.dtype
#         device = k.device

#         # Per-token key representation (mean over heads) for similarity, in fp32.
#         key_repr = F.normalize(k[0].float().mean(0), dim=-1)    # [S, D]
#         sim = key_repr[throw_idx] @ key_repr[anchor_idx].T      # [n_throw, n_anchor]
#         nearest = anchor_idx[sim.argmax(dim=-1)]                # [n_throw] absolute idx

#         counts = torch.ones(k.size(self.k_seq_dim), device=device, dtype=torch.float32)
#         counts.index_add_(0, nearest, torch.ones_like(nearest, dtype=torch.float32))

#         k_f = k.float()
#         v_f = v.float()
#         k_f.index_add_(self.k_seq_dim, nearest, k[:, :, throw_idx, :].float())
#         v_f.index_add_(self.v_seq_dim, nearest, v[:, :, throw_idx, :].float())

#         denom = counts.view(1, 1, -1, 1)
#         return (k_f / denom).to(orig_dtype), (v_f / denom).to(orig_dtype)

#     def _gather(self, k, v, idx):
#         idx = idx.to(k.device)
#         ke = idx.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, k.shape[-1])
#         ve = idx.view(1, 1, -1, 1).expand(v.shape[0], v.shape[1], -1, v.shape[-1])
#         return k.gather(-2, ke), v.gather(-2, ve)

#     # =========================================================
#     # 5. DECODE  (Step 5: real attention-min eviction, TGV-style)
#     # =========================================================
#     def _decode(self, past_key_values, num_of_token=None, attentions=None, input_ids=None):
#         seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])

#         if self.ratios is None:
#             self.ratios = np.zeros(len(past_key_values))
#         forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
#         forget_nums[forget_nums < 0] = 0

#         if np.all(forget_nums <= 0):
#             return DynamicCache([list(p) for p in past_key_values])

#         past_key_values_return = []
#         for i, (k, v) in enumerate(past_key_values):
#             if forget_nums[i] == 0:
#                 past_key_values_return.append([k, v])
#                 continue

#             seq_len = int(seq_lens[i])
#             protected_suffix = (
#                 self.initial_text_len_list[i]
#                 if i < len(self.initial_text_len_list) else self.protect_size
#             )
#             evict_start = self.start_size
#             evict_end = seq_len - protected_suffix
#             if evict_start >= evict_end:
#                 past_key_values_return.append([k, v])
#                 continue

#             # Lowest accumulated attention inside the eviction window is removed.
#             decode_score = attentions[i].mean(1).squeeze(0).sum(0)      # [seq_len]
#             pruned = decode_score[evict_start:evict_end].argmin().item() + evict_start

#             past_key_values_return.append([
#                 torch.cat([k[:, :, :pruned], k[:, :, pruned + 1:seq_len]], dim=self.k_seq_dim),
#                 torch.cat([v[:, :, :pruned], v[:, :, pruned + 1:seq_len]], dim=self.v_seq_dim),
#             ])

#         return DynamicCache(past_key_values_return)