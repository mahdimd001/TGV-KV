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

        all_attn = attention.squeeze(0).mean(0)
        text_image_attns = all_attn[text_start:, image_start:text_start]
        text_text_attns = all_attn[text_start:, text_start:]
        text_text_score = text_text_attns.sum(0, keepdim=True)
        b = torch.arange(1, text_text_score.shape[-1] + 1).flip([0]).to(text_text_score.device).unsqueeze(0)
        text_text_score = text_text_score / b
        text_image_score = (text_image_attns * text_text_score.transpose(-1, -2)).sum(0, keepdim=True)
        pre_score = all_attn[:, :image_start].sum(0, keepdim=True) + 100
        post_score = all_attn[:, image_start + visual_token_num :].sum(0, keepdim=True) + 100

        stats["score_sums"][layer_idx] = torch.cat([pre_score, text_image_score, post_score], dim=1)
        stats["text_image_attn_sums"][layer_idx] = text_image_attns.reshape(-1).sum()

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
            "score_sums": tuple(stats["score_sums"]),
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

    def _prefill_from_online_stats(self, past_key_values, num_of_token=None, attentions=None):
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        forget_num = int(seq_len - num_of_token * (1 - self.ratio)) * self.layer_num
        if forget_num <= 0:
            print(f"{Fore.YELLOW}[WARNING] No KV to prune!{Fore.RESET}")
            return past_key_values

        text_start = attentions["text_start"]
        score_sum = torch.stack(list(attentions["score_sums"]))
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
