import os

from colorama import Fore, Style

from .tgv_kv import TGVKVCache


def get_kv_cache(
    method="tgv_kv",
    start_size=4,
    recent_size=2047,
    k_seq_dim=2,
    v_seq_dim=2,
    prune_ratio=0.2,
    layer_num=36,
    model_name="llava-v1.5-7b",
):
    first_call = not hasattr(get_kv_cache, "_printed")
    if first_call:
        setattr(get_kv_cache, "_printed", True)

    if method is None or method.lower() == "none":
        if first_call:
            print(f"{Fore.RED}!!! No caching method is used. !!!{Style.RESET_ALL}")
        return None
    method = method.lower()

    if method not in ("tgv_kv", "tgv-kv", "tgvkv"):
        raise ValueError(f"Unsupported KV cache method: {method}. Only TGV-KV is available.")

    model_type = os.environ.get("MODEL_TYPE", None)
    if model_type == "llava-7B":
        image_token_id = 32000
        layer_num = 32
    elif model_type == "qwen-8B" or model_type == "qwen-4B":
        image_token_id = 151655
        layer_num = 36
    elif model_type == "qwen-8B-video" or model_type == "qwen-4B-video":
        image_token_id = 151656
        layer_num = 36
    elif model_type == "llava-ov-0.5B":
        image_token_id = 151646
        layer_num = 24
    elif model_type == "qwen2_5_vl_7B":
        image_token_id = 151655
        layer_num = 28
    elif model_type == "qwen2_5_vl_3B":
        image_token_id = 151655
        layer_num = 36
    else:
        raise NotImplementedError("Not supported model! Please manually set image_token_id and layer_num.")

    if method in ("tgv_kv", "tgv-kv", "tgvkv"):
        if first_call:
            print(f"{Fore.GREEN}+++ Using TGV-KV Cache +++{Style.RESET_ALL}")
        return TGVKVCache(
            image_token_id=image_token_id,
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
            ratio=prune_ratio,
            layer_num=layer_num,
        )
