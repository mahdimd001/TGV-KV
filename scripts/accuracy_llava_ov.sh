# # export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=2,3
export MODEL_TYPE=llava-ov-0.5B
export VISION_ASPECT_RATIO=anyres_max_2
export HF_HUB_OFFLINE=1
export MAX_GENERATED_TOKENS=64
ckpt=/root/share/llava-onevision-qwen2-0.5b-ov-hf

export KV_CACHE_TYPE=tgv_kv
  for ratio in 0.5 0.8 0.9 0.95
  do
  export PRUNE_RATIO=$ratio
  echo ${KV_CACHE_TYPE}
  echo ${PRUNE_RATIO}
  accelerate launch --num_processes=2 --main_process_port 29507 -m lmms_eval --model llava_hf \
      --model_args "pretrained=$ckpt,attn_implementation=eager" \
      --tasks "chartqa,textvqa_val,docvqa_val,vizwiz_vqa_val" --batch_size 1 --log_samples \
      --log_samples_suffix reproduce --output_path ./logs/
  done