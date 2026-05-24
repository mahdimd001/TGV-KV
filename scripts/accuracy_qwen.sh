# export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_TYPE=qwen-8B
export IMAGE_MAX_TOKEN_NUM=1024
export MAX_GENERATED_TOKENS=32

ckpt=/root/share/Qwen3-VL-8B-Instruct


export KV_CACHE_TYPE=tgv_kv
  for ratio in 0.95 0.9 0.8 0.5
  do
  export PRUNE_RATIO=$ratio
  echo ${KV_CACHE_TYPE}
  echo ${PRUNE_RATIO}
  accelerate launch --num_processes=4 --main_process_port 29507 -m lmms_eval --model qwen3_vl \
      --model_args "pretrained=$ckpt,attn_implementation=eager" \
      --tasks "chartqa,textvqa_val,docvqa_val,vizwiz_vqa_val" --batch_size 1 --log_samples \
      --log_samples_suffix reproduce --output_path ./logs/
  done