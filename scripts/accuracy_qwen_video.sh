# export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_TYPE=qwen-4B-video
export MAX_GENERATED_TOKENS=1500
export DECORD_EOF_RETRY_MAX=20480
export OPENAI_API_URL=https://api.deepseek.com/chat/completions
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export GPT_EVAL_MODEL_NAME=deepseek-chat
export HF_HUB_OFFLINE=1

ckpt=/root/autodl-tmp/weights/Qwen3-VL-4B-Instruct

export KV_CACHE_TYPE=tgv_kv
  for ratio in 0.5 0.8 0.9
  do
  export PRUNE_RATIO=$ratio
  echo ${KV_CACHE_TYPE}
  echo ${PRUNE_RATIO}
  accelerate launch --num_processes=4 --main_process_port 29502 -m lmms_eval --model qwen3_vl \
      --model_args "pretrained=$ckpt,attn_implementation=eager,max_pixels=203840" \
      --tasks "videott_no_leading_oe" --batch_size 1 --log_samples \
      --log_samples_suffix reproduce --output_path ./logs/
  done
