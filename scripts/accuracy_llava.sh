# # export HF_ENDPOINT=https://hf-mirror.com # If you encounter network issue, please uncomment this
export CUDA_VISIBLE_DEVICES=2,3
export MODEL_TYPE=llava-7B
# export HF_HUB_OFFLINE=1
export MAX_GENERATED_TOKENS=32
ckpt=/root/share/llava-1.5-7b-hf

export KV_CACHE_TYPE=tgv_kv
for ratio in 0.95 0.9 0.8 0.5
do
    export PRUNE_RATIO=$ratio
    echo ${KV_CACHE_TYPE}
    echo ${PRUNE_RATIO}
    accelerate launch --num_processes=2 --main_process_port 29508 -m lmms_eval --model llava_hf \
        --model_args "pretrained=$ckpt,attn_implementation=eager,device_map=cuda" \
        --tasks "chartqa,textvqa_val,docvqa_val,vizwiz_vqa_val" --batch_size 1 --log_samples \
        --log_samples_suffix reproduce --output_path ./logs/
done
