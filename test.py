print(44)

from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    cache_dir="./models"
)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir="./models"
)


# # LLaVA-1.5-7B evaluation
export CUDA_VISIBLE_DEVICES=0
export MODEL_TYPE=llava-7B
export MAX_GENERATED_TOKENS=32
export KV_CACHE_TYPE=tgv_kv
export PRUNE_RATIO=0.50

# accelerate launch --num_processes=2 -m lmms_eval --model llava_hf \
#     --model_args "pretrained=llava-7B,attn_implementation=eager,device_map=cuda" \
#     --tasks "chartqa,textvqa_val,docvqa_val,vizwiz_vqa_val" \
#     --batch_size 1 --log_samples --output_path ./logs/


accelerate launch --num_processes=2 -m lmms_eval --model llava_hf     --model_args "pretrained=llava-hf/llava-1.5-7b-hf,attn_implementation=eager,device_map=cuda"     --tasks "chartqa"     --batch_size 1 --log_samples --output_path ./logs/ 

accelerate launch --num_processes=1 -m lmms_eval --model llava_hf \
    --model_args "pretrained=llava-hf/llava-1.5-7b-hf,attn_implementation=eager,device_map=cuda" \
    --tasks "chartqa" \
    --batch_size 1 --log_samples --output_path ./logs/


    llava-hf/llava-1.5-7b-hf


