#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --job-name="qwen4_tgv"
#SBATCH --mail-user=msamani@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# === Step 1: Load Required Modules ===
module load python/3.10

# === Step 2: Activate Your Environment ===
source /lustre/hdd/LAS/jannesar-lab/msamani/pythonenv_tgv_kv/bin/activate

# === Step 3: Environment Variables ===
# export HF_ENDPOINT=https://hf-mirror.com  # Uncomment if network issues
# export HF_HUB_OFFLINE=1                   # Uncomment to use cached models only

export CUDA_VISIBLE_DEVICES=0,1             # SLURM assigns GPUs starting from 0
export MODEL_TYPE=qwen-4B
export MAX_GENERATED_TOKENS=32
export IMAGE_MAX_TOKEN_NUM=1024
export KV_CACHE_TYPE=tgv_kv

export HF_TOKEN=""
hf auth login --token $HF_TOKEN


export HF_HOME=/lustre/hdd/LAS/jannesar-lab/msamani/.cache/huggingface
export HF_DATASETS_CACHE=/lustre/hdd/LAS/jannesar-lab/msamani/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/lustre/hdd/LAS/jannesar-lab/msamani/.cache/huggingface/hub
export HF_HUB_CACHE=/lustre/hdd/LAS/jannesar-lab/msamani/.cache/huggingface/hub

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True





# === Step 4: Run Evaluation Loop ===
ratios=(0.95 0.9 0.8 0.5)

for ratio in "${ratios[@]}"; do

  export PRUNE_RATIO=$ratio
  echo "========================================"
  echo "Running: KV_CACHE_TYPE=${KV_CACHE_TYPE}  PRUNE_RATIO=${PRUNE_RATIO}"
  echo "========================================"

  accelerate launch \
    --num_processes=2 \
    --main_process_port 29508 \
    -m lmms_eval \
    --model qwen3_vl \
    --model_args "pretrained=Qwen/Qwen3-VL-4B-Instruct,attn_implementation=eager,device_map=cuda" \
    --tasks "chartqa,textvqa_val,docvqa_val,vizwiz_vqa_val" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix reproduce \
    --output_path ./logs_qwen/

done
