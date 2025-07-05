#!/bin/bash

# 保守的多卡训练配置 - 适合从单卡调试过渡
echo "Starting conservative multi-GPU training..."

# ================================
# 保守配置 (降低风险)
# ================================


# 硬件配置 - 先用2卡测试
NUM_GPUS=2
GPU_IDS="0,1" 

# 批次配置 - 保守设置
GLOBAL_BATCH_SIZE=16                  # 较小的全局批次
BATCH_PER_DEVICE=2                    # 每卡批次保持和单卡一致
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_GPUS)))

# 输出配置
OUTPUT_DIR="output/conservative_multi_gpu"

echo "Conservative Multi-GPU Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Global Batch: $GLOBAL_BATCH_SIZE"  
echo "  Per-device Batch: $BATCH_PER_DEVICE"
echo "  Grad Accumulation: $GRAD_ACCUM_STEPS"
echo ""

# 环境设置
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$GPU_IDS

mkdir -p $OUTPUT_DIR

--save_steps 500 \           # 每多少步保存一次（当 save_strategy="steps" 时）
--save_total_limit 3 \       # 最多保留几个检查点

# 保守的训练配置
deepspeed --num_gpus=$NUM_GPUS train_sft_mult.py \
    --use_liger False \
    --lora_enable True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --model_id "/Qwen/Qwen2-VL-2B-Instruct" \
    --data_path "/clean_up/valid_data.json" \
    --image_folder "" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "output/mult_gpu_train_validDataset_0702" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --gradient_checkpointing False \
    --report_to "none" \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    

echo "Conservative training completed!"