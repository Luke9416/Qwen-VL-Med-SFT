#!/bin/bash

# 快速启动脚本 - 自动处理所有路径问题
echo "🚀 Quick Start Training Script"



# 项目根目录
PROJECT_ROOT=${YOUR_PATH}

echo "切换到项目目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 检查修复版脚本是否存在
if [ ! -f "train_sft_single_gpu_v0.py" ]; then
    echo "❌ train_sft_single_gpu_v0.py 不存在"
    echo "请将脚本放在项目根目录: $PROJECT_ROOT"
    exit 1
fi

echo "✅ 找到修复版训练脚本"

# 运行训练
echo "开始训练..."

python train_sft_single_gpu_eval.py \
    --use_liger False \
    --lora_enable True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules "['gate_proj', 'up_proj', 'down_proj']" \
    --model_id "output/staget1_merge/Qwen_2_VL_Med_stage1" \
    --data_path "data_path/clean_up/formatted_single_turn_dialogue_20K_valid_0.json" \
    --eval_data_path "data_path/valid_qa_47.json" \
    --image_folder "" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir "output/single_gpu_train_validDataset_stage2_instruct" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing False \
    --report_to "none" \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    --max_steps 100

# 记录结束信息
echo "=========================================="
echo "Training completed"
echo "Time: $(date)"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="