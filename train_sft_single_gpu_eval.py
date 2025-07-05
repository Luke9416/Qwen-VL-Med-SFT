import os
import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback as HFTensorBoardCallback
from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_eval_data_module
from src.params_eval import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from src.eval.evaluation_0702 import create_compute_metrics_with_types, load_type_mapping

import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from src.train.monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from src.utils import inspect_lora_model
from src.logger_utils import init_tensorboard, TensorBoardCallback

import logging
import sys
from datetime import datetime
import json
import numpy as np




# 设置单卡调试环境
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
local_rank = 0

# 全局logger变量
logger = None


class EvaluationCallback(TrainerCallback):
    """自定义评估回调，确保评估被执行并记录结果"""
    
    def __init__(self, trainer, eval_dataset, compute_metrics, output_dir, logger):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.output_dir = output_dir
        self.logger = logger
        self.eval_history = []
        
    def on_step_end(self, args, state, control, **kwargs):
        """在每个步骤结束时检查是否需要评估"""
        # 检查是否到了评估步骤
        if args.evaluation_strategy == "steps" and state.global_step % args.eval_steps == 0:
            self.logger.info(f"🔍 Step {state.global_step}: Running evaluation...")
            
            # 强制执行评估
            metrics = self.trainer.evaluate(
                eval_dataset=self.eval_dataset,
                metric_key_prefix="eval"
            )
            
            # 记录评估结果
            eval_result = {
                "step": state.global_step,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.eval_history.append(eval_result)
            
            # 保存评估历史
            with open(os.path.join(self.output_dir, "eval_history.json"), "w") as f:
                json.dump(self.eval_history, f, indent=2)
            
            # 打印关键指标
            self.logger.info(f"📊 Evaluation Results at Step {state.global_step}:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
            
            # 检查是否是最佳模型
            if args.metric_for_best_model in metrics:
                current_metric = metrics[args.metric_for_best_model]
                if not hasattr(state, 'best_metric') or state.best_metric is None:
                    state.best_metric = current_metric
                    state.best_model_checkpoint = state.global_step
                    self.logger.info(f"🏆 New best model at step {state.global_step}!")
                elif (args.greater_is_better and current_metric > state.best_metric) or \
                     (not args.greater_is_better and current_metric < state.best_metric):
                    state.best_metric = current_metric
                    state.best_model_checkpoint = state.global_step
                    self.logger.info(f"🏆 New best model at step {state.global_step}!")
        
        return control


def setup_logger(output_dir: str, log_file: str = "training.log"):
    """设置日志系统，同时输出到文件和控制台"""
    global logger
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger("QwenTraining")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 文件handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, log_file),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_logger_with_tensorboard_compatibility(output_dir, log_file="training.log"):
    """创建与 TensorBoard 兼容的 logger"""
    # 获取当前活跃的 loggers
    root_logger = logging.getLogger()
    
    # 创建新的 logger
    new_logger = logging.getLogger('QwenTraining')
    new_logger.setLevel(logging.INFO)
    new_logger.propagate = False  # 不传播到根 logger
    new_logger.handlers.clear()
    
    # 检查根 logger 是否有 handlers（TensorBoard 可能添加的）
    if root_logger.handlers:
        # 复用根 logger 的控制台 handler 格式
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                # 创建相同格式的 handler
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(handler.level)
                console_handler.setFormatter(handler.formatter)
                new_logger.addHandler(console_handler)
                break
    else:
        # 创建默认的控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler.setFormatter(formatter)
        new_logger.addHandler(console_handler)
    
    # 添加文件 handler
    file_handler = logging.FileHandler(
        os.path.join(output_dir, log_file),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    new_logger.addHandler(file_handler)
    
    return new_logger


def rank0_print(*args):
    """使用logger输出信息"""
    global logger
    message = " ".join(str(arg) for arg in args)
    if logger:
        logger.info(message)
    else:
        print(message)

def verify_lora_weights_loaded(model, expected_modules=None):
    """验证 LoRA 权重是否成功加载"""
    rank0_print("\n" + "="*60)
    rank0_print("🔍 验证 LoRA 加载情况")
    rank0_print("="*60)
    
    lora_modules_found = {}
    total_lora_params = 0
    
    # 检查所有参数
    for name, param in model.named_parameters():
        if "lora_" in name:
            # 提取模块路径
            module_path = name.split(".lora_")[0]
            base_module = module_path.split(".")[-1]
            
            if module_path not in lora_modules_found:
                lora_modules_found[module_path] = {
                    'base_module': base_module,
                    'lora_A': False,
                    'lora_B': False,
                    'params': 0
                }
            
            if "lora_A" in name:
                lora_modules_found[module_path]['lora_A'] = True
            elif "lora_B" in name:
                lora_modules_found[module_path]['lora_B'] = True
            
            lora_modules_found[module_path]['params'] += param.numel()
            total_lora_params += param.numel()
    
    # 打印找到的 LoRA 模块
    if lora_modules_found:
        rank0_print(f"\n✅ 找到 {len(lora_modules_found)} 个 LoRA 模块:")
        for path, info in sorted(lora_modules_found.items()):
            status = "✓" if (info['lora_A'] and info['lora_B']) else "✗"
            rank0_print(f"  {status} {path}")
            rank0_print(f"     - 基础模块: {info['base_module']}")
            rank0_print(f"     - LoRA A: {'✓' if info['lora_A'] else '✗'}")
            rank0_print(f"     - LoRA B: {'✓' if info['lora_B'] else '✗'}")
            rank0_print(f"     - 参数数: {info['params']:,}")
    else:
        rank0_print("❌ 未找到任何 LoRA 模块")
        return False
    
    rank0_print(f"\n📊 总 LoRA 参数数: {total_lora_params:,}")
    
    # 检查预期模块
    if expected_modules:
        rank0_print(f"\n🎯 检查预期模块:")
        all_found = True
        for expected in expected_modules:
            found = any(expected in path for path in lora_modules_found.keys())
            if found:
                rank0_print(f"  ✅ {expected}")
            else:
                rank0_print(f"  ❌ {expected}")
                all_found = False
        
        return all_found and len(lora_modules_found) > 0
    
    return len(lora_modules_found) > 0

def load_and_merge_existing_lora(model, lora_model_id, device, compute_dtype):
    """加载已有的 LoRA 权重并合并到基础模型"""
    rank0_print("\n" + "="*60)
    rank0_print("🔄 加载并合并已有 LoRA 权重")
    rank0_print("="*60)
    
    try:
        # 先检查路径是否存在
        if not os.path.exists(lora_model_id):
            rank0_print(f"❌ LoRA 路径不存在: {lora_model_id}")
            return model, False
        
        # 检查必要文件
        adapter_config_path = os.path.join(lora_model_id, "adapter_config.json")
        adapter_model_path = os.path.join(lora_model_id, "adapter_model.bin")
        
        if not os.path.exists(adapter_config_path):
            adapter_model_path = os.path.join(lora_model_id, "adapter_model.safetensors")
            if not os.path.exists(adapter_model_path):
                rank0_print("❌ 未找到 adapter_model.bin 或 adapter_model.safetensors")
                return model, False
        
        # 读取适配器配置
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
            
        rank0_print(f"📋 加载的 LoRA 配置:")
        rank0_print(f"  - rank: {adapter_config.get('r', 'N/A')}")
        rank0_print(f"  - alpha: {adapter_config.get('lora_alpha', 'N/A')}")
        rank0_print(f"  - target_modules: {adapter_config.get('target_modules', 'N/A')}")
        
        # 获取原始的 target_modules
        original_target_modules = adapter_config.get('target_modules', [])
        
        # 使用 PeftModel 加载
        rank0_print(f"\n📥 从 {lora_model_id} 加载 LoRA 权重...")
        peft_model = PeftModel.from_pretrained(
            model, 
            lora_model_id,
            torch_dtype=compute_dtype,
            device_map={"": device}
        )
        
        # 验证加载是否成功
        loaded_successfully = verify_lora_weights_loaded(peft_model, original_target_modules)
        
        if loaded_successfully:
            rank0_print("\n✅ LoRA 权重加载成功!")
            
            # 合并 LoRA 权重到基础模型
            rank0_print("\n🔀 合并 LoRA 权重到基础模型...")
            merged_model = peft_model.merge_and_unload()
            rank0_print("✅ 合并完成!")
            
            # 验证合并后的模型
            rank0_print("\n🔍 验证合并后的模型:")
            total_params = sum(p.numel() for p in merged_model.parameters())
            rank0_print(f"  - 总参数数: {total_params:,}")
            
            # 检查合并后不应该有 LoRA 层
            lora_layers_after_merge = sum(1 for n, _ in merged_model.named_parameters() if "lora_" in n)
            if lora_layers_after_merge == 0:
                rank0_print("  - ✅ 合并成功：没有残留的 LoRA 层")
            else:
                rank0_print(f"  - ⚠️ 警告：发现 {lora_layers_after_merge} 个残留的 LoRA 层")
            
            return merged_model, True
        else:
            rank0_print("❌ LoRA 权重加载失败")
            return model, False
            
    except Exception as e:
        rank0_print(f"❌ 加载 LoRA 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return model, False


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def print_model_info(model):
    """打印模型信息用于调试"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    rank0_print(f"Model loaded successfully!")
    rank0_print(f"Total parameters: {total_params:,}")
    rank0_print(f"Trainable parameters: {trainable_params:,}")
    rank0_print(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    # 打印一些可训练参数的名称
    rank0_print("\nSample trainable parameters:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and count < 10:
            rank0_print(f"  {name}: {param.shape}")
            count += 1


def setup_evaluation(data_args, training_args):
    """设置验证配置"""
    # 检查是否有验证数据
    eval_data_path = getattr(data_args, 'eval_data_path', None)
    
    if eval_data_path and os.path.exists(eval_data_path):
        rank0_print(f"📊 Found evaluation data: {eval_data_path}")
        
        # 检查是否有对应的type_mapping文件
        type_mapping_file = None
        possible_mapping_files = [
            eval_data_path.replace('.json', '_mapping.json'),
            eval_data_path.replace('.json', '_type_mapping.json'),
            os.path.join(os.path.dirname(eval_data_path), 'type_mapping.json'),
            os.path.join(os.path.dirname(eval_data_path), 'validation_mapping.json')
        ]
        
        for mapping_file in possible_mapping_files:
            if os.path.exists(mapping_file):
                type_mapping_file = mapping_file
                rank0_print(f"📋 Found type mapping: {mapping_file}")
                break
        
        if not type_mapping_file:
            rank0_print("⚠️ No type mapping file found, using standard evaluation")
        
        # 强制启用验证相关参数
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = max(1, training_args.logging_steps)  # 确保eval_steps有值
        training_args.metric_for_best_model = "eval_soft_match" if training_args.metric_for_best_model is None else training_args.metric_for_best_model
        training_args.greater_is_better = True if training_args.greater_is_better is None else training_args.greater_is_better
        training_args.load_best_model_at_end = True
        training_args.save_total_limit = 3 if training_args.save_total_limit is None else training_args.save_total_limit
        
        # 确保do_eval为True
        training_args.do_eval = True
        
        rank0_print(f"✅ Evaluation enabled:")
        rank0_print(f"   - Strategy: {training_args.evaluation_strategy}")
        rank0_print(f"   - Eval steps: {training_args.eval_steps}")
        rank0_print(f"   - Metric: {training_args.metric_for_best_model}")
        rank0_print(f"   - Do eval: {training_args.do_eval}")
        
        return True, type_mapping_file
    else:
        rank0_print("⚠️ No evaluation data found, skipping validation")
        training_args.evaluation_strategy = "no"
        training_args.do_eval = False
        return False, None


def train():
    global local_rank, logger
    
    # lora_target_modules = ["visual.merger.mlp.0", "visual.merger.mlp.2"]
    
    # 在解析参数之前就禁用分布式训练
    os.environ['LOCAL_RANK'] = '-1'
    os.environ['WORLD_SIZE'] = '1' 
    os.environ['RANK'] = '0'
    # 清除可能导致分布式初始化的环境变量
    for key in ['MASTER_ADDR', 'MASTER_PORT', 'NODE_RANK']:
        os.environ.pop(key, None)

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ['labels']
    training_args.include_for_metrics = ['inputs']
    training_args.batch_eval_metrics = True
    
    
    
    # 设置日志系统
    # logger = setup_logger(training_args.output_dir)
    # original_logger = logger
    # original_handlers = logger.handlers[:] if logger else []
    
    # 保存配置信息到文件
    config_info = {
        "model_args": vars(model_args),
        "data_args": vars(data_args),
        "training_args": training_args.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv)
    }
    with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    # 获取设备信息
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 设置验证配置 - 在加载模型之前
    has_eval_data, type_mapping_file = setup_evaluation(data_args, training_args)
    # 初始化TensorBoard
    tb_writer = init_tensorboard(training_args, model_args, data_args)
    progress_callback = TensorBoardCallback(tb_writer, output_dir=training_args.output_dir)

    logger = setup_logger_with_tensorboard_compatibility(training_args.output_dir)
    
    # 调试信息
    rank0_print("=" * 60)
    rank0_print("QWEN2.5-VL TRAINING WITH EVALUATION")
    rank0_print("=" * 60)
    rank0_print(f"🔧 Device: {device}")
    rank0_print(f"🤖 Model: {model_args.model_id}")
    rank0_print(f"📁 Output dir: {training_args.output_dir}")
    rank0_print(f"🎯 LoRA enabled: {training_args.lora_enable}")
    rank0_print(f"📊 Evaluation: {'✅ Enabled' if has_eval_data else '❌ Disabled'}")
    if type_mapping_file:
        rank0_print(f"📋 Type-based evaluation: ✅ Enabled")
    elif has_eval_data:
        rank0_print(f"📋 Type-based evaluation: ⚠️ No mapping file")
    rank0_print(f"⚡ Max steps: {training_args.max_steps}")
    rank0_print(f"📦 Batch size: {training_args.per_device_train_batch_size}")
    rank0_print(f"🔄 Gradient accumulation: {training_args.gradient_accumulation_steps}")
    rank0_print(f"📈 Learning rate: {training_args.learning_rate}")
    if has_eval_data:
        rank0_print(f"🔍 Eval frequency: every {training_args.eval_steps} steps")
        rank0_print(f"🏆 Best model metric: {training_args.metric_for_best_model}")
    rank0_print("=" * 60)
    


    

    
    # 应用优化
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)

    # 验证参数配置
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    # 处理LoRA排除列表
    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = []
    if training_args.lora_target_modules is not None:
        training_args.lora_target_modules = ast.literal_eval(training_args.lora_target_modules)
    else:
        training_args.lora_target_modules = []
    lora_target_modules = training_args.lora_target_modules
    
    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # 量化配置
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # 加载模型
    rank0_print("🤖 Loading model...")
    # if model_args.lora_model_id is None:
    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
            )


    model.config.use_cache = False
    model_to_configure = model
    
    # 配置模型组件
    rank0_print("⚙️ Configuring model components...")
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    # 量化训练准备
    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if model_args.lora_model_id is not None:
        model, merge_success = load_and_merge_existing_lora(
            model, 
            model_args.lora_model_id, 
            device, 
            compute_dtype
        )
        if not merge_success:
            rank0_print("⚠️ 继续使用原始模型进行训练...")

    # 设置LoRA
    if training_args.lora_enable:
        rank0_print("🎯 Setting up LoRA...")
        # lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        rank0_print("\n📊 新 LoRA 配置信息:")
        rank0_print(f"  - 目标模块: {lora_target_modules}")
        rank0_print(f"  - Rank: {training_args.lora_rank}")
        rank0_print(f"  - Alpha: {training_args.lora_alpha}")
        rank0_print(f"  - Dropout: {training_args.lora_dropout}")

        verify_lora_weights_loaded(model, lora_target_modules)
        model.print_trainable_parameters()
        
        # 确保vision相关参数的设置
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
                    

    # 检查LoRA配置
    inspect_lora_model(model)
    print_model_info(model)

    # 加载处理器
    rank0_print("🔧 Loading processor...")
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # 量化训练的额外设置
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # 创建数据模块
    rank0_print("📊 Creating data module...")
    data_module = make_supervised_eval_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)
    
    # 检查是否成功加载了验证数据集
    eval_dataset = data_module.get('eval_dataset', None)
    if has_eval_data and eval_dataset is None:
        rank0_print("⚠️ Warning: Evaluation requested but no eval dataset created!")
        has_eval_data = False
        training_args.evaluation_strategy = "no"
        training_args.do_eval = False
    elif eval_dataset is not None:
        rank0_print(f"✅ Evaluation dataset loaded: {len(eval_dataset)} samples")

    # 设置评估函数
    compute_metrics = None
    if has_eval_data and eval_dataset is not None:
        rank0_print("🔍 Setting up evaluation metrics...")
        
        if type_mapping_file:
            rank0_print(f"📋 Using type-based evaluation with: {type_mapping_file}")
            # 预加载type_mapping
            load_type_mapping(type_mapping_file)
            compute_metrics = create_compute_metrics_with_types(
                processor, 
                training_args.output_dir, 
                type_mapping_file,
                eval_dataset=eval_dataset
            )
        else:
            rank0_print("📊 Using standard evaluation")
            compute_metrics = create_compute_metrics_with_types(
                processor, 
                training_args.output_dir,
                eval_dataset=eval_dataset
            )


    # 创建训练器
    rank0_print("🚀 Creating trainer...")
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module
    )
    
    # 添加回调
    callbacks_to_add = []
    
    # TensorBoard回调
    if hasattr(trainer, 'add_callback'):
        callbacks_to_add.append(progress_callback)
    
    # 添加评估回调
    if has_eval_data and eval_dataset is not None:
        eval_callback = EvaluationCallback(
            trainer=trainer,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            output_dir=training_args.output_dir,
            logger=logger
        )
        callbacks_to_add.append(eval_callback)
        rank0_print("✅ Added evaluation callback")
    


    # 添加所有回调
    for callback in callbacks_to_add:
        if hasattr(trainer, 'add_callback'):
            trainer.add_callback(callback)
    
    rank0_print(f"📎 Added {len(callbacks_to_add)} callbacks to trainer")
    
    # 开始训练前，先运行一次评估作为baseline
    
    
    if has_eval_data and eval_dataset is not None:
        rank0_print("📊 Running initial evaluation as baseline...")
        try:
            initial_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            rank0_print("📊 Initial Evaluation Results:")
            for key, value in initial_metrics.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    rank0_print(f"  {key}: {value:.4f}")
        except Exception as e:
            rank0_print(f"⚠️ Initial evaluation failed: {e}")
    
    # 开始训练
    rank0_print("🚀 Starting training...")
    try:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            rank0_print("🔄 Found existing checkpoint, resuming training...")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        raise

    # 保存模型
    rank0_print("💾 Saving final model...")
    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        # 保存LoRA模型
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        processor.save_pretrained(training_args.output_dir)
        torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
        rank0_print(f"✅ LoRA model saved to {training_args.output_dir}")
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
        rank0_print(f"✅ Full model saved to {training_args.output_dir}")

    # 最终验证
    if has_eval_data and eval_dataset is not None:
        rank0_print("🔍 Running final evaluation...")
        try:
            final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
            rank0_print("📊 Final Evaluation Results:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    rank0_print(f"  {key}: {value:.4f}")
            
            # 计算改进
            if 'initial_metrics' in locals():
                rank0_print("\n📈 Improvements:")
                for key in final_metrics:
                    if key in initial_metrics and isinstance(final_metrics[key], (int, float)):
                        improvement = final_metrics[key] - initial_metrics[key]
                        rank0_print(f"  {key}: {improvement:+.4f}")
            
            # 保存最终评估结果
            with open(os.path.join(training_args.output_dir, "final_evaluation.json"), "w") as f:
                json.dump(final_metrics, f, indent=2)
                
        except Exception as e:
            rank0_print(f"⚠️ Final evaluation failed: {e}")

       # 记录训练完成信息
    training_summary = {
        "status": "completed",
        "completion_time": datetime.now().isoformat(),
        "total_steps": trainer.state.global_step,
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
    }
    
    with open(os.path.join(training_args.output_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2)

    rank0_print("🎉" + "=" * 58 + "🎉")
    rank0_print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    rank0_print("🎉" + "=" * 58 + "🎉")
    rank0_print(f"📁 All results saved in: {training_args.output_dir}")
    rank0_print(f"📄 Check training.log for complete logs")
    rank0_print(f"📊 Check evaluation_history.json for eval metrics")
    rank0_print(f"📈 Use TensorBoard to visualize: tensorboard --logdir {training_args.output_dir}/logs")



if __name__ == "__main__":
    train()