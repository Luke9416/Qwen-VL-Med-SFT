from tensorboardX import SummaryWriter
import datetime
import logging
import os
import sys
from transformers import TrainerCallback
import psutil
import torch

def init_tensorboard(training_args, model_args, data_args):
    """初始化TensorBoard"""
    # 创建带时间戳的日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{training_args.output_dir}/tensorboard_{timestamp}"
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # 记录超参数
    hparams = {
        "model_name": model_args.model_id.split('/')[-1],
        "lora_rank": training_args.lora_rank,
        "lora_alpha": training_args.lora_alpha,
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "max_steps": training_args.max_steps,
        "freeze_vision_tower": training_args.freeze_vision_tower,
        "freeze_llm": training_args.freeze_llm,
    }
    
    # 记录超参数到TensorBoard
    writer.add_hparams(hparams, {})
    
    print(f"📊 TensorBoard日志目录: {log_dir}")
    print(f"📊 启动TensorBoard: tensorboard --logdir {log_dir}")
    
    return writer


def rank0_print(*args):
    """打印函数"""
    print(*args)


class TensorBoardCallback(TrainerCallback):
    """增强的TensorBoard回调 - 包含文件日志功能"""
    
    def __init__(self, tb_writer=None, output_dir=None):
        self.tb_writer = tb_writer
        self.output_dir = output_dir
        self.start_time = None
        self.last_log_step = 0
        
        # GPU信息
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
            self.current_device = torch.cuda.current_device()
        
        # 设置文件日志
        self.file_logger = None
        if output_dir:
            self._setup_file_logger(output_dir)
    
    def _setup_file_logger(self, output_dir):
        """设置文件日志记录器"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建logger
        self.file_logger = logging.getLogger("QwenTraining")
        self.file_logger.setLevel(logging.INFO)
        
        # 清除已有的handlers
        self.file_logger.handlers = []
        
        # 文件handler
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # 添加handler
        self.file_logger.addHandler(file_handler)
        
        # 记录初始信息
        self.file_logger.info("=" * 60)
        self.file_logger.info("QWEN2-VL TRAINING LOG")
        self.file_logger.info("=" * 60)
        self.file_logger.info(f"Log file: {log_file}")
        self.file_logger.info(f"Output directory: {output_dir}")
    
    def log_print(self, *args, **kwargs):
        """同时输出到控制台和文件"""
        message = " ".join(str(arg) for arg in args)
        
        # 控制台输出
        print(message, **kwargs)
        
        # 文件日志输出
        if self.file_logger:
            self.file_logger.info(message)
    
    def get_memory_info(self):
        """获取内存使用信息"""
        memory_info = {}
        
        # CPU内存
        try:
            cpu_memory = psutil.virtual_memory()
            memory_info.update({
                'cpu_memory_used_gb': cpu_memory.used / (1024**3),
                'cpu_memory_total_gb': cpu_memory.total / (1024**3),
                'cpu_memory_percent': cpu_memory.percent,
                'cpu_memory_available_gb': cpu_memory.available / (1024**3)
            })
        except Exception as e:
            self.log_print(f"⚠️  CPU内存获取失败: {e}")
        
        # GPU内存
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_stats(self.current_device)
                memory_info.update({
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated(self.current_device) / (1024**3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved(self.current_device) / (1024**3),
                    'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated(self.current_device) / (1024**3),
                    'gpu_memory_max_reserved_gb': torch.cuda.max_memory_reserved(self.current_device) / (1024**3),
                })
                
                # GPU使用率（如果有nvidia-ml-py库）
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.current_device)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info['gpu_utilization_percent'] = gpu_util.gpu
                    memory_info['gpu_memory_utilization_percent'] = gpu_util.memory
                except ImportError:
                    pass  # pynvml not available
                except Exception as e:
                    pass  # GPU监控失败
                    
            except Exception as e:
                self.log_print(f"⚠️  GPU内存获取失败: {e}")
        
        return memory_info
    
    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时记录训练配置"""
        if self.file_logger:
            self.file_logger.info("\nTraining Configuration:")
            self.file_logger.info(f"  Model: {kwargs.get('model', 'N/A')}")
            self.file_logger.info(f"  Learning Rate: {args.learning_rate}")
            self.file_logger.info(f"  Batch Size: {args.per_device_train_batch_size}")
            self.file_logger.info(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
            self.file_logger.info(f"  Max Steps: {args.max_steps}")
            self.file_logger.info(f"  Evaluation Strategy: {args.evaluation_strategy}")
            if args.evaluation_strategy != "no":
                self.file_logger.info(f"  Eval Steps: {args.eval_steps}")
            self.file_logger.info("=" * 60)
    
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        self.log_print("🚀 Training started!")
        
        if self.tb_writer:
            self.tb_writer.add_text("training/status", "Training started", 0)
            
            # 记录初始内存状态
            memory_info = self.get_memory_info()
            for key, value in memory_info.items():
                self.tb_writer.add_scalar(f"memory/{key}", value, 0)
        
        # 更新output_dir（如果trainer中设置了）
        if hasattr(args, 'output_dir') and args.output_dir and not self.file_logger:
            self._setup_file_logger(args.output_dir)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        current_step = state.global_step
        if current_step <= self.last_log_step:
            return
        self.last_log_step = current_step
        
        # 计算训练速度
        import time
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0
        
        # 获取内存信息
        memory_info = self.get_memory_info()
        epoch = state.epoch if state.epoch else 0
        
        # 获取GPU内存
        if int(os.environ.get("LOCAL_RANK", -1)) == -1:
            max_gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 if self.gpu_available else 0
        else:
            local_rank = int(os.environ["LOCAL_RANK"]) 
            device = torch.device(f"cuda:{local_rank}")
            max_gpu_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        
        # 准备日志信息
        loss = logs.get('train_loss', logs.get('loss', 'N/A'))
        lr = logs.get('learning_rate', 'N/A')
        grad_norm = logs.get('grad_norm', 'N/A')
        
        # 构建日志消息
        log_parts = [f"📈 Step {current_step}: "]
        if isinstance(loss, (int, float)):
            log_parts.append(f"Loss={loss:.4f}")
        else:
            log_parts.append(f"Loss={loss}")
            
        if isinstance(lr, (int, float)):
            log_parts.append(f"LR={lr:.2e}")
        else:
            log_parts.append(f"LR={lr}")
            
        if isinstance(grad_norm, (int, float)):
            log_parts.append(f"GradNorm={grad_norm:.2f}")
        else:
            log_parts.append(f"GradNorm={grad_norm}")
            
        log_parts.extend([
            f"Speed={steps_per_sec:.2f} steps/s",
            f"Epoch={epoch:.2f}",
            f"GPU MaxMemory={max_gpu_memory:.2f} MB"
        ])
        
        # 添加内存信息
        cpu_mem = memory_info.get('cpu_memory_percent', 0)
        gpu_mem = memory_info.get('gpu_memory_allocated_gb', 0)
        log_parts.extend([
            f"CPU={cpu_mem:.1f}%",
            f"GPU={gpu_mem:.2f}GB"
        ])
        
        # 定期输出日志（每20步）
        if current_step % 20 == 0:
            self.log_print(", ".join(log_parts))
        
        # 记录评估结果
        eval_logs = {k: v for k, v in logs.items() if k.startswith('eval_')}
        if eval_logs:
            self.log_print(f"\n📊 Evaluation at step {current_step}:")
            for key, value in eval_logs.items():
                if isinstance(value, (int, float)):
                    self.log_print(f"  {key}: {value:.4f}")
                else:
                    self.log_print(f"  {key}: {value}")
            self.log_print("")  # 空行
        
        # TensorBoard记录
        if self.tb_writer:
            try:
                # 原有训练指标
                for key, value in logs.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f"training/{key}", value, current_step)
                
                # 训练速度
                self.tb_writer.add_scalar("training/steps_per_second", steps_per_sec, current_step)
                self.tb_writer.add_scalar("training/elapsed_minutes", elapsed / 60, current_step)
                
                # 内存使用情况
                for key, value in memory_info.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(f"memory/{key}", value, current_step)
                
                # 每50步刷新一次
                if current_step % 50 == 0:
                    self.tb_writer.flush()
                    
            except Exception as e:
                if current_step % 100 == 0:
                    self.log_print(f"⚠️  TensorBoard记录出错: {e}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """评估完成时记录"""
        if metrics:
            self.log_print(f"\n✅ Evaluation completed at step {state.global_step}")
            # 详细指标已在on_log中记录
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时记录"""
        self.log_print(f"\n💾 Model checkpoint saved at step {state.global_step}")
        if hasattr(state, 'best_metric') and state.best_metric is not None:
            self.log_print(f"  Best metric so far: {state.best_metric:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        self.log_print("\n" + "=" * 60)
        self.log_print("✅ Training completed!")
        self.log_print(f"Total steps: {state.global_step}")
        
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.log_print(f"Total time: {hours}h {minutes}m {seconds}s")
        
        if hasattr(state, 'best_metric') and state.best_metric is not None:
            self.log_print(f"Best metric: {state.best_metric:.4f}")
        if hasattr(state, 'best_model_checkpoint') and state.best_model_checkpoint:
            self.log_print(f"Best checkpoint: {state.best_model_checkpoint}")
        
        self.log_print("=" * 60)
        
        if self.tb_writer:
            # 记录最终内存状态
            memory_info = self.get_memory_info()
            for key, value in memory_info.items():
                self.tb_writer.add_scalar(f"memory_final/{key}", value, state.global_step)
            
            self.tb_writer.add_text("training/status", "Training completed", state.global_step)
            self.tb_writer.close()
            self.log_print("📊 TensorBoard日志已关闭")
        
        # 关闭文件日志
        if self.file_logger:
            self.file_logger.info("\nTraining log closed.")
            for handler in self.file_logger.handlers:
                handler.close()
