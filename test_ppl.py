"""
推理和困惑度计算脚本
用于运行LoRA和Base模型的推理，计算困惑度，并保存详细结果
"""

import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import math

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
from PIL import Image

# 导入自定义模块
from src.dataset import SupervisedDataset
from src.params_eval import DataArguments


class PerplexityCalculator:
    """困惑度计算器"""
    
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
    
    def calculate_perplexity(
        self, 
        input_text: str, 
        target_text: str,
        images: Optional[List] = None
    ) -> Tuple[float, float]:
        """
        计算困惑度
        
        Args:
            input_text: 输入文本
            target_text: 目标文本
            images: 图像列表
            
        Returns:
            (困惑度, 负对数似然)
        """
        try:
            # 方法1: 使用chat template格式
            if images is not None:
                # 构建带图像的消息格式
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": input_text}
                    ]
                }]
                
                # 应用chat template
                input_formatted = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 完整文本（输入+目标）
                full_text = input_formatted + target_text
                
                # 处理输入
                inputs = self.processor(
                    text=[full_text],
                    images=images,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # 计算仅输入部分的长度
                input_only = self.processor(
                    text=[input_formatted],
                    images=images,
                    padding=True,
                    return_tensors="pt"
                )
                input_length = input_only.input_ids.shape[1]
                
            else:
                # 无图像情况
                messages = [{
                    "role": "user", 
                    "content": input_text
                }]
                
                input_formatted = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                full_text = input_formatted + target_text
                
                inputs = self.processor(
                    text=[full_text],
                    images=None,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                input_only = self.processor(
                    text=[input_formatted],
                    images=None,
                    padding=True,
                    return_tensors="pt"
                )
                input_length = input_only.input_ids.shape[1]
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 获取目标部分的logits和labels
                if input_length >= inputs.input_ids.shape[1]:
                    # 如果输入长度过长，返回默认值
                    return float('inf'), float('inf')
                
                target_logits = logits[0, input_length-1:-1, :]  # 预测下一个token
                target_ids = inputs.input_ids[0, input_length:]    # 真实的目标token
                
                if target_logits.shape[0] == 0 or target_ids.shape[0] == 0:
                    return float('inf'), float('inf')
                
                # 计算交叉熵损失
                loss = F.cross_entropy(target_logits, target_ids, reduction='mean')
                
                # 计算困惑度
                perplexity = torch.exp(loss).item()
                nll = loss.item()
                
                return perplexity, nll
                
        except Exception as e:
            # 如果出错，尝试方法2
            print(f"⚠️ Method 1 failed, trying method 2: {e}")
            return self._calculate_perplexity_v2(input_text, target_text, images)
    
    def _calculate_perplexity_v2(
        self, 
        input_text: str, 
        target_text: str,
        images: Optional[List] = None
    ) -> Tuple[float, float]:
        """
        困惑度计算备用方法
        """
        try:
            # 方法2: 直接拼接文本
            if images is not None:
                input_formatted = f"<image>\n{input_text}"
                full_text = f"<image>\n{input_text}{target_text}"
            else:
                input_formatted = input_text
                full_text = f"{input_text}{target_text}"
            
            # 处理完整输入
            inputs = self.processor(
                text=[full_text],
                images=images,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            
            # 处理仅输入部分
            input_only = self.processor(
                text=[input_formatted],
                images=images,
                padding=True,
                return_tensors="pt"
            )
            input_length = input_only.input_ids.shape[1]
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # 获取目标部分的logits和labels
                if input_length >= inputs.input_ids.shape[1]:
                    return float('inf'), float('inf')
                
                target_logits = logits[0, input_length-1:-1, :]
                target_ids = inputs.input_ids[0, input_length:]
                
                if target_logits.shape[0] == 0 or target_ids.shape[0] == 0:
                    return float('inf'), float('inf')
                
                # 计算交叉熵损失
                loss = F.cross_entropy(target_logits, target_ids, reduction='mean')
                
                # 计算困惑度
                perplexity = torch.exp(loss).item()
                nll = loss.item()
                
                return perplexity, nll
                
        except Exception as e:
            print(f"⚠️ Method 2 also failed: {e}")
            # 返回默认值
            return float('inf'), float('inf')


class ModelInference:
    """模型推理类"""
    
    def __init__(
        self,
        model_name_or_path: str,
        lora_weights_path: Optional[str] = None,
        processor_path: Optional[str] = None,
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "float16",
        load_base_model: bool = True
    ):
        """
        初始化推理器
        
        Args:
            model_name_or_path: 基础模型路径
            lora_weights_path: LoRA权重路径（可选）
            processor_path: Processor路径（如果为None，使用model_name_or_path）
            device: 设备
            load_in_8bit: 是否8bit量化
            load_in_4bit: 是否4bit量化
            torch_dtype: 数据类型
            load_base_model: 是否加载原始模型进行对比
        """
        self.device = device
        self.model_name = model_name_or_path
        self.lora_path = lora_weights_path
        self.load_base_model = load_base_model
        
        # 设置数据类型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)
        
        # 加载模型和处理器
        self._load_models_and_processor(
            model_name_or_path,
            lora_weights_path,
            processor_path,
            load_in_8bit,
            load_in_4bit
        )
        
        # 初始化困惑度计算器
        self.lora_perplexity_calc = PerplexityCalculator(
            self.lora_model if hasattr(self, 'lora_model') else self.base_model,
            self.processor,
            self.device
        )
        
        if hasattr(self, 'base_model'):
            self.base_perplexity_calc = PerplexityCalculator(
                self.base_model,
                self.processor,
                self.device
            )
    
    def _load_models_and_processor(
        self,
        model_name_or_path: str,
        lora_weights_path: Optional[str],
        processor_path: Optional[str],
        load_in_8bit: bool,
        load_in_4bit: bool
    ):
        """加载模型和处理器"""
        print(f"🔄 Loading base model from: {model_name_or_path}")
        
        # 量化配置
        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # 加载基础模型
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # 加载基础模型
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        # 如果需要对比，保存原始模型
        if self.load_base_model:
            self.base_model = base_model
            self.base_model.eval()
            print("✅ Base model loaded for comparison")
        
        # 加载LoRA模型
        if lora_weights_path:
            print(f"🔄 Loading LoRA weights from: {lora_weights_path}")
            
            # 为LoRA创建单独的模型实例
            if self.load_base_model:
                lora_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
            else:
                lora_model = base_model
            
            self.lora_model = PeftModel.from_pretrained(
                lora_model,
                lora_weights_path,
                torch_dtype=self.torch_dtype
            )
            self.lora_model.eval()
            print("✅ LoRA model loaded")
        else:
            # 只有基础模型
            self.lora_model = None
            if not self.load_base_model:
                self.base_model = base_model
        
        # 加载处理器
        processor_path = processor_path or model_name_or_path
        print(f"🔄 Loading processor from: {processor_path}")
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True
        )
        
        print("✅ Models and processor loaded successfully")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        model_type: str = "lora"
    ) -> str:
        """
        生成响应
        
        Args:
            messages: 对话消息列表
            images: 图像列表（可选）
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样参数
            repetition_penalty: 重复惩罚
            model_type: 模型类型 ("lora" 或 "base")
            
        Returns:
            生成的文本
        """
        # 选择模型
        if model_type == "base" and hasattr(self, 'base_model'):
            model = self.base_model
        else:
            model = self.lora_model if hasattr(self, 'lora_model') else self.base_model
        
        # 修复消息格式以包含图像信息
        if images is not None:
            formatted_messages = []
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    # 为第一个用户消息添加图像
                    if i == 0:
                        formatted_messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": msg["content"]}
                            ]
                        })
                    else:
                        formatted_messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                else:
                    formatted_messages.append(msg)
            
            messages = formatted_messages
        
        # 准备输入
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0.0
            )
        
        # 只获取生成的部分
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return generated_text
    
    def run_inference(
        self,
        dataset: SupervisedDataset,
        output_dir: str,
        max_samples: Optional[int] = None,
        max_new_tokens: int = 512,
        calculate_perplexity: bool = True,
        run_lora: bool = True,
        run_base: bool = True
    ) -> None:
        """
        运行推理并保存结果
        
        Args:
            dataset: 评估数据集
            output_dir: 输出目录
            max_samples: 最大评估样本数
            max_new_tokens: 最大生成token数
            calculate_perplexity: 是否计算困惑度
            run_lora: 是否运行LoRA模型
            run_base: 是否运行基础模型
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备评估
        total_samples = len(dataset)
        if max_samples:
            total_samples = min(total_samples, max_samples)
        
        print(f"📊 Running inference on {total_samples} samples...")
        
        # 收集结果
        inference_results = {
            "metadata": {
                "base_model": self.model_name,
                "lora_weights": self.lora_path,
                "inference_time": datetime.now().isoformat(),
                "total_samples": total_samples,
                "run_lora": run_lora,
                "run_base": run_base,
                "calculate_perplexity": calculate_perplexity
            },
            "results": []
        }
        
        # 进度条
        pbar = tqdm(range(total_samples), desc="Running inference")
        
        for idx in pbar:
            try:
                # 获取数据
                item = dataset.list_data_dict[idx]
                item_id = dataset.get_item_id(idx)
                
                # 提取对话和图像
                conversations = item.get("conversations", [])
                if not conversations:
                    continue
                
                # 构建消息
                messages = []
                reference = ""
                
                for i in range(0, len(conversations), 2):
                    if i < len(conversations) - 1:
                        user_msg = conversations[i]
                        assistant_msg = conversations[i + 1]
                        
                        # 清理用户消息中的图像token
                        user_content = user_msg["value"]
                        if user_content.startswith("<image>\n"):
                            user_content = user_content[8:]
                        
                        messages.append({
                            "role": "user",
                            "content": user_content
                        })
                        
                        if i == len(conversations) - 2:
                            reference = assistant_msg["value"]
                
                # 处理图像
                images = None
                if "image" in item:
                    image_path = item["image"]
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(dataset.data_args.image_folder, image_path)
                    
                    if os.path.exists(image_path):
                        images = [Image.open(image_path)]
                
                # 准备结果记录
                result_record = {
                    "item_id": item_id,
                    "messages": messages,
                    "reference": reference,
                    "has_image": images is not None
                }
                
                # 提取输入文本用于困惑度计算
                input_text = messages[0]["content"] if messages else ""
                
                # 运行LoRA模型推理
                if run_lora and (hasattr(self, 'lora_model') or not self.lora_path):
                    lora_prediction = self.generate_response(
                        messages=messages,
                        images=images,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,
                        model_type="lora"
                    )
                    result_record["lora_prediction"] = lora_prediction
                    
                    # 计算LoRA困惑度
                    if calculate_perplexity:
                        lora_ppl, lora_nll = self.lora_perplexity_calc.calculate_perplexity(
                            input_text, reference, images
                        )
                        result_record["lora_perplexity"] = lora_ppl
                        result_record["lora_nll"] = lora_nll
                
                # 运行基础模型推理
                if run_base and hasattr(self, 'base_model'):
                    base_prediction = self.generate_response(
                        messages=messages,
                        images=images,
                        max_new_tokens=max_new_tokens,
                        temperature=0.1,
                        model_type="base"
                    )
                    result_record["base_prediction"] = base_prediction
                    
                    # 计算基础模型困惑度
                    if calculate_perplexity:
                        base_ppl, base_nll = self.base_perplexity_calc.calculate_perplexity(
                            input_text, reference, images
                        )
                        result_record["base_perplexity"] = base_ppl
                        result_record["base_nll"] = base_nll
                
                # 添加到结果
                inference_results["results"].append(result_record)
                
                # 更新进度条
                info = {"item_id": item_id}
                if "lora_prediction" in result_record:
                    info["lora_len"] = len(result_record["lora_prediction"].split())
                if "base_prediction" in result_record:
                    info["base_len"] = len(result_record["base_prediction"].split())
                pbar.set_postfix(info)
                
            except Exception as e:
                print(f"\n⚠️ Error processing item {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存结果
        output_file = os.path.join(output_dir, "inference_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(inference_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved inference results to: {output_file}")
        
        # 打印统计信息
        self._print_inference_summary(inference_results)
    
    def _print_inference_summary(self, results: Dict[str, Any]):
        """打印推理摘要"""
        print("\n" + "="*50)
        print("📊 INFERENCE SUMMARY")
        print("="*50)
        
        total_results = len(results["results"])
        print(f"Total samples processed: {total_results}")
        
        # 统计LoRA结果
        lora_results = [r for r in results["results"] if "lora_prediction" in r]
        if lora_results:
            print(f"\nLoRA model:")
            print(f"  - Predictions generated: {len(lora_results)}")
            
            if any("lora_perplexity" in r for r in lora_results):
                valid_ppls = [r["lora_perplexity"] for r in lora_results 
                             if "lora_perplexity" in r and not math.isinf(r["lora_perplexity"])]
                if valid_ppls:
                    print(f"  - Average perplexity: {np.mean(valid_ppls):.4f}")
                    print(f"  - Median perplexity: {np.median(valid_ppls):.4f}")
        
        # 统计基础模型结果
        base_results = [r for r in results["results"] if "base_prediction" in r]
        if base_results:
            print(f"\nBase model:")
            print(f"  - Predictions generated: {len(base_results)}")
            
            if any("base_perplexity" in r for r in base_results):
                valid_ppls = [r["base_perplexity"] for r in base_results 
                             if "base_perplexity" in r and not math.isinf(r["base_perplexity"])]
                if valid_ppls:
                    print(f"  - Average perplexity: {np.mean(valid_ppls):.4f}")
                    print(f"  - Median perplexity: {np.median(valid_ppls):.4f}")
        
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Model inference and perplexity calculation")
    
    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, 
                        default=None
    parser.add_argument("--lora_weights", type=str, 
                        default=None,
                       help="Path to LoRA weights")
    parser.add_argument("--processor_path", type=str, default=None,
                       help="Path to processor (default: same as model)")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, 
                        default='data_path/test_data/slake_test.json',
                       help="Path to evaluation data JSON")
    parser.add_argument("--image_folder", type=str, default="",
                       help="Base folder for images")
    
    # 推理参数
    # parser.add_argument("--output_dir", type=str, 
    #                     help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    
    # 功能开关
    parser.add_argument("--calculate_perplexity", action="store_true", default=True,
                        help="Calculate perplexity for each sample")
    parser.add_argument("--no_perplexity", dest="calculate_perplexity", action="store_false",
                        help="Don't calculate perplexity")
    parser.add_argument("--run_lora", action="store_true", default=True,
                        help="Run LoRA model inference")
    parser.add_argument("--no_lora", dest="run_lora", action="store_false",
                        help="Don't run LoRA model")
    parser.add_argument("--run_base", action="store_true", default=True,
                        help="Run base model inference")
    parser.add_argument("--no_base", dest="run_base", action="store_false",
                        help="Don't run base model")
    
    # 模型加载参数
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Torch dtype for model")
    
    # 图像参数
    parser.add_argument("--image_min_pixels", type=int, default=336*336,
                        help="Minimum pixels for images")
    parser.add_argument("--image_max_pixels", type=int, default=1024*1024,
                        help="Maximum pixels for images")
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.run_lora and not args.run_base:
        parser.error("At least one of --run_lora or --run_base must be enabled")
    
    if args.run_base and not args.lora_weights:
        args.run_base = False
        print("⚠️ No LoRA weights provided, will only run base model")
    
    # 创建输出目录
    if args.lora_weights:
        args.output_dir = os.path.join(os.path.dirname(args.lora_weights)+'_result_ppl', args.lora_weights.split('/')[-1])
    else:
        args.output_dir = os.path.join(os.path.dirname(args.model_name_or_path), 'base_model_evaluation')
    
    # 初始化推理器
    print("🚀 Initializing model inference...")
    inferencer = ModelInference(
        model_name_or_path=args.model_name_or_path,
        lora_weights_path=args.lora_weights,
        processor_path=args.processor_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        load_base_model=args.run_base
    )
    
    # 准备数据参数
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
        video_min_pixels=args.image_min_pixels,
        video_max_pixels=args.image_max_pixels,
        fps=8
    )
    
    # 加载数据集
    print("📂 Loading dataset...")
    dataset = SupervisedDataset(
        data_path=args.data_path,
        processor=inferencer.processor,
        data_args=data_args,
        model_id=args.model_name_or_path,
        padding=False
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # 运行推理
    print("🏃 Starting inference...")
    start_time = time.time()
    
    inferencer.run_inference(
        dataset=dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        calculate_perplexity=args.calculate_perplexity,
        run_lora=args.run_lora,
        run_base=args.run_base
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Inference completed in {elapsed_time:.2f} seconds")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()