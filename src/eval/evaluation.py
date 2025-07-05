"""
简化的评估模块，用于Qwen2.5-VL训练验证
支持基于type_mapping的分类评估 (close-set vs open-set)
"""

import os
import re
import json
import torch
import numpy as np
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
from difflib import SequenceMatcher

# 尝试导入NLTK，如果失败则使用简化版本
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    FULL_EVAL = True
except ImportError:
    FULL_EVAL = False
    print("⚠️ NLTK/rouge_score not available, using simplified evaluation")

# 全局变量存储type_mapping
TYPE_MAPPING = {}
TYPE_MAPPING_LOADED = True


def load_type_mapping(mapping_file_path: str) -> bool:
    """加载type_mapping文件作为全局变量"""
    global TYPE_MAPPING, TYPE_MAPPING_LOADED
    
    if not os.path.exists(mapping_file_path):
        print(f"⚠️ Type mapping file not found: {mapping_file_path}")
        return False
    
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # 提取type_mapping部分
        if "type_mapping" in mapping_data:
            TYPE_MAPPING = mapping_data["type_mapping"]
        elif "id_type_mapping" in mapping_data:  # 兼容旧格式
            TYPE_MAPPING = mapping_data["id_type_mapping"]
        else:
            TYPE_MAPPING = mapping_data
        
        TYPE_MAPPING_LOADED = True
        
        # 统计类型分布
        type_stats = {}
        for item_type in TYPE_MAPPING.values():
            type_stats[item_type] = type_stats.get(item_type, 0) + 1
        
        print(f"✅ Type mapping loaded: {len(TYPE_MAPPING)} items")
        print(f"📊 Type distribution: {type_stats}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load type mapping: {e}")
        return False


def get_item_type(item_id: str) -> str:
    """获取item的类型"""
    if not TYPE_MAPPING_LOADED:
        return "unknown"
    return TYPE_MAPPING.get(item_id, "unknown")

class SimpleTextEvaluator:
    """简化的文本评估器，专注于核心指标"""
    
    def __init__(self, use_full_metrics: bool = None):
        self.use_full_metrics = FULL_EVAL if use_full_metrics is None else use_full_metrics
        
        if self.use_full_metrics:
            try:
                # 设置自定义NLTK路径
                custom_path = None
                if os.path.exists(custom_path):
                    nltk.data.path.insert(0, custom_path)
                
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                self.smoothing = SmoothingFunction().method1
                print("✅ Full evaluation metrics loaded")
            except Exception as e:
                print(f"⚠️ Failed to load full metrics: {e}, using simplified version")
                self.use_full_metrics = False
    
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        return text
    
    def simple_tokenize(self, text: str) -> List[str]:
        """简单分词（不依赖NLTK）"""
        text = self.preprocess_text(text).lower()
        # 简单的单词分割
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def calculate_exact_match(self, reference: str, candidate: str) -> float:
        """精确匹配"""
        ref = self.preprocess_text(reference).lower()
        cand = self.preprocess_text(candidate).lower()
        return 1.0 if ref == cand else 0.0
    
    def calculate_soft_match(self, reference: str, candidate: str, threshold: float = 0.6) -> float:
        """软匹配（基于字符串相似度）"""
        ref = self.preprocess_text(reference).lower()
        cand = self.preprocess_text(candidate).lower()
        
        if not cand.strip():
            return 0.0
        
        if ref == cand:
            return 1.0
        
        similarity = SequenceMatcher(None, ref, cand).ratio()
        return similarity
    
    def calculate_word_overlap(self, reference: str, candidate: str) -> float:
        """词汇重叠度"""
        ref_words = set(self.simple_tokenize(reference))
        cand_words = set(self.simple_tokenize(candidate))
        
        if not ref_words or not cand_words:
            return 0.0
        
        overlap = len(ref_words.intersection(cand_words))
        union = len(ref_words.union(cand_words))
        return overlap / union if union > 0 else 0.0
    
    def calculate_bleu_4(self, reference: str, candidate: str) -> float:
        """BLEU-4分数（如果可用）"""
        if not self.use_full_metrics:
            # 使用简化版本
            return self.calculate_word_overlap(reference, candidate)
        
        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            
            if len(cand_tokens) == 0:
                return 0.0
            
            weights = (0.25, 0.25, 0.25, 0.25)
            score = sentence_bleu([ref_tokens], cand_tokens, weights=weights, smoothing_function=self.smoothing)
            return score
        except:
            return self.calculate_word_overlap(reference, candidate)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """ROUGE-L分数（如果可用）"""
        if not self.use_full_metrics:
            return self.calculate_word_overlap(reference, candidate)
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return scores['rougeL'].fmeasure
        except:
            return self.calculate_word_overlap(reference, candidate)
    
    def evaluate_batch_with_types(self, predictions: List[str], references: List[str], 
                                  item_ids: List[str]) -> Dict[str, Any]:
        """
        批量评估并按类型分组返回指标
        
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            item_ids: item ID列表
            
        Returns:
            包含总体指标和分类指标的字典
        """
        if len(predictions) != len(references) or len(predictions) != len(item_ids):
            raise ValueError("Predictions, references, and item_ids must have the same length")
        
        # 按类型分组数据
        type_groups = defaultdict(list)
        for pred, ref, item_id in zip(predictions, references, item_ids):
            item_type = get_item_type(item_id)
            type_groups[item_type].append((pred, ref, item_id))
        
        # 计算总体指标
        overall_metrics = self._calculate_metrics_for_group(predictions, references, "overall")
        
        # 计算各类型指标
        type_metrics = {}
        for item_type, group_data in type_groups.items():
            if group_data:
                group_preds = [item[0] for item in group_data]
                group_refs = [item[1] for item in group_data]
                type_metrics[item_type] = self._calculate_metrics_for_group(
                    group_preds, group_refs, item_type
                )
        
        # 组装结果
        results = {
            "overall": overall_metrics,
            "by_type": type_metrics,
            "sample_counts": {
                "total": len(predictions),
                **{f"{item_type}_count": len(group_data) for item_type, group_data in type_groups.items()}
            }
        }
        
        return results
    
    def _calculate_metrics_for_group(self, predictions: List[str], references: List[str], 
                                   group_name: str) -> Dict[str, float]:
        """为一组数据计算指标"""
        if not predictions:
            return {}
        
        metrics = {
            'exact_match': [],
            'soft_match': [],
            'word_overlap': [],
            'bleu_4': [],
            'rouge_l': []
        }
        
        for pred, ref in zip(predictions, references):
            metrics['exact_match'].append(self.calculate_exact_match(ref, pred))
            metrics['soft_match'].append(self.calculate_soft_match(ref, pred))
            metrics['word_overlap'].append(self.calculate_word_overlap(ref, pred))
            metrics['bleu_4'].append(self.calculate_bleu_4(ref, pred))
            metrics['rouge_l'].append(self.calculate_rouge_l(ref, pred))
        
        # 返回平均值，添加前缀
        avg_metrics = {}
        for key, values in metrics.items():
            avg_metrics[f'eval_{key}'] = np.mean(values) if values else 0.0
        
        avg_metrics['eval_samples'] = len(predictions)
        return avg_metrics
    
    def evaluate_batch(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """向后兼容的批量评估方法"""
        return self._calculate_metrics_for_group(predictions, references, "overall")

    
    

class EvaluationCallback:
    """评估回调，用于记录验证过程中的样例和指标"""
    
    def __init__(self, output_dir: str, save_examples: bool = True):
        self.output_dir = output_dir
        self.save_examples = save_examples
        self.eval_history = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def log_evaluation_results(self, step: int, results: Dict[str, Any], 
                             predictions: List[str] = None, references: List[str] = None,
                             item_ids: List[str] = None):
        """记录评估结果"""
        
        # 保存评估指标历史
        eval_record = {
            "step": step,
            "timestamp": time.time(),
            "metrics": results
        }
        self.eval_history.append(eval_record)
        
        # 保存评估历史到文件
        history_file = os.path.join(self.output_dir, "evaluation_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.eval_history, f, indent=2, ensure_ascii=False)
        
        # 保存当前评估结果
        current_result_file = os.path.join(self.output_dir, f"evaluation_step_{step}.json")
        with open(current_result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印评估结果
        self._print_evaluation_results(step, results)
        
        # 保存样例（如果提供）
        if self.save_examples and predictions and references and item_ids:
            self._save_examples(step, predictions, references, item_ids)
    
    def _print_evaluation_results(self, step: int, results: Dict[str, Any]):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION RESULTS - Step {step}")
        print(f"{'='*60}")
        
        # 打印总体指标
        overall = results.get("overall", {})
        if overall:
            print("🎯 Overall Metrics:")
            for metric, value in overall.items():
                if metric.startswith('eval_'):
                    print(f"  {metric}: {value:.4f}")
        
        # 打印分类指标
        by_type = results.get("by_type", {})
        if by_type:
            print("\n📋 Metrics by Type:")
            for item_type, metrics in by_type.items():
                if metrics:
                    print(f"\n  📂 {item_type.upper()}:")
                    for metric, value in metrics.items():
                        if metric.startswith('eval_'):
                            print(f"    {metric}: {value:.4f}")
        
        # 打印样本统计
        sample_counts = results.get("sample_counts", {})
        if sample_counts:
            print(f"\n📊 Sample Counts:")
            for count_type, count in sample_counts.items():
                print(f"  {count_type}: {count}")
        
        print(f"{'='*60}\n")
    
    def _save_examples(self, step: int, predictions: List[str], references: List[str], item_ids: List[str]):
        """保存评估样例"""
        examples = []
        max_examples = min(10, len(predictions))  # 最多保存10个样例
        
        for i in range(max_examples):
            item_type = get_item_type(item_ids[i])
            examples.append({
                'step': step,
                'item_id': item_ids[i],
                'item_type': item_type,
                'prediction': predictions[i],
                'reference': references[i]
            })
        
        # 保存样例到文件
        examples_file = os.path.join(self.output_dir, f"examples_step_{step}.json")
        with open(examples_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(examples)} evaluation examples to {examples_file}")



def create_compute_metrics_with_types(processor, output_dir: str = None, type_mapping_file: str = None, eval_dataset=None):   
    """
    创建支持类型分组的compute_metrics函数
    
    Args:
        processor: Qwen处理器
        output_dir: 输出目录（用于保存结果和样例）
        type_mapping_file: 类型映射文件路径
        eval_dataset: 评估数据集（用于获取item_ids）
    
    Returns:
        compute_metrics函数
    """
    
    # 加载type_mapping
    if type_mapping_file:
        load_type_mapping(type_mapping_file)
    
    evaluator = SimpleTextEvaluator()
    callback = EvaluationCallback(output_dir) if output_dir else None
    
    # 添加步数计数器和累积变量
    accumulated_predictions = []
    accumulated_labels = []
    accumulated_item_ids = []
    step_counter = [0]
    
    def compute_metrics(eval_preds, compute_result=None):
        """
        计算评估指标
        
        Args:
            eval_preds: EvalPrediction对象
            compute_result: 是否计算并返回最终结果（用于batch_eval_metrics）
        """
        # 从EvalPrediction对象中提取数据
        predictions = eval_preds.predictions
        labels = eval_preds.label_ids
        inputs = eval_preds.inputs if hasattr(eval_preds, 'inputs') else None
        
        if isinstance(predictions, tuple):
            # Qwen2-VL 返回 tuple，第一个元素是 language logits
            predictions = predictions[0]
            print(f"[Qwen2-VL] Processing tuple output, using first element with shape: {predictions.shape}")
        
        
        # 处理 predictions 和 labels 的维度
        # 在某些情况下，predictions 可能是 logits，需要转换为 token ids
        if len(predictions.shape) == 3:  # [batch_size, sequence_length, vocab_size]
            # 取 argmax 获得 token ids
            predictions = predictions.argmax(axis=-1)
        
        # 解码当前批次
        batch_decoded_preds = []
        batch_decoded_labels = []
        batch_item_ids = []
        
        for i, (pred_ids, label_ids) in enumerate(zip(predictions, labels)):
            # 过滤掉-100 (padding)
            valid_indices = label_ids != -100
            if not valid_indices.any():
                # 如果全是-100，跳过这个样本
                continue
            
            # 获取第一个和最后一个有效位置
            first_valid_idx = valid_indices.nonzero()[0].item() if valid_indices.any() else 0
            last_valid_idx = valid_indices.nonzero()[-1].item() + 1 if valid_indices.any() else len(label_ids)
            
            # 只保留需要计算loss的部分（对应label_ids中非-100的部分）
            pred_ids_valid = pred_ids[first_valid_idx:last_valid_idx]
            label_ids_valid = label_ids[first_valid_idx:last_valid_idx]
            
            # 再次过滤可能残留的-100（虽然理论上不应该有）
            mask = label_ids_valid != -100
            pred_ids_valid = pred_ids_valid[mask]
            label_ids_valid = label_ids_valid[mask]
            
            try:
                pred_text = processor.tokenizer.decode(pred_ids_valid, skip_special_tokens=True)
                label_text = processor.tokenizer.decode(label_ids_valid, skip_special_tokens=True)
                
                batch_decoded_preds.append(pred_text)
                batch_decoded_labels.append(label_text)
                
                # 获取item_id
                if eval_dataset and hasattr(eval_dataset, 'get_item_id'):
                    # 计算全局索引
                    global_idx = len(accumulated_predictions) + i
                    item_id = eval_dataset.get_item_id(global_idx)
                else:
                    item_id = f"eval_item_{len(accumulated_predictions) + i}"
                batch_item_ids.append(item_id)
                
            except Exception as e:
                print(f"⚠️ Decoding error: {e}")
                batch_decoded_preds.append("")
                batch_decoded_labels.append("")
                batch_item_ids.append(f"eval_item_{len(accumulated_predictions) + i}")
        
        # 累积结果
        accumulated_predictions.extend(batch_decoded_preds)
        accumulated_labels.extend(batch_decoded_labels)
        accumulated_item_ids.extend(batch_item_ids)
        
        # 如果不需要计算最终结果，返回空字典
        # if compute_result is False:
        #     return {}
        
        # 计算最终结果
        if compute_result is True or (compute_result is None and len(accumulated_predictions) > 0):
            # 使用所有累积的数据计算指标
            if TYPE_MAPPING_LOADED:
                results = evaluator.evaluate_batch_with_types(
                    accumulated_predictions, 
                    accumulated_labels, 
                    accumulated_item_ids
                )
            else:
                overall_metrics = evaluator.evaluate_batch(
                    accumulated_predictions, 
                    accumulated_labels
                )
                results = {"overall": overall_metrics}
            
            # 记录评估结果
            if callback:
                step_counter[0] += 1
                callback.log_evaluation_results(
                    step_counter[0], 
                    results, 
                    accumulated_predictions[:10],  # 只保存前10个样例
                    accumulated_labels[:10], 
                    accumulated_item_ids[:10]
                )
            
            # 打印样例
            print("\n📝 Evaluation Examples:")
            for i in range(min(3, len(accumulated_predictions))):
                item_type = get_item_type(accumulated_item_ids[i]) if TYPE_MAPPING_LOADED else "unknown"
                print(f"Example {i+1} [{item_type}]:")
                print(f"  Prediction: {accumulated_predictions[i][:100]}...")
                print(f"  Reference:  {accumulated_labels[i][:100]}...")
                print()
            
            # 清空累积数据（准备下一轮评估）
            accumulated_predictions.clear()
            accumulated_labels.clear()
            accumulated_item_ids.clear()
            
            # 返回扁平化的指标
            flattened_metrics = {}
            
            # 添加总体指标
            overall = results.get("overall", {})
            for key, value in overall.items():
                flattened_metrics[key] = value
            
            # 添加分类指标
            by_type = results.get("by_type", {})
            for item_type, metrics in by_type.items():
                for key, value in metrics.items():
                    flattened_metrics[f"{item_type}_{key}"] = value
            
            return flattened_metrics
        
        # 默认返回空字典
        return {}
    
    return compute_metrics


# 用于在 Trainer 中设置 compute_metrics 的辅助函数
def setup_trainer_compute_metrics(trainer, processor, output_dir=None, type_mapping_file=None):
    """
    为 Trainer 设置自定义的 compute_metrics 函数
    
    Args:
        trainer: Hugging Face Trainer 实例
        processor: Qwen处理器
        output_dir: 输出目录
        type_mapping_file: 类型映射文件路径
    
    Example:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )
        
        # 设置自定义的 compute_metrics
        setup_trainer_compute_metrics(
            trainer, 
            processor, 
            output_dir="./eval_results",
            type_mapping_file="type_mapping.json"
        )
    """
    eval_dataset = trainer.eval_dataset
    compute_metrics_fn = create_compute_metrics_with_types(
        processor=processor,
        output_dir=output_dir,
        type_mapping_file=type_mapping_file,
        eval_dataset=eval_dataset
    )
    
    # 设置 trainer 的 compute_metrics
    trainer.compute_metrics = compute_metrics_fn
    
    return trainer