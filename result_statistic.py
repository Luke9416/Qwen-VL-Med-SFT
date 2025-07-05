"""
完整版评估指标计算脚本
用于读取推理结果，计算各种文本匹配指标，并进行模型对比
支持按original_answer_type和question_type分类统计
"""

import argparse
import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import math
import csv
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas not installed. Pivot tables will not be generated.")

# 导入评估器
from src.eval.evaluation_0702 import (
    SimpleTextEvaluator,
    # load_type_mapping,
    # get_item_type,
    # TYPE_MAPPING_LOADED
)


TYPE_MAPPING_LOADED = True
TYPE_MAPPING = {}
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
        TYPE_MAPPING = {item['id']: item['original_answer_type'] for item in mapping_data["entries"]}
        
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
    # if not TYPE_MAPPING_LOADED:
    #     return "unknown"
    return TYPE_MAPPING.get(item_id, "unknown")

class ExtendedTextEvaluator(SimpleTextEvaluator):
    """扩展的文本评估器，添加了evaluate_pair方法"""
    
    def evaluate_pair(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        评估单个预测-参考对
        
        Args:
            prediction: 预测文本
            reference: 参考文本
            
        Returns:
            包含各项指标的字典
        """
        # 使用父类的方法计算各项指标
        metrics = {
            "exact_match": self.calculate_exact_match(reference, prediction),
            "soft_match": self.calculate_soft_match(reference, prediction),
            "word_overlap": self.calculate_word_overlap(reference, prediction),
            "bleu_4": self.calculate_bleu_4(reference, prediction),
            "rouge_l": self.calculate_rouge_l(reference, prediction)
        }
        
        # 计算额外的指标
        pred_clean = self.preprocess_text(prediction)
        ref_clean = self.preprocess_text(reference)
        
        # 字符级别的指标
        if len(ref_clean) > 0:
            char_precision = sum(c in ref_clean for c in pred_clean) / max(len(pred_clean), 1)
            char_recall = sum(c in pred_clean for c in ref_clean) / len(ref_clean)
            char_f1 = 2 * char_precision * char_recall / max(char_precision + char_recall, 1e-8)
            
            metrics["char_precision"] = char_precision
            metrics["char_recall"] = char_recall
            metrics["char_f1"] = char_f1
        else:
            metrics["char_precision"] = 0.0
            metrics["char_recall"] = 0.0
            metrics["char_f1"] = 0.0
        
        # 词级别的指标
        pred_words = self.simple_tokenize(prediction)
        ref_words = self.simple_tokenize(reference)
        
        if len(ref_words) > 0:
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            
            if len(pred_set) > 0:
                word_precision = len(pred_set & ref_set) / len(pred_set)
            else:
                word_precision = 0.0
                
            word_recall = len(pred_set & ref_set) / len(ref_set)
            word_f1 = 2 * word_precision * word_recall / max(word_precision + word_recall, 1e-8)
            
            metrics["word_precision"] = word_precision
            metrics["word_recall"] = word_recall
            metrics["word_f1"] = word_f1
        else:
            metrics["word_precision"] = 0.0
            metrics["word_recall"] = 0.0
            metrics["word_f1"] = 0.0
        
        return metrics

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





def load_enhanced_type_mapping(type_mapping_file: str) -> Dict[str, Dict[str, Any]]:
    """
    加载增强的类型映射文件
    
    Args:
        type_mapping_file: 类型映射文件路径
        
    Returns:
        映射字典 {item_id: {question_type, answer_category, original_answer_type, ...}}
    """
    with open(type_mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    type_mapping = {}
    for entry in data.get("entries", []):
        item_id = entry["id"]
        type_mapping[item_id] = {
            "question_type": entry.get("question_type", "unknown"),
            "answer_category": entry.get("answer_category", "unknown"),
            "original_answer_type": entry.get("original_answer_type", "unknown"),
            "domain": entry.get("domain", {})
        }
    
    return type_mapping


def generate_detailed_metrics_csv(
    evaluation_results: Dict[str, Any],
    type_mapping: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """
    生成按original_answer_type和question_type分类的详细指标CSV
    
    Args:
        evaluation_results: 评估结果字典
        type_mapping: 类型映射字典
        output_dir: 输出目录
    """
    # 准备数据结构存储分类指标
    metrics_by_category = {
        "lora": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "base": defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    }
    
    # 从详细结果中提取指标
    for model_type in ["lora", "base"]:
        if model_type not in evaluation_results.get("detailed_metrics", {}):
            continue
            
        detailed_results = evaluation_results["detailed_metrics"][model_type]
        
        for result in detailed_results:
            item_id = result["item_id"]
            
            # 获取分类信息
            if item_id in type_mapping:
                mapping_info = type_mapping[item_id]
                original_answer_type = mapping_info["original_answer_type"]
                question_type = mapping_info["question_type"]
                
                # 收集各项指标
                metrics = result.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics_by_category[model_type][original_answer_type][question_type][metric_name].append(metric_value)
    
    # 生成CSV文件
    for model_type in ["lora", "base"]:
        if not metrics_by_category[model_type]:
            continue
            
        csv_file = os.path.join(output_dir, f"{model_type}_metrics_by_category.csv")
        
        # 准备CSV数据
        csv_data = []
        headers = ["original_answer_type", "question_type", "sample_count"]
        metric_names = set()
        
        # 计算平均值并准备数据
        for answer_type, question_types in metrics_by_category[model_type].items():
            for q_type, metrics in question_types.items():
                row = {
                    "original_answer_type": answer_type,
                    "question_type": q_type,
                    "sample_count": 0
                }
                
                # 计算每个指标的平均值
                for metric_name, values in metrics.items():
                    if values:
                        avg_value = sum(values) / len(values)
                        row[f"avg_{metric_name}"] = round(avg_value, 4)
                        metric_names.add(f"avg_{metric_name}")
                        row["sample_count"] = len(values)
                
                if row["sample_count"] > 0:
                    csv_data.append(row)
        
        # 更新headers
        headers.extend(sorted(metric_names))
        
        # 写入CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            
            # 按original_answer_type和question_type排序
            csv_data.sort(key=lambda x: (x["original_answer_type"], x["question_type"]))
            writer.writerows(csv_data)
        
        print(f"💾 Saved {model_type} metrics by category to: {csv_file}")
        
        # 使用pandas生成更美观的汇总表
        if PANDAS_AVAILABLE:
            try:
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    pivot_file = os.path.join(output_dir, f"{model_type}_metrics_pivot.csv")
                    
                    # 为每个主要指标创建透视表
                    main_metrics = ["avg_exact_match", "avg_soft_match", "avg_rouge_l", "avg_bleu_4"]
                    existing_metrics = [m for m in main_metrics if m in df.columns]
                    
                    if existing_metrics:
                        # 创建一个多级索引的DataFrame
                        pivot_dfs = []
                        
                        for metric in existing_metrics:
                            pivot = df.pivot_table(
                                values=metric,
                                index="original_answer_type",
                                columns="question_type",
                                aggfunc='first'
                            ).round(4)
                            
                            # 添加行和列的平均值
                            pivot['Average'] = pivot.mean(axis=1).round(4)
                            pivot.loc['Average'] = pivot.mean(axis=0).round(4)
                            
                            # 添加指标名称作为标识
                            pivot.index = [f"{metric}_{idx}" for idx in pivot.index]
                            pivot_dfs.append(pivot)
                        
                        # 合并所有透视表
                        combined_pivot = pd.concat(pivot_dfs)
                        combined_pivot.to_csv(pivot_file)
                        print(f"💾 Saved {model_type} pivot table to: {pivot_file}")
            except Exception as e:
                print(f"⚠️ Could not create pivot table: {e}")
    
    # 生成对比CSV（如果两个模型都有结果）
    if "lora" in metrics_by_category and "base" in metrics_by_category:
        comparison_file = os.path.join(output_dir, "model_comparison_by_category.csv")
        comparison_data = []
        
        # 获取所有的answer_type和question_type组合
        all_combinations = set()
        for model_metrics in metrics_by_category.values():
            for answer_type, question_types in model_metrics.items():
                for q_type in question_types.keys():
                    all_combinations.add((answer_type, q_type))
        
        # 对每个组合计算对比
        for answer_type, q_type in sorted(all_combinations):
            row = {
                "original_answer_type": answer_type,
                "question_type": q_type
            }
            
            # 获取主要指标的对比
            main_metrics = ["exact_match", "soft_match", "rouge_l", "bleu_4"]
            
            for metric in main_metrics:
                lora_values = metrics_by_category["lora"][answer_type][q_type].get(metric, [])
                base_values = metrics_by_category["base"][answer_type][q_type].get(metric, [])
                
                if lora_values and base_values:
                    lora_avg = sum(lora_values) / len(lora_values)
                    base_avg = sum(base_values) / len(base_values)
                    improvement = ((lora_avg - base_avg) / base_avg * 100) if base_avg != 0 else 0
                    
                    row[f"lora_{metric}"] = round(lora_avg, 4)
                    row[f"base_{metric}"] = round(base_avg, 4)
                    row[f"{metric}_improvement_%"] = round(improvement, 2)
            
            row["lora_samples"] = len(metrics_by_category["lora"][answer_type][q_type].get("exact_match", []))
            row["base_samples"] = len(metrics_by_category["base"][answer_type][q_type].get("exact_match", []))
            
            if row["lora_samples"] > 0 or row["base_samples"] > 0:
                comparison_data.append(row)
        
        # 写入对比CSV
        if comparison_data:
            comparison_headers = ["original_answer_type", "question_type", "lora_samples", "base_samples"]
            for metric in ["exact_match", "soft_match", "rouge_l", "bleu_4"]:
                comparison_headers.extend([f"lora_{metric}", f"base_{metric}", f"{metric}_improvement_%"])
            
            with open(comparison_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=comparison_headers)
                writer.writeheader()
                writer.writerows(comparison_data)
            
            print(f"💾 Saved model comparison by category to: {comparison_file}")


def print_category_summary(
    evaluation_results: Dict[str, Any],
    type_mapping: Dict[str, Dict[str, Any]]
) -> None:
    """
    打印按类别分组的指标摘要
    
    Args:
        evaluation_results: 评估结果字典
        type_mapping: 类型映射字典
    """
    print("\n" + "="*80)
    print("📊 METRICS BY CATEGORY")
    print("="*80)
    
    # 统计各类别的指标
    for model_type in ["lora", "base"]:
        if model_type not in evaluation_results.get("detailed_metrics", {}):
            continue
            
        print(f"\n🤖 {model_type.upper()} MODEL - Metrics by Category:")
        print("-" * 70)
        
        # 收集按类别分组的指标
        category_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        detailed_results = evaluation_results["detailed_metrics"][model_type]
        
        for result in detailed_results:
            item_id = result["item_id"]
            
            if item_id in type_mapping:
                mapping_info = type_mapping[item_id]
                original_answer_type = mapping_info["original_answer_type"]
                question_type = mapping_info["question_type"]
                
                metrics = result.get("metrics", {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        category_metrics[original_answer_type][question_type][metric_name].append(metric_value)
        
        # 打印结果
        for answer_type in sorted(category_metrics.keys()):
            print(f"\n📁 Original Answer Type: {answer_type}")
            
            for q_type in sorted(category_metrics[answer_type].keys()):
                metrics = category_metrics[answer_type][q_type]
                sample_count = len(next(iter(metrics.values()), []))
                
                print(f"  📋 Question Type: {q_type} (n={sample_count})")
                
                # 打印主要指标
                main_metrics = ["exact_match", "soft_match", "rouge_l", "bleu_4"]
                for metric_name in main_metrics:
                    if metric_name in metrics and metrics[metric_name]:
                        avg_value = sum(metrics[metric_name]) / len(metrics[metric_name])
                        print(f"    - {metric_name}: {avg_value:.4f}")
    
    print("\n" + "="*80)


class MetricsEvaluator:
    """指标评估器"""
    
    def __init__(self, type_mapping_file: Optional[str] = None):
        """
        初始化评估器
        
        Args:
            type_mapping_file: 类型映射文件路径
        """
        self.text_evaluator = ExtendedTextEvaluator()
        self.type_mapping = {}  # 存储增强的类型映射
        
        # 加载类型映射（如果提供）
        # if type_mapping_file and os.path.exists(type_mapping_file):
        #     load_enhanced_type_mapping(type_mapping_file)
            # 加载增强的类型映射
        self.type_mapping = load_enhanced_type_mapping(type_mapping_file)
        print(f"✅ Loaded enhanced type mapping from: {type_mapping_file}")
    
    def evaluate_from_results(
        self,
        results_file: str,
        output_dir: str,
        evaluate_lora: bool = True,
        evaluate_base: bool = True,
        compare_models: bool = True
    ):
        """
        从推理结果文件计算评估指标
        
        Args:
            results_file: 推理结果JSON文件路径
            output_dir: 输出目录
            evaluate_lora: 是否评估LoRA模型
            evaluate_base: 是否评估基础模型
            compare_models: 是否进行模型对比
            
        Returns:
            评估结果字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载推理结果
        print(f"📂 Loading inference results from: {results_file}")
        with open(results_file, 'r', encoding='utf-8') as f:
            inference_data = json.load(f)
        
        results = inference_data["results"]
        metadata = inference_data["metadata"]
        
        print(f"✅ Loaded {len(results)} samples")
        
        # 准备评估结果
        evaluation_results = {
            "metadata": {
                **metadata,
                "evaluation_time": datetime.now().isoformat(),
                "evaluate_lora": evaluate_lora,
                "evaluate_base": evaluate_base,
                "compare_models": compare_models
            },
            "metrics": {},
            "detailed_metrics": {}
        }
        
        # 评估LoRA模型
        if evaluate_lora:
            lora_metrics = self._evaluate_model_results(
                results, 
                model_type="lora",
                output_dir=output_dir
            )
            if lora_metrics:
                evaluation_results["metrics"]["lora"] = lora_metrics["metrics"]
                evaluation_results["detailed_metrics"]["lora"] = lora_metrics["detailed"]
        
        # 评估基础模型
        if evaluate_base:
            base_metrics = self._evaluate_model_results(
                results,
                model_type="base",
                output_dir=output_dir
            )
            if base_metrics:
                evaluation_results["metrics"]["base"] = base_metrics["metrics"]
                evaluation_results["detailed_metrics"]["base"] = base_metrics["detailed"]
        
        # 模型对比
        if compare_models and "lora" in evaluation_results["metrics"] and "base" in evaluation_results["metrics"]:
            comparison = self._compare_models(
                evaluation_results["metrics"]["lora"],
                evaluation_results["metrics"]["base"],
                results
            )
            evaluation_results["comparison"] = comparison
        
        # 生成按类别分组的CSV文件
        if self.type_mapping:
            generate_detailed_metrics_csv(evaluation_results, self.type_mapping, output_dir)
            print_category_summary(evaluation_results, self.type_mapping)
        
        # 保存评估结果
        self._save_evaluation_results(evaluation_results, output_dir)
        
        return evaluation_results
    
    def _evaluate_model_results(
        self,
        results: List[Dict[str, Any]],
        model_type: str,
        output_dir: str
    ):
        """
        评估单个模型的结果
        
        Args:
            results: 推理结果列表
            model_type: 模型类型 ("lora" 或 "base")
            output_dir: 输出目录
            
        Returns:
            评估指标字典
        """
        print(f"\n📊 Evaluating {model_type} model...")
        
        # 收集预测和参考
        predictions = []
        references = []
        item_ids = []
        perplexities = []
        nlls = []
        
        prediction_key = f"{model_type}_prediction"
        perplexity_key = f"{model_type}_perplexity"
        nll_key = f"{model_type}_nll"
        
        for result in results:
            if prediction_key in result:
                predictions.append(result[prediction_key])
                references.append(result["reference"])
                item_ids.append(result["item_id"])
                
                # 收集困惑度
                if perplexity_key in result and not math.isinf(result[perplexity_key]):
                    perplexities.append(result[perplexity_key])
                    if nll_key in result:
                        nlls.append(result[nll_key])
        
        if not predictions:
            print(f"⚠️ No {model_type} predictions found")
            return None
        
        print(f"📊 Evaluating {len(predictions)} {model_type} predictions")
        
        # 计算文本指标
        if TYPE_MAPPING_LOADED:
            metrics = self.text_evaluator.evaluate_batch_with_types(
                predictions,
                references,
                item_ids
            )
        else:
            overall_metrics = self.text_evaluator.evaluate_batch(
                predictions,
                references
            )
            metrics = {"overall": overall_metrics}
        
        # 添加困惑度统计
        if perplexities:
            perplexity_stats = {
                "mean_perplexity": np.mean(perplexities),
                "std_perplexity": np.std(perplexities),
                "median_perplexity": np.median(perplexities),
                "min_perplexity": np.min(perplexities),
                "max_perplexity": np.max(perplexities),
                "num_valid_samples": len(perplexities),
                "valid_ratio": len(perplexities) / len(predictions)
            }
            
            if nlls:
                perplexity_stats.update({
                    "mean_nll": np.mean(nlls),
                    "std_nll": np.std(nlls),
                    "median_nll": np.median(nlls)
                })
            
            metrics["perplexity"] = perplexity_stats
        
        # 准备详细结果
        detailed_results = []
        for i, (pred, ref, item_id) in enumerate(zip(predictions, references, item_ids)):
            detail = {
                "item_id": item_id,
                "prediction": pred,
                "reference": ref,
                "metrics": self.text_evaluator.evaluate_pair(pred, ref)
            }
            
            # 添加困惑度信息
            if i < len(results) and perplexity_key in results[i]:
                detail["perplexity"] = results[i][perplexity_key]
                if nll_key in results[i]:
                    detail["nll"] = results[i][nll_key]
            
            detailed_results.append(detail)
        
        return {
            "metrics": metrics,
            "detailed": detailed_results
        }
    
    def _compare_models(
        self,
        lora_metrics: Dict[str, Any],
        base_metrics: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        比较两个模型的性能
        
        Args:
            lora_metrics: LoRA模型指标
            base_metrics: 基础模型指标
            results: 原始推理结果
            
        Returns:
            对比结果字典
        """
        print("\n📊 Comparing models...")
        
        comparison = {
            "overall": {},
            "by_type": {},
            "perplexity": {},
            "sample_level": []
        }
        
        # 比较总体指标
        if "overall" in lora_metrics and "overall" in base_metrics:
            for metric_name in lora_metrics["overall"]:
                if metric_name.startswith("eval_") and metric_name in base_metrics["overall"]:
                    lora_val = lora_metrics["overall"][metric_name]
                    base_val = base_metrics["overall"][metric_name]
                    
                    if isinstance(lora_val, (int, float)) and isinstance(base_val, (int, float)):
                        improvement = ((lora_val - base_val) / base_val * 100) if base_val != 0 else 0
                        
                        comparison["overall"][metric_name] = {
                            "base": base_val,
                            "lora": lora_val,
                            "improvement_%": improvement,
                            "better_model": "lora" if lora_val > base_val else "base"
                        }
        
        # 比较类型级别的指标
        if "by_type" in lora_metrics and "by_type" in base_metrics:
            for type_name in lora_metrics["by_type"]:
                if type_name in base_metrics["by_type"]:
                    comparison["by_type"][type_name] = {}
                    
                    for metric_name in lora_metrics["by_type"][type_name]:
                        if metric_name.startswith("eval_") and metric_name in base_metrics["by_type"][type_name]:
                            lora_val = lora_metrics["by_type"][type_name][metric_name]
                            base_val = base_metrics["by_type"][type_name][metric_name]
                            
                            if isinstance(lora_val, (int, float)) and isinstance(base_val, (int, float)):
                                improvement = ((lora_val - base_val) / base_val * 100) if base_val != 0 else 0
                                
                                comparison["by_type"][type_name][metric_name] = {
                                    "base": base_val,
                                    "lora": lora_val,
                                    "improvement_%": improvement,
                                    "better_model": "lora" if lora_val > base_val else "base"
                                }
        
        # 比较困惑度
        if "perplexity" in lora_metrics and "perplexity" in base_metrics:
            for ppl_metric in ["mean_perplexity", "median_perplexity", "mean_nll"]:
                if ppl_metric in lora_metrics["perplexity"] and ppl_metric in base_metrics["perplexity"]:
                    lora_val = lora_metrics["perplexity"][ppl_metric]
                    base_val = base_metrics["perplexity"][ppl_metric]
                    
                    # 困惑度越低越好
                    improvement = ((base_val - lora_val) / base_val * 100) if base_val != 0 else 0
                    
                    comparison["perplexity"][ppl_metric] = {
                        "base": base_val,
                        "lora": lora_val,
                        "improvement_%": improvement,
                        "better_model": "lora" if lora_val < base_val else "base"
                    }
        
        # 样本级别的比较
        for result in results:
            if "lora_prediction" in result and "base_prediction" in result:
                lora_pred = result["lora_prediction"]
                base_pred = result["base_prediction"]
                reference = result["reference"]
                
                # 计算每个样本的指标
                lora_sample_metrics = self.text_evaluator.evaluate_pair(lora_pred, reference)
                base_sample_metrics = self.text_evaluator.evaluate_pair(base_pred, reference)
                
                sample_comparison = {
                    "item_id": result["item_id"],
                    "lora_metrics": lora_sample_metrics,
                    "base_metrics": base_sample_metrics,
                    "better_model": self._determine_better_model(lora_sample_metrics, base_sample_metrics)
                }
                
                # 添加困惑度比较
                if "lora_perplexity" in result and "base_perplexity" in result:
                    sample_comparison["lora_perplexity"] = result["lora_perplexity"]
                    sample_comparison["base_perplexity"] = result["base_perplexity"]
                    
                    if not math.isinf(result["lora_perplexity"]) and not math.isinf(result["base_perplexity"]):
                        sample_comparison["perplexity_improvement_%"] = (
                            (result["base_perplexity"] - result["lora_perplexity"]) / 
                            result["base_perplexity"] * 100
                        )
                
                comparison["sample_level"].append(sample_comparison)
        
        # 统计获胜情况
        if comparison["sample_level"]:
            wins = {"lora": 0, "base": 0, "tie": 0}
            for sample in comparison["sample_level"]:
                wins[sample["better_model"]] += 1
            
            comparison["win_statistics"] = {
                "total_samples": len(comparison["sample_level"]),
                "lora_wins": wins["lora"],
                "base_wins": wins["base"],
                "ties": wins["tie"],
                "lora_win_rate": wins["lora"] / len(comparison["sample_level"]),
                "base_win_rate": wins["base"] / len(comparison["sample_level"])
            }
        
        return comparison
    
    def _determine_better_model(
        self,
        lora_metrics: Dict[str, float],
        base_metrics: Dict[str, float]
    ) -> str:
        """
        根据指标确定哪个模型更好
        
        Args:
            lora_metrics: LoRA模型指标
            base_metrics: 基础模型指标
            
        Returns:
            "lora", "base" 或 "tie"
        """
        # 定义重要指标及其权重
        metric_weights = {
            "exact_match": 3.0,
            "soft_match": 2.0,
            "rouge_l": 1.5,
            "bleu_4": 1.0
        }
        
        lora_score = 0
        base_score = 0
        
        for metric, weight in metric_weights.items():
            if metric in lora_metrics and metric in base_metrics:
                if lora_metrics[metric] > base_metrics[metric]:
                    lora_score += weight
                elif base_metrics[metric] > lora_metrics[metric]:
                    base_score += weight
        
        if lora_score > base_score:
            return "lora"
        elif base_score > lora_score:
            return "base"
        else:
            return "tie"
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: str):
        """保存评估结果"""
        # 保存完整结果
        full_results_file = os.path.join(output_dir, "evaluation_metrics.json")
        with open(full_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved full evaluation results to: {full_results_file}")
        
        # 保存简化的指标摘要
        summary = {
            "metadata": results["metadata"],
            "metrics": results["metrics"]
        }
        
        if "comparison" in results and "overall" in results["comparison"]:
            summary["comparison_summary"] = {
                "overall": results["comparison"]["overall"],
                "perplexity": results["comparison"].get("perplexity", {}),
                "win_statistics": results["comparison"].get("win_statistics", {})
            }
        
        summary_file = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved metrics summary to: {summary_file}")
        
        # 保存详细的对比报告
        if "comparison" in results:
            comparison_file = os.path.join(output_dir, "model_comparison.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(results["comparison"], f, indent=2, ensure_ascii=False)
            print(f"💾 Saved model comparison to: {comparison_file}")
        
        # 打印结果摘要
        self._print_evaluation_summary(results)
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """打印评估摘要"""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        metrics = results["metrics"]
        
        # 打印每个模型的指标
        for model_type, model_metrics in metrics.items():
            print(f"\n🤖 {model_type.upper()} MODEL METRICS:")
            print("-" * 50)
            
            # 总体指标
            if "overall" in model_metrics:
                print("\n🎯 Overall Metrics:")
                overall = model_metrics["overall"]
                for key, value in overall.items():
                    if key.startswith('eval_') and isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
            
            # 困惑度指标
            if "perplexity" in model_metrics:
                print("\n🧠 Perplexity Metrics:")
                perplexity = model_metrics["perplexity"]
                for key, value in perplexity.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
            
            # 分类指标
            if "by_type" in model_metrics:
                print("\n📋 Metrics by Type:")
                by_type = model_metrics["by_type"]
                for item_type, type_metrics in by_type.items():
                    print(f"\n  📂 {item_type.upper()}:")
                    for key, value in type_metrics.items():
                        if key.startswith('eval_') and isinstance(value, (int, float)):
                            print(f"    {key}: {value:.4f}")
        
        # 打印对比结果
        if "comparison" in results:
            print("\n" + "="*80)
            print("📈 MODEL COMPARISON")
            print("="*80)
            
            comparison = results["comparison"]
            
            # 总体对比
            if "overall" in comparison:
                print("\n🎯 Overall Comparison:")
                for metric_name, comp in comparison["overall"].items():
                    print(f"\n  {metric_name}:")
                    print(f"    Base:  {comp['base']:.4f}")
                    print(f"    LoRA:  {comp['lora']:.4f}")
                    print(f"    Improvement: {comp['improvement_%']:+.2f}%")
                    print(f"    Better: {comp['better_model'].upper()}")
            
            # 困惑度对比
            if "perplexity" in comparison:
                print("\n🧠 Perplexity Comparison:")
                for metric_name, comp in comparison["perplexity"].items():
                    print(f"\n  {metric_name}:")
                    print(f"    Base:  {comp['base']:.4f}")
                    print(f"    LoRA:  {comp['lora']:.4f}")
                    print(f"    Improvement: {comp['improvement_%']:+.2f}%")
                    print(f"    Better: {comp['better_model'].upper()}")
            
            # 获胜统计
            if "win_statistics" in comparison:
                print("\n🏆 Win Statistics:")
                stats = comparison["win_statistics"]
                print(f"  Total samples: {stats['total_samples']}")
                print(f"  LoRA wins: {stats['lora_wins']} ({stats['lora_win_rate']*100:.1f}%)")
                print(f"  Base wins: {stats['base_wins']} ({stats['base_win_rate']*100:.1f}%)")
                print(f"  Ties: {stats['ties']}")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model metrics from inference results")
    
    # 输入输出参数
    parser.add_argument("--results_file", type=str, 
                        default=None)
    parser.add_argument("--type_mapping_file", type=str, 
                        default=None,
                        help="Path to type mapping JSON file for categorized evaluation")
    # 评估选项
    parser.add_argument("--evaluate_lora", action="store_true", default=True,
                        help="Evaluate LoRA model results")
    parser.add_argument("--no_lora", dest="evaluate_lora", action="store_false",
                        help="Don't evaluate LoRA model")
    parser.add_argument("--evaluate_base", action="store_true", default=True,
                        help="Evaluate base model results")
    parser.add_argument("--no_base", dest="evaluate_base", action="store_false",
                        help="Don't evaluate base model")
    parser.add_argument("--compare_models", action="store_true", default=True,
                        help="Compare LoRA and base models")
    parser.add_argument("--no_compare", dest="compare_models", action="store_false",
                        help="Don't compare models")
    
    args = parser.parse_args()
    

    args.output_dir = os.path.join(os.path.dirname(args.results_file), 'analysis')

    if args.type_mapping_file:
        load_type_mapping(args.type_mapping_file)
    
    
    # 验证参数
    if not args.evaluate_lora and not args.evaluate_base:
        parser.error("At least one of --evaluate_lora or --evaluate_base must be enabled")
    
    if args.compare_models and not (args.evaluate_lora and args.evaluate_base):
        print("⚠️ Model comparison requires both models to be evaluated")
        args.compare_models = False
    
    # 检查输入文件
    if not os.path.exists(args.results_file):
        parser.error(f"Results file not found: {args.results_file}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估器
    print("🚀 Initializing metrics evaluator...")
    evaluator = MetricsEvaluator(type_mapping_file=args.type_mapping_file)
    
    # 运行评估
    print("📊 Starting evaluation...")
    results = evaluator.evaluate_from_results(
        results_file=args.results_file,
        output_dir=args.output_dir,
        evaluate_lora=args.evaluate_lora,
        evaluate_base=args.evaluate_base,
        compare_models=args.compare_models
    )
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    # 打印输出文件列表
    print("\n📄 Generated files:")
    output_files = os.listdir(args.output_dir)
    for file in sorted(output_files):
        print(f"  - {file}")


if __name__ == "__main__":
    main()