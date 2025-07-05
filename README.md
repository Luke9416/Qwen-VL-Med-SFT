# Qwen-VL-Med-SFT
![Qwen-VL-Med-SFT 流程](flowchart.png)
一个基于 Qwen2-VL 系列模型的医学视觉语言模型微调框架，专门针对生物医学领域的多模态任务进行优化。

## 🎯 项目概述

本项目基于 [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) 架构，使用 HuggingFace Transformers 和 DeepSpeed 进行训练。在原有基础上增强了以下功能：

- ✅ 训练过程中的验证模块
- ✅ 数据预处理检查
- ✅ 模型效果验证
- ✅ 结果统计对比分析

项目采用 [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) 数据集，实现两阶段微调策略：**概念对齐**（Concept Alignment）和**指令跟随**（Instruction Following）。

## 📊 数据预处理流程

### 训练数据处理

1. **数据格式转换**
   - 参考 [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) 生成标准 JSON 文件
   - 按照 DataLoader 的数据处理方式进行迭代验证，筛选有效数据

2. **验证数据处理**
   - 使用 `./data_process/data_trans.py` 脚本处理
   - 生成 `test.json` 和 `type_mapping.json` 用于测试和分类

### 数据分类体系

根据问题类型，我们将数据分为以下类别：

#### 封闭式问题（Close-set）

| 类型 | 任务描述 |
|------|----------|
| **Yes/No 判断** | 对医学影像中特定特征或异常的存在性进行二元判断 |
| **模态识别** | 识别医学影像的成像方式或技术类型 |
| **通用问题** | 不属于特定类别的封闭式问题 |
| **位置定位** | 询问病变或结构的具体位置（限定选项） |

#### 开放式问题（Open-end）

| 类型 | 任务描述 |
|------|----------|
| **通用描述** | 需要综合描述或解释的开放性问题 |
| **解剖识别** | 识别和描述影像中的解剖结构或器官 |
| **位置描述** | 详细描述病变或结构的位置 |
| **异常识别** | 识别和描述病理改变或异常发现 |
| **计数任务** | 计算影像中特定对象的数量 |
| **比较分析** | 比较不同结构或时间点的变化 |
| **外观描述** | 描述病变或结构的视觉特征 |
| **影响评估** | 评估病变对周围结构的影响 |

## 🚀 模型训练

### 训练架构

基于 HuggingFace Transformers 的 `Trainer` 类进行训练，通过 `trainer.add_callback(callback)` 方式集成：
- 验证流程回调
- TensorBoard 日志记录
- 训练过程监控

### 启动训练

#### 单卡训练/调试
```bash
bash finetune_lora_single_gpu.sh
```

#### 多卡训练（DeepSpeed）
```bash
bash finetune_lora_mult_gpu.sh
```

## 🧪 模型测试与评估

### 测试配置

支持两种测试模式：
- **自定义 Prompt 测试**：使用自设定的提示词
- **原始 System Prompt 测试**：使用模型默认提示词

测试过程会计算困惑度（Perplexity），并对比基础模型和 LoRA 模型的结果。

### 启动测试
```bash
bash run_eval.sh
```

### 结果统计分析

使用 `result_statistic.py` 脚本进行分类别统计分析。

#### 评估指标体系

| 指标名称 | 值域范围 | 合理分数区间 | 含义解释 | 说明 |
|----------|----------|--------------|----------|------|
| **BLEU-4** | 0 ~ 1 | 0.2 ~ 0.6 | n-gram 精确匹配率 | 衡量局部语言匹配，惩罚重复、缺词 |
| **ROUGE-L** | 0 ~ 1 | 0.3 ~ 0.6 | 最长公共子序列的召回率 | 重视信息覆盖，偏向召回率 |
| **METEOR** | 0 ~ 1 | 0.3 ~ 0.5 | 精确度 + 召回 + 语义同义词 + 词序惩罚综合指标 | 宽容表达差异，适合生成式任务 |
| **CIDEr** | 0 ~ ∞ | 0.5 ~ 2.0 | 基于 TF-IDF 的加权 n-gram 匹配评分 | 多参考时鲁棒，常用于图文任务 |
| **BERTScore (F1)** | 0 ~ 1 | 0.85 ~ 0.95 | 基于 BERT 的句向量语义相似度 | 表达灵活时依旧有效 |
| **Soft Matching** | 0 ~ 1 | 0.6 ~ 0.95 | 字符级相似度（如 SequenceMatcher/LCS 比例） | 捕捉字符串的部分相似，容错性强 |
| **Substring Match** | 0 或 1 | 二值型（0/1） | 是否完全包含（参考 ∈ 预测 或反之） | 非常严格，适用于答案短明确场景 |

## 📈 实验结果

基于 Qwen2-VL-Instruct 进行概念对齐微调，对 visual-merger-proj 模块应用 LoRA 技术。对比结果json:result/evaluation_metrics.json

### 性能对比

| 数据类型 | 模型 | Exact Match | Soft Match | Word Overlap | BLEU-4 | ROUGE-L |
|----------|------|-------------|------------|--------------|--------|---------|
| **Overall** | Base | 0.000 | 0.056 | 0.034 | 0.004 | 0.051 |
|  | **LoRA** | **0.117** | **0.238** | **0.175** | **0.030** | **0.210** |
| **Open-end** | Base | 0.000 | 0.069 | 0.035 | 0.004 | 0.052 |
|  | **LoRA** | **0.004** | **0.163** | **0.071** | **0.012** | **0.109** |
| **Closed-set** | Base | 0.000 | 0.030 | 0.031 | 0.004 | 0.048 |
|  | **LoRA** | **0.341** | **0.388** | **0.384** | **0.067** | **0.411** |

### 关键发现

- **封闭式问题**：LoRA 模型在所有指标上都有显著提升，特别是 Exact Match 从 0.000 提升到 0.341
- **开放式问题**：虽然提升相对较小，但在 Soft Match 和 ROUGE-L 指标上仍有明显改善
- **整体性能**：LoRA 微调后模型在医学视觉问答任务上展现出更好的理解和生成能力

## 🔧 环境要求

requirement.txt

## 📚 参考资料

- [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)
- [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [Qwen2-VL 官方文档](https://github.com/QwenLM/Qwen2-VL)

## 📄 许可证

本项目遵循相应的开源许可证，请参考各个依赖项目的许可证要求。