# Qwen-VL-Med-SFT
![Qwen-VL-Med-SFT 流程](flowchart0.png)
一个基于 Qwen2-VL 系列模型的医学视觉语言模型微调框架，专门针对生物医学领域的多模态任务进行优化。

## 🎯 项目概述

本项目基于 [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) 架构，使用 HuggingFace Transformers 和 DeepSpeed 进行训练。在原有基础上增强了以下功能：

- ✅ 训练过程中的验证模块
- ✅ 数据预处理检查
- ✅ 模型效果验证
- ✅ 结果统计对比分析

项目采用 [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) 数据集，实现两阶段微调策略：**概念对齐**（Concept Alignment）和**指令跟随**（Instruction Following）。
项目采用 [Med-GRIT-270K](https://github.com/ShawnHuang497/BiRD) 数据集，进行grounding相关能力的效果优化
项目采用 Reasonging&direct 为基于 LLaVA-Med 的instruct 数据集自建，代码 [`data_process/reason_direct_gene.py`]

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

### 📈 实验结果

#### Qwen2-VL 生物医疗问答微调与性能分析

*  微调方案与实验设置

* 概念对齐微调（Stage 1）
- 基于 Qwen2-VL-Instruct 2B 与 7B 模型
- 使用约 16 万条多模态对齐数据，主要采用 LoRA 技术微调视觉-语言融合模块（visual-merger-proj）

* Instruct SFT 微调（Stage 2）
- 采用 LoRA 微调 attention 层 FNN 部分，提升信息整合能力
- **方案 1**：基于 LLaVA-Med Instruct 数据集（16K），3 轮训练，显著提升回复质量，困惑度下降，Word F1 提高
- **方案 2**：基于 Slake 训练集（约 5K），3 轮训练，回复风格更简洁，Word F1 显著提升

- **测试集**：Slake 生物医学视觉问答数据集

#### 结果总结

- **概念对齐微调**：有助于视觉与语言信息的专业领域对齐，增强模型在医学问答中的表现
- **Instruct SFT**：进一步优化模型回复风格与语言表达
- **模型局限**：
  - 病变位置确认与相对位置表述能力仍待提升
  - 对部分病症有“轻度”描述倾向
* 表格说明：

| 模型简称                            | 训练方式及说明                                                |
| ----------------------------------- | ------------------------------------------------------------ |
| Stage1               | Qwen2-VL-2B instruct，概念对齐微调        |
| Stage1CTL                | Qwen2-VL-2B instruct 概念对齐微调  context_learning       |
| Stage2               | Qwen2-VL-2B instruct，概念对齐微调+指令微调       |
| 7BOri                       | Qwen2-VL-7B instruct，无微调                   |
| Stage1Reasoning                           | Qwen2-VL-7B 概念对齐微调  Reasoning_SFT                      |
| Stage1ReasoningCTL                | Qwen2-VL-7B 概念对齐微调  Reasoning_SFT  context Learning             |
| Qwen25VL32BInst                     | Qwen2-VL-32B instruct 无训练                   |

- **Reasoning训练**：针对每个QA对添加了Reasoning prompt，使用Qwen3根据LLaVA Med QA训练集中的一部分，将问题的answer作为输入，生成Reasoning回答和Directly回答，构建样本集。
* 数据集构建代码见：data_process/reason_direct_gene.py

#### 性能对比
* **整体性能** 

详细结果见 [`result/model_comparison_tables.txt`](result/model_comparison_tables.txt)，  
个例统计见 [`result/evaluation_metrics.json`](result/evaluation_metrics.json)

| Category | Metric | Stage1CTL | Stage1 | Stage2 | Stage1Reasoning | Stage1ReasoningCTL | 7BOriCTL | 7BOri | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.4807 | **0.4816** | 0.4722 | 0.0452 | 0.2413 | 0.2705 | 0.0000 | 0.0000 |
| Text Matching | Soft Match | 0.5822 | **0.5835** | 0.5791 | 0.3156 | 0.4713 | 0.3630 | 0.0960 | 0.0037 |
| Text Matching | ROUGE-L | 0.5267 | **0.5285** | 0.5125 | 0.3200 | 0.4623 | 0.3331 | 0.0897 | 0.0156 |
| Text Matching | BLEU-4 | 0.0916 | **0.0920** | 0.0899 | 0.0432 | 0.0736 | 0.0540 | 0.0082 | 0.0009 |
| Text Matching | Word Overlap | 0.5077 | **0.5088** | 0.4954 | 0.2514 | 0.4189 | 0.3092 | 0.0565 | 0.0125 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Word Contain | 0.5938 | 0.5957 | 0.5533 | 0.5749 | 0.6136 | 0.6051 | 0.5975 | **0.7135** |
| Fine-grained | Char F1 | 0.6554 | **0.6569** | 0.6247 | 0.4249 | 0.5550 | 0.5466 | 0.3525 | 0.3236 |
| Fine-grained | Word F1 | 0.5198 | **0.5211** | 0.5052 | 0.3088 | 0.4533 | 0.3362 | 0.0994 | 0.0244 |
| Fine-grained | Char Precision | 0.6541 | **0.6557** | 0.6370 | 0.3581 | 0.5132 | 0.4875 | 0.2581 | 0.2264 |
| Fine-grained | Char Recall | 0.6958 | 0.6992 | 0.6383 | 0.7208 | 0.7159 | 0.7640 | 0.7999 | **0.8596** |
| Fine-grained | Word Precision | 0.5176 | **0.5192** | 0.5095 | 0.2628 | 0.4314 | 0.3106 | 0.0574 | 0.0125 |
| Fine-grained | Word Recall | 0.5490 | 0.5497 | 0.5154 | 0.5200 | 0.5667 | 0.5457 | 0.5403 | **0.6631** |


## Open-ended Questions

| Category | Metric | Stage1CTL | Stage1 | Stage2 | Stage1Reasoning | Stage1ReasoningCTL | 7BOriCTL | 7BOri | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | **0.3371** | 0.3371 | 0.3329 | 0.0609 | 0.2266 | 0.1601 | 0.0000 | 0.0000 |
| Text Matching | Soft Match | 0.4866 | 0.4872 | **0.4934** | 0.3782 | 0.4627 | 0.2859 | 0.1123 | 0.0029 |
| Fine-grained | Word Contain | 0.4986 | 0.5000 | 0.4547 | 0.5297 | 0.5326 | 0.5368 | 0.5467 | **0.7394** |
| Fine-grained | Word F1 | 0.3902 | 0.3908 | 0.3824 | 0.3319 | **0.3969** | 0.2381 | 0.0930 | 0.0259 |
| Fine-grained | Word Precision | 0.3883 | **0.3893** | 0.3889 | 0.2942 | 0.3802 | 0.2076 | 0.0546 | 0.0133 |
| Fine-grained | Word Recall | 0.4312 | 0.4310 | 0.3978 | 0.4472 | 0.4621 | 0.4475 | 0.4607 | **0.6637** |


* 总体结果分析
- **常规测试**：Qwen系列模型对医疗的文字有解答和回答的能力，但是对于医疗图像认知稍显不足，对于常规的MRI、X-Ray等有一定认知，但是对于病灶、病例判断不足，对于生物组织、细胞图的认识稍微差一些
- **Reasoning**：Reasoning虽然没有完全生成遵循Reasoning Propmt的能力，但是数据集相对应Stage2是一个更高质量，回答更加简洁信息的数据集，因此在Open-ended数据集的ROUGE-L统计上，效果有了一个较大的提升，说明通过指令SFT可以提升模型对于专业领域任务的理解
- **封闭式问题**：LoRA 模型在所有指标上都有显著提升，特别是 Word F1 从 0.1121 提升到 0.7493
- **开放式问题**：Word F1 从 0.0930 提升到 0.3824
- **整体性能**：Word F1 从 0.0994 提升到 0.5052，LoRA 微调后模型在医学视觉问答任务上展现出更好的理解和生成能力

* 结果情况：（请忽略不准确的机器翻译）
![result compare 1](imgs/img1.jpg)


### Grounding能力效果优化
* **数据集介绍**


* **结果分析**
* Visual Grounding (200 sample)
| Metric | ReasonLora32 | ReasonLoraPmt | OriPmt | GroundLoraPmt | Stage1AllPmt | Stage1Lora16PMT | Stage1 | ReasonLora | RefausePmt |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Consistency Rate | 0.0500 | 0.7600 | 0.7950 | 0.7273 | **0.8800** | 0.8800 | 0.0600 | 0.0500 | 0.2800 |
| Average IoU | NAN | 0.1232 | 0.1695 | 0.4301 | 0.1927 | 0.1861 | NAN | NAN | 0.1141 |
| IoU > 0.5 Rate | NAN | 0.0043 | 0.0303 | **0.2667** | 0.0433 | 0.0519 | NAN | NAN | 0.0043 |
| Center Inclusion Accuracy | NAN | 0.0996 | 0.1255 | **0.3333** | 0.1429 | 0.2338 | 0.0087 | 0.0000 | 0.0736 |
| Average CPE | NAN | 111.4301 | 114.6439 | **69.0272** | 121.8034 | 162.8852 | NAN | NAN | 27.5950 |
| Avg Position Match Rate | 0.0768 | 0.1690 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.2165** | 0.0768 | 0.0000 |

* 指标介绍：
| 指标               | 说明                                                   | 指标                        | 说明                                                |
| ---------------- | ---------------------------------------------------- | ------------------------- | ------------------------------------------------- |
| Consistency Rate | Predict中包含有Boxes的比例              | Average IoU               | 平均交并比（Intersection over Union），衡量预测框与真实框重叠程度的平均值。 |
| IoU > 0.5 Rate   | 预测框与真实框的 IoU 超过 0.5 的帧（或样本）占总数的比例   | Center Inclusion Accuracy | 预测框中心点落在真实框内部的比例   |
| Average CPE      | 平均中心点误差（Center Point Error） | Avg Position Match Rate   | 位置关键词匹配    |
Avg Position Match Rate: 将坐标转换为图像的相对位置，对比输出结果中是否有关键词匹配

*  Refering Object Classification (200 sample)
| Category | Metric | ReasonLora | Stage1Lora | Stage1All | Ground | Ori |
|----------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.0000 | 0.0100 | 0.0000 | **0.5500** | 0.0000 |
| Text Matching | Soft Match | 0.2544 | 0.1616 | 0.1082 | **0.7217** | 0.0588 |
| Text Matching | Word Overlap | 0.0379 | 0.0240 | 0.0092 | **0.6217** | 0.0018 |
| Text Matching | BERTScore F1 | 0.3513 | 0.2746 | 0.2449 | **0.7467** | 0.2731 |
| Text Matching | ROUGE 1 | 0.0211 | 0.0102 | 0.0117 | **0.6088** | 0.0000 |
| Text Matching | METOR | 0.0306 | 0.0243 | 0.0244 | **0.4009** | 0.0053 |
| Fine-grained | Word Contain | 0.0800 | 0.0800 | 0.1200 | **0.6600** | 0.0909 |
| Fine-grained | Char F1 | 0.4395 | 0.4493 | 0.4678 | **0.7712** | 0.4680 |
| Fine-grained | Word F1 | 0.0479 | 0.0313 | 0.0169 | **0.6330** | 0.0035 |
| Fine-grained | Char Precision | 0.4178 | 0.3667 | 0.3392 | **0.8119** | 0.3268 |
| Fine-grained | Char Recall | 0.5532 | 0.7409 | 0.8796 | 0.7918 | **0.8855** |
| Fine-grained | Word Precision | 0.0504 | 0.0318 | 0.0096 | **0.6408** | 0.0018 |
| Fine-grained | Word Recall | 0.0512 | 0.0554 | 0.0896 | **0.6308** | 0.0909 |


* 总体分析：
- **Gronding**：通过propmt引导，模型本身就可以具备给出Boxes的能力，没有进行概念对齐Ori7B会出现检测偏差较大和概念模糊的问题；经过概念对齐后，模型可以有较好的器官检测能力，病灶检测能力不足，偏向于检测更大区域，且多框检测的能力不足；Refause的CPE最低，是因为经过拒绝感知，模型只对自己有完全把握的问题进行boxes输出，对于医疗镜像图像容易左右位置混淆；另外就是对于低分辨率图像，效果不佳。
- **回复能力**：通过Grounding数据训练后，模型在针对location相关回复时，会直接给出boxes而不是基于位置的模糊回复，Reason数据集在没有给出Reasoning Require的propmt时，回复简洁清晰；经过Reasoning和direct sft后，模型有了明显的判断能力，指进行stage1 微调的时候，模型对于问题的判断解答存在一定的偏向性。

* 结果展示：
- 部分图像展示：[Grounding](./imgs/grounding)

## 🔧 环境要求

requirement.txt

## 📚 参考资料

- [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)
- [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [Qwen2-VL 官方文档](https://github.com/QwenLM/Qwen2-VL)

## 📄 许可证

本项目遵循相应的开源许可证，请参考各个依赖项目的许可证要求。



# Qwen-VL-Med-SFT

![Qwen-VL-Med-SFT Workflow](flowchart0.png)

A medical vision-language model fine-tuning framework based on the Qwen2-VL series, specifically optimized for biomedical multimodal tasks.

## 🎯 Project Overview

This project is built upon the [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) architecture, utilizing HuggingFace Transformers and DeepSpeed for training. Enhanced features include:

- ✅ Validation module during training
- ✅ Data preprocessing validation
- ✅ Model effectiveness verification
- ✅ Statistical comparison analysis

The project uses the [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) dataset and implements a two-stage fine-tuning strategy: **Concept Alignment** and **Instruction Following**.

## 📊 Data Preprocessing Pipeline

### Training Data Processing
1. **Data Format Conversion**
   - Generate standard JSON files following [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) format
   - Iterative validation according to DataLoader processing to filter valid data

2. **Validation Data Processing**
   - Use `./data_process/data_trans.py` script for processing
   - Generate `test.json` and `type_mapping.json` for testing and classification

### Data Classification System

Based on question types, we categorize data into the following classes:


## 🚀 Model Training

### Training Architecture
Based on HuggingFace Transformers' `Trainer` class with integrated callbacks via `trainer.add_callback(callback)`:
- Validation workflow callbacks
- TensorBoard logging
- Training process monitoring

### Starting Training

#### Single GPU Training/Debugging
```bash
bash finetune_lora_single_gpu.sh
```

#### Multi-GPU Training (DeepSpeed)
```bash
bash finetune_lora_mult_gpu.sh
```

## 🧪 Model Testing and Evaluation

### Test Configuration
Supports two testing modes:
- **Custom Prompt Testing**: Using custom prompts
- **Original System Prompt Testing**: Using model default prompts

The testing process calculates perplexity and compares results between base and LoRA models.

### Running Tests
```bash
bash run_eval.sh
```

### Statistical Result Analysis
Use the `result_statistic.py` script for category-wise statistical analysis.

#### Evaluation Metrics System
| Metric Name | Value Range | Reasonable Score Range | Meaning | Description |
|-------------|-------------|----------------------|---------|-------------|
| **BLEU-4** | 0 ~ 1 | 0.2 ~ 0.6 | n-gram precision matching rate | Measures local language matching, penalizes repetition and missing words |
| **ROUGE-L** | 0 ~ 1 | 0.3 ~ 0.6 | Longest common subsequence recall | Emphasizes information coverage, favors recall |
| **METEOR** | 0 ~ 1 | 0.3 ~ 0.5 | Comprehensive metric: precision + recall + semantic synonyms + word order penalty | Tolerant of expression differences, suitable for generative tasks |
| **CIDEr** | 0 ~ ∞ | 0.5 ~ 2.0 | TF-IDF weighted n-gram matching score | Robust with multiple references, common in vision-language tasks |
| **BERTScore (F1)** | 0 ~ 1 | 0.85 ~ 0.95 | BERT-based sentence vector semantic similarity | Effective even with flexible expressions |
| **Soft Matching** | 0 ~ 1 | 0.6 ~ 0.95 | Character-level similarity (e.g., SequenceMatcher/LCS ratio) | Captures partial string similarity, strong fault tolerance |
| **Substring Match** | 0 or 1 | Binary (0/1) | Whether completely contained (reference ∈ prediction or vice versa) | Very strict, suitable for short and clear answer scenarios |

## 📈 Experimental Results

Based on Qwen2-VL-Instruct with concept alignment fine-tuning, applying LoRA technique to the visual-merger-proj module. Comparison results available in: `result/evaluation_metrics.json`

### Performance Comparison
| Category | Metric | Stage1CTL | Stage1 | Stage2 | Stage1Reasoning | Stage1ReasoningCTL | 7BOriCTL | 7BOri | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.4807 | **0.4816** | 0.4722 | 0.0452 | 0.2413 | 0.2705 | 0.0000 | 0.0000 |
| Text Matching | Soft Match | 0.5822 | **0.5835** | 0.5791 | 0.3156 | 0.4713 | 0.3630 | 0.0960 | 0.0037 |
| Text Matching | ROUGE-L | 0.5267 | **0.5285** | 0.5125 | 0.3200 | 0.4623 | 0.3331 | 0.0897 | 0.0156 |
| Text Matching | BLEU-4 | 0.0916 | **0.0920** | 0.0899 | 0.0432 | 0.0736 | 0.0540 | 0.0082 | 0.0009 |
| Text Matching | Word Overlap | 0.5077 | **0.5088** | 0.4954 | 0.2514 | 0.4189 | 0.3092 | 0.0565 | 0.0125 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Word Contain | 0.5938 | 0.5957 | 0.5533 | 0.5749 | 0.6136 | 0.6051 | 0.5975 | **0.7135** |
| Fine-grained | Char F1 | 0.6554 | **0.6569** | 0.6247 | 0.4249 | 0.5550 | 0.5466 | 0.3525 | 0.3236 |
| Fine-grained | Word F1 | 0.5198 | **0.5211** | 0.5052 | 0.3088 | 0.4533 | 0.3362 | 0.0994 | 0.0244 |
| Fine-grained | Char Precision | 0.6541 | **0.6557** | 0.6370 | 0.3581 | 0.5132 | 0.4875 | 0.2581 | 0.2264 |
| Fine-grained | Char Recall | 0.6958 | 0.6992 | 0.6383 | 0.7208 | 0.7159 | 0.7640 | 0.7999 | **0.8596** |
| Fine-grained | Word Precision | 0.5176 | **0.5192** | 0.5095 | 0.2628 | 0.4314 | 0.3106 | 0.0574 | 0.0125 |
| Fine-grained | Word Recall | 0.5490 | 0.5497 | 0.5154 | 0.5200 | 0.5667 | 0.5457 | 0.5403 | **0.6631** |

### Key Findings
- **Closed-set Questions**: LoRA model shows significant improvement across all metrics, particularly Exact Match improving from 0.000 to 0.341
- **Open-end Questions**: Although improvements are relatively smaller, notable enhancements in Soft Match and ROUGE-L metrics
- **Overall Performance**: LoRA fine-tuned model demonstrates better understanding and generation capabilities in medical visual question answering tasks

## 🔧 Environment Requirements

See `requirements.txt` for detailed dependencies.

## 📚 References

- [2U1/Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune)
- [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [Qwen2-VL Official Documentation](https://github.com/QwenLM/Qwen2-VL)

## 📄 License

This project follows the corresponding open-source licenses. Please refer to the license requirements of each dependency project.

---

## 🌏 Language Versions

- [English](README.md)
- [中文](README_zh.md)
