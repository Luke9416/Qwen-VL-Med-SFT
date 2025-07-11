# Multi-Model Comparison Report

Generated at: 2025-07-11 06:59:54

Number of models: 7

## Models Information

| Model Name | Base Model | Total Samples | Custom Prompt | Language |
|------------|------------|---------------|---------------|----------|
| Qwen2VL2BInst_Stage1 | Qwen_2_VL_Med_stage1 | 1061 | ✓ | en |
| Qwen2VL7BInst | Qwen2-VL-7B-Instruct | 1061 | ✓ | en |
| Qwen2VL7B | Qwen2-VL-7B | 1061 | ✓ | en |
| Qwen2VL7BBase_Stage1 | stage1_Qwen7BBase_stage1 | 1061 | ✓ | en |
| Qwen2VL7BInst_Stage1 | stage1_Qwen7BInst_stage1 | 1061 | ✓ | en |
| Qwen2VL7BInst_Stage2 | stage1_Qwen7BInst_stage2 | 1061 | ✓ | en |
| Qwen25VL32BInst | Qwen2.5-VL-32B-Instruct | 1061 | ✓ | en |

## Overall Results
*Sample counts: Qwen2VL2BInst_Stage1: 1061 samples, Qwen2VL7BInst: 1061 samples, Qwen2VL7B: 1061 samples, Qwen2VL7BBase_Stage1: 1061 samples, Qwen2VL7BInst_Stage1: 1061 samples, Qwen2VL7BInst_Stage2: 1061 samples, Qwen25VL32BInst: 1061 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

| Category | Metric | Qwen2VL2BInst_Stage1 | Qwen2VL7BInst | Qwen2VL7B | Qwen2VL7BBase_Stage1 | Qwen2VL7BInst_Stage1 | Qwen2VL7BInst_Stage2 | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.2026 | 0.0000 | 0.0000 | 0.1225 | 0.4354 | **0.4722** | 0.0000 |
| Text Matching | Soft Match | 0.3291 | 0.0960 | 0.0139 | 0.2949 | 0.5477 | **0.5791** | 0.0037 |
| Text Matching | ROUGE-L | 0.2796 | 0.0897 | 0.0254 | 0.2669 | 0.4809 | **0.5125** | 0.0156 |
| Text Matching | BLEU-4 | 0.0458 | 0.0082 | 0.0019 | 0.0420 | 0.0845 | **0.0899** | 0.0009 |
| Text Matching | Word Overlap | 0.2513 | 0.0565 | 0.0192 | 0.2107 | 0.4621 | **0.4954** | 0.0125 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Char F1 | 0.4898 | 0.3525 | 0.3322 | 0.4378 | 0.6094 | **0.6247** | 0.3236 |
| Fine-grained | Word F1 | 0.2844 | 0.0994 | 0.0368 | 0.2607 | 0.4733 | **0.5052** | 0.0244 |
| Fine-grained | Char Precision | 0.4288 | 0.2581 | 0.2400 | 0.3699 | 0.6095 | **0.6370** | 0.2264 |
| Fine-grained | Char Recall | 0.7300 | 0.7999 | 0.7973 | 0.7368 | 0.6523 | 0.6383 | **0.8596** |
| Fine-grained | Word Precision | 0.2551 | 0.0574 | 0.0193 | 0.2173 | 0.4717 | **0.5095** | 0.0125 |
| Fine-grained | Word Recall | 0.4746 | 0.5403 | 0.5565 | 0.5060 | 0.5028 | 0.5154 | **0.6631** |
| Perplexity | Mean Perplexity | 882.7204 | 240741.5970 | 621617.2925 | 798.9098 | 278.8062 | **173.8327** | 694010380428.6918 |
| Perplexity | Median Perplexity | 31.6250 | 216.0000 | 66.0000 | 9.9375 | 3.8906 | **2.8906** | 13376.0000 |
| Perplexity | Std Perplexity | 7680.5538 | 1523977.9827 | 12043923.3773 | 10558.1424 | 2424.2811 | **1582.0417** | 4472871553015.6406 |
| Perplexity | Min Perplexity | 1.4297 | **1.0000** | 1.0391 | 1.0391 | 1.0391 | 1.0156 | 1.0000 |
| Perplexity | Max Perplexity | 152576.0000 | 21364736.0000 | 293601280.0000 | 323584.0000 | 52736.0000 | **32000.0000** | 61572651155456.0000 |

## Open-ended Questions
*Sample counts: Qwen2VL2BInst_Stage1: 706 samples, Qwen2VL7BInst: 706 samples, Qwen2VL7B: 706 samples, Qwen2VL7BBase_Stage1: 706 samples, Qwen2VL7BInst_Stage1: 706 samples, Qwen2VL7BInst_Stage2: 706 samples, Qwen25VL32BInst: 706 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

| Category | Metric | Qwen2VL2BInst_Stage1 | Qwen2VL7BInst | Qwen2VL7B | Qwen2VL7BBase_Stage1 | Qwen2VL7BInst_Stage1 | Qwen2VL7BInst_Stage2 | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.0595 | 0.0000 | 0.0000 | 0.0397 | 0.2805 | **0.3329** | 0.0000 |
| Text Matching | Soft Match | 0.2332 | 0.1123 | 0.0131 | 0.2637 | 0.4488 | **0.4934** | 0.0029 |
| Text Matching | ROUGE-L | 0.1504 | 0.0844 | 0.0253 | 0.2084 | 0.3480 | **0.3934** | 0.0167 |
| Text Matching | BLEU-4 | 0.0229 | 0.0076 | 0.0018 | 0.0326 | 0.0604 | **0.0681** | 0.0009 |
| Text Matching | Word Overlap | 0.1179 | 0.0531 | 0.0194 | 0.1422 | 0.3202 | **0.3677** | 0.0133 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Char F1 | 0.4507 | 0.4205 | 0.4050 | 0.4421 | 0.5426 | **0.5626** | 0.3947 |
| Fine-grained | Word F1 | 0.1564 | 0.0930 | 0.0372 | 0.1964 | 0.3367 | **0.3824** | 0.0259 |
| Fine-grained | Char Precision | 0.3755 | 0.3234 | 0.3062 | 0.3687 | 0.5430 | **0.5812** | 0.2873 |
| Fine-grained | Char Recall | 0.7067 | 0.7706 | 0.8061 | 0.7019 | 0.6014 | 0.5825 | **0.8794** |
| Fine-grained | Word Precision | 0.1236 | 0.0546 | 0.0196 | 0.1523 | 0.3345 | **0.3889** | 0.0133 |
| Fine-grained | Word Recall | 0.3464 | 0.4607 | 0.5403 | 0.4078 | 0.3760 | 0.3978 | **0.6637** |

## Closed-set Questions
*Sample counts: Qwen2VL2BInst_Stage1: 355 samples, Qwen2VL7BInst: 355 samples, Qwen2VL7B: 355 samples, Qwen2VL7BBase_Stage1: 355 samples, Qwen2VL7BInst_Stage1: 355 samples, Qwen2VL7BInst_Stage2: 355 samples, Qwen25VL32BInst: 355 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

| Category | Metric | Qwen2VL2BInst_Stage1 | Qwen2VL7BInst | Qwen2VL7B | Qwen2VL7BBase_Stage1 | Qwen2VL7BInst_Stage1 | Qwen2VL7BInst_Stage2 | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.4873 | 0.0000 | 0.0000 | 0.2873 | 0.7437 | **0.7493** | 0.0000 |
| Text Matching | Soft Match | 0.5198 | 0.0634 | 0.0154 | 0.3570 | 0.7443 | **0.7495** | 0.0053 |
| Text Matching | ROUGE-L | 0.5366 | 0.1002 | 0.0258 | 0.3834 | 0.7450 | **0.7493** | 0.0134 |
| Text Matching | BLEU-4 | 0.0914 | 0.0093 | 0.0021 | 0.0607 | 0.1324 | **0.1332** | 0.0010 |
| Text Matching | Word Overlap | 0.5167 | 0.0631 | 0.0188 | 0.3467 | 0.7444 | **0.7493** | 0.0109 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Char F1 | 0.5674 | 0.2173 | 0.1875 | 0.4293 | 0.7424 | **0.7481** | 0.1822 |
| Fine-grained | Word F1 | 0.5390 | 0.1121 | 0.0361 | 0.3886 | 0.7451 | **0.7493** | 0.0214 |
| Fine-grained | Char Precision | 0.5349 | 0.1283 | 0.1083 | 0.3722 | 0.7417 | **0.7480** | 0.1052 |
| Fine-grained | Char Recall | 0.7765 | **0.8582** | 0.7798 | 0.8061 | 0.7535 | 0.7493 | 0.8202 |
| Fine-grained | Word Precision | 0.5167 | 0.0631 | 0.0188 | 0.3467 | 0.7444 | **0.7493** | 0.0109 |
| Fine-grained | Word Recall | 0.7296 | 0.6986 | 0.5887 | 0.7014 | **0.7549** | 0.7493 | 0.6620 |

## Performance Summary

### Overall Results - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Qwen2VL7BInst_Stage2 | 13 | 76.5% |
| Qwen25VL32BInst | 2 | 11.8% |
| Qwen2VL2BInst_Stage1 | 1 | 5.9% |
| Qwen2VL7BInst | 1 | 5.9% |

### Open-ended Questions - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Qwen2VL7BInst_Stage2 | 9 | 75.0% |
| Qwen25VL32BInst | 2 | 16.7% |
| Qwen2VL2BInst_Stage1 | 1 | 8.3% |

### Closed-set Questions - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Qwen2VL7BInst_Stage2 | 9 | 75.0% |
| Qwen2VL2BInst_Stage1 | 1 | 8.3% |
| Qwen2VL7BInst | 1 | 8.3% |
| Qwen2VL7BInst_Stage1 | 1 | 8.3% |

## Overall Analysis

**Best Overall Results Model:** Qwen2VL7BInst_Stage2 with 13 best metrics out of 17 (76.5%)

**Best Open-ended Questions Model:** Qwen2VL7BInst_Stage2 with 9 best metrics out of 12 (75.0%)

**Best Closed-set Questions Model:** Qwen2VL7BInst_Stage2 with 9 best metrics out of 12 (75.0%)

**Model Strengths Across Different Question Types:**
- **Qwen2VL2BInst_Stage1**: Best in Overall Results (1 metrics), Open-ended Questions (1 metrics), Closed-set Questions (1 metrics)
- **Qwen2VL7BInst**: Best in Overall Results (1 metrics), Closed-set Questions (1 metrics)
- **Qwen2VL7B**: No significant advantages
- **Qwen2VL7BBase_Stage1**: No significant advantages
- **Qwen2VL7BInst_Stage1**: Best in Closed-set Questions (1 metrics)
- **Qwen2VL7BInst_Stage2**: Best in Overall Results (13 metrics), Open-ended Questions (9 metrics), Closed-set Questions (9 metrics)
- **Qwen25VL32BInst**: Best in Overall Results (2 metrics), Open-ended Questions (2 metrics)

**Key Observations:**
- Qwen2VL2BInst_Stage1: exact match is 8.2x better on closed-set questions (0.487 vs 0.059); ROUGE-L shows 256.7% difference between closed-set and open-ended
- Qwen2VL7BInst: exact match is infx better on closed-set questions (0.000 vs 0.000)
- Qwen2VL7B: exact match is infx better on closed-set questions (0.000 vs 0.000)
- Qwen2VL7BBase_Stage1: exact match is 7.2x better on closed-set questions (0.287 vs 0.040); ROUGE-L shows 84.0% difference between closed-set and open-ended
- Qwen2VL7BInst_Stage1: exact match is 2.7x better on closed-set questions (0.744 vs 0.280); ROUGE-L shows 114.1% difference between closed-set and open-ended
- Qwen2VL7BInst_Stage2: exact match is 2.3x better on closed-set questions (0.749 vs 0.333); ROUGE-L shows 90.5% difference between closed-set and open-ended
- Qwen25VL32BInst: exact match is infx better on closed-set questions (0.000 vs 0.000)

**Overall Performance Patterns:**