# Multi-Model Comparison Report

Generated at: 2025-07-22 10:21:51

Number of models: 8

## Models Information

| Model Name | Base Model | Total Samples | Custom Prompt | Language |
|------------|------------|---------------|---------------|----------|
| Stage1CTL | stage1_Qwen7BInst_stage1 | 1061 | ✓ | en |
| Stage1 | stage1_Qwen7BInst_stage1 | 1061 | ✓ | en |
| Stage2 | stage1_Qwen7BInst_stage2 | 1061 | ✓ | en |
| Stage1Reasoning | Merge_7B_instruct_Reasoning_Direct | 1061 | ✓ | en |
| Stage1ReasoningCTL | Merge_7B_instruct_Reasoning_Direct | 1061 | ✓ | en |
| 7BOriCTL | Qwen2-VL-7B-Instruct | 1061 | ✓ | en |
| 7BOri | Qwen2-VL-7B-Instruct | 1061 | ✓ | en |
| Qwen25VL32BInst | Qwen2.5-VL-32B-Instruct | 1061 | ✓ | en |

## Overall Results
*Sample counts: Stage1CTL: 1061 samples, Stage1: 1061 samples, Stage2: 1061 samples, Stage1Reasoning: 1061 samples, Stage1ReasoningCTL: 1061 samples, 7BOriCTL: 1061 samples, 7BOri: 1061 samples, Qwen25VL32BInst: 1061 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

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
| Perplexity | Mean Perplexity | 278.8062 | 278.8062 | **173.8327** | 598.1220 | 598.1220 | 240741.5970 | 240741.5970 | 694010380428.6918 |
| Perplexity | Median Perplexity | 3.8906 | 3.8906 | **2.8906** | 3.7969 | 3.7969 | 216.0000 | 216.0000 | 13376.0000 |
| Perplexity | Std Perplexity | 2424.2811 | 2424.2811 | **1582.0417** | 6534.2893 | 6534.2893 | 1523977.9827 | 1523977.9827 | 4472871553015.6406 |
| Perplexity | Min Perplexity | 1.0391 | 1.0391 | 1.0156 | 1.0156 | 1.0156 | **1.0000** | 1.0000 | 1.0000 |
| Perplexity | Max Perplexity | 52736.0000 | 52736.0000 | **32000.0000** | 119296.0000 | 119296.0000 | 21364736.0000 | 21364736.0000 | 61572651155456.0000 |

## Open-ended Questions
*Sample counts: Stage1CTL: 706 samples, Stage1: 706 samples, Stage2: 706 samples, Stage1Reasoning: 706 samples, Stage1ReasoningCTL: 706 samples, 7BOriCTL: 706 samples, 7BOri: 706 samples, Qwen25VL32BInst: 706 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

| Category | Metric | Stage1CTL | Stage1 | Stage2 | Stage1Reasoning | Stage1ReasoningCTL | 7BOriCTL | 7BOri | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | **0.3371** | 0.3371 | 0.3329 | 0.0609 | 0.2266 | 0.1601 | 0.0000 | 0.0000 |
| Text Matching | Soft Match | 0.4866 | 0.4872 | **0.4934** | 0.3782 | 0.4627 | 0.2859 | 0.1123 | 0.0029 |
| Text Matching | ROUGE-L | 0.4007 | 0.4019 | 0.3934 | 0.3520 | **0.4120** | 0.2351 | 0.0844 | 0.0167 |
| Text Matching | BLEU-4 | 0.0682 | **0.0685** | 0.0681 | 0.0491 | 0.0664 | 0.0355 | 0.0076 | 0.0009 |
| Text Matching | Word Overlap | 0.3734 | **0.3737** | 0.3677 | 0.2771 | 0.3616 | 0.2055 | 0.0531 | 0.0133 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Word Contain | 0.4986 | 0.5000 | 0.4547 | 0.5297 | 0.5326 | 0.5368 | 0.5467 | **0.7394** |
| Fine-grained | Char F1 | 0.5964 | **0.5971** | 0.5626 | 0.4896 | 0.5609 | 0.5315 | 0.4205 | 0.3947 |
| Fine-grained | Word F1 | 0.3902 | 0.3908 | 0.3824 | 0.3319 | **0.3969** | 0.2381 | 0.0930 | 0.0259 |
| Fine-grained | Char Precision | 0.5957 | **0.5965** | 0.5812 | 0.4360 | 0.5317 | 0.4598 | 0.3234 | 0.2873 |
| Fine-grained | Char Recall | 0.6512 | 0.6549 | 0.5825 | 0.6708 | 0.6685 | 0.7501 | 0.7706 | **0.8794** |
| Fine-grained | Word Precision | 0.3883 | **0.3893** | 0.3889 | 0.2942 | 0.3802 | 0.2076 | 0.0546 | 0.0133 |
| Fine-grained | Word Recall | 0.4312 | 0.4310 | 0.3978 | 0.4472 | 0.4621 | 0.4475 | 0.4607 | **0.6637** |

## Closed-set Questions
*Sample counts: Stage1CTL: 355 samples, Stage1: 355 samples, Stage2: 355 samples, Stage1Reasoning: 355 samples, Stage1ReasoningCTL: 355 samples, 7BOriCTL: 355 samples, 7BOri: 355 samples, Qwen25VL32BInst: 355 samples*

*Note: Higher is better for all metrics except perplexity (lower is better)*

| Category | Metric | Stage1CTL | Stage1 | Stage2 | Stage1Reasoning | Stage1ReasoningCTL | 7BOriCTL | 7BOri | Qwen25VL32BInst |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Text Matching | Exact Match | 0.7662 | **0.7690** | 0.7493 | 0.0141 | 0.2704 | 0.4901 | 0.0000 | 0.0000 |
| Text Matching | Soft Match | 0.7722 | **0.7751** | 0.7495 | 0.1913 | 0.4882 | 0.5164 | 0.0634 | 0.0053 |
| Text Matching | ROUGE-L | 0.7775 | **0.7803** | 0.7493 | 0.2563 | 0.5622 | 0.5281 | 0.1002 | 0.0134 |
| Text Matching | BLEU-4 | 0.1382 | **0.1387** | 0.1332 | 0.0315 | 0.0880 | 0.0910 | 0.0093 | 0.0010 |
| Text Matching | Word Overlap | 0.7746 | **0.7775** | 0.7493 | 0.2004 | 0.5330 | 0.5153 | 0.0631 | 0.0109 |
| Text Matching | BERTScore F1 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Fine-grained | Word Contain | 0.7831 | **0.7859** | 0.7493 | 0.6648 | 0.7746 | 0.7408 | 0.6986 | 0.6620 |
| Fine-grained | Char F1 | 0.7729 | **0.7757** | 0.7481 | 0.2964 | 0.5434 | 0.5765 | 0.2173 | 0.1822 |
| Fine-grained | Word F1 | 0.7775 | **0.7803** | 0.7493 | 0.2627 | 0.5654 | 0.5314 | 0.1121 | 0.0214 |
| Fine-grained | Char Precision | 0.7704 | **0.7732** | 0.7480 | 0.2030 | 0.4764 | 0.5424 | 0.1283 | 0.1052 |
| Fine-grained | Char Recall | 0.7845 | 0.7873 | 0.7493 | 0.8202 | 0.8103 | 0.7915 | **0.8582** | 0.8202 |
| Fine-grained | Word Precision | 0.7746 | **0.7775** | 0.7493 | 0.2004 | 0.5330 | 0.5153 | 0.0631 | 0.0109 |
| Fine-grained | Word Recall | 0.7831 | **0.7859** | 0.7493 | 0.6648 | 0.7746 | 0.7408 | 0.6986 | 0.6620 |

## Performance Summary

### Overall Results - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Stage1 | 9 | 50.0% |
| Stage2 | 4 | 22.2% |
| Qwen25VL32BInst | 3 | 16.7% |
| Stage1CTL | 1 | 5.6% |
| 7BOriCTL | 1 | 5.6% |

### Open-ended Questions - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Stage1 | 5 | 38.5% |
| Qwen25VL32BInst | 3 | 23.1% |
| Stage1CTL | 2 | 15.4% |
| Stage1ReasoningCTL | 2 | 15.4% |
| Stage2 | 1 | 7.7% |

### Closed-set Questions - Best Performance Count

| Model | Total Best Count | Percentage |
|-------|------------------|------------|
| Stage1 | 11 | 84.6% |
| Stage1CTL | 1 | 7.7% |
| 7BOri | 1 | 7.7% |

## Overall Analysis

**Best Overall Results Model:** Stage1 with 9 best metrics out of 18 (50.0%)

**Best Open-ended Questions Model:** Stage1 with 5 best metrics out of 13 (38.5%)

**Best Closed-set Questions Model:** Stage1 with 11 best metrics out of 13 (84.6%)

**Model Strengths Across Different Question Types:**
- **Stage1CTL**: Best in Overall Results (1 metrics), Open-ended Questions (2 metrics), Closed-set Questions (1 metrics)
- **Stage1**: Best in Overall Results (9 metrics), Open-ended Questions (5 metrics), Closed-set Questions (11 metrics)
- **Stage2**: Best in Overall Results (4 metrics), Open-ended Questions (1 metrics)
- **Stage1Reasoning**: No significant advantages
- **Stage1ReasoningCTL**: Best in Open-ended Questions (2 metrics)
- **7BOriCTL**: Best in Overall Results (1 metrics)
- **7BOri**: Best in Closed-set Questions (1 metrics)
- **Qwen25VL32BInst**: Best in Overall Results (3 metrics), Open-ended Questions (3 metrics)

**Key Observations:**
- Stage1CTL: exact match is 2.3x better on closed-set questions (0.766 vs 0.337); ROUGE-L shows 94.0% difference between closed-set and open-ended
- Stage1: exact match is 2.3x better on closed-set questions (0.769 vs 0.337); ROUGE-L shows 94.1% difference between closed-set and open-ended
- Stage2: exact match is 2.3x better on closed-set questions (0.749 vs 0.333); ROUGE-L shows 90.5% difference between closed-set and open-ended
- Stage1Reasoning: exact match is 4.3x better on open-ended questions (0.061 vs 0.014)
- 7BOriCTL: exact match is 3.1x better on closed-set questions (0.490 vs 0.160); ROUGE-L shows 124.6% difference between closed-set and open-ended
- 7BOri: exact match is infx better on closed-set questions (0.000 vs 0.000)
- Qwen25VL32BInst: exact match is infx better on closed-set questions (0.000 vs 0.000)

**Overall Performance Patterns:**