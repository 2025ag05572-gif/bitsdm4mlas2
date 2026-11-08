# Model Evaluation Results

Time-based split at **2014-02-26**. 2,235 warm test users, 11,453 relevant test interactions (rating >= threshold), 12,078 rated test interactions for RMSE/MAE. Train recipes already reviewed by the user are excluded from the ranking.

## Ranking metrics @ 5

| model | precision@5 | recall@5 | ndcg@5 | hit_rate@5 | coverage@5 | diversity@5 | category_diversity@5 | novelty@5 |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.0070 | 0.0143 | 0.0110 | 0.0318 | 0.0006 | 0.9134 | 0.4001 | 4.4099 |
| item_knn | 0.0072 | 0.0146 | 0.0130 | 0.0322 | 0.0031 | 0.9008 | 0.3613 | 4.5383 |
| content | 0.0004 | 0.0013 | 0.0009 | 0.0022 | 0.0659 | 0.6271 | 0.4414 | 10.3295 |
| mf | 0.0002 | 0.0003 | 0.0003 | 0.0009 | 0.0025 | 0.9606 | 0.8090 | 11.1075 |
| hybrid | 0.0074 | 0.0149 | 0.0120 | 0.0340 | 0.0007 | 0.9046 | 0.3593 | 4.3881 |

## Ranking metrics @ 10

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | category_diversity@10 | novelty@10 |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.0068 | 0.0246 | 0.0148 | 0.0591 | 0.0011 | 0.9131 | 0.2983 | 4.5614 |
| item_knn | 0.0069 | 0.0238 | 0.0164 | 0.0609 | 0.0062 | 0.9052 | 0.2864 | 4.7009 |
| content | 0.0006 | 0.0020 | 0.0013 | 0.0058 | 0.1028 | 0.6494 | 0.2886 | 10.3435 |
| mf | 0.0001 | 0.0003 | 0.0003 | 0.0013 | 0.0041 | 0.9573 | 0.6634 | 11.1012 |
| hybrid | 0.0074 | 0.0251 | 0.0160 | 0.0640 | 0.0012 | 0.9063 | 0.2611 | 4.5749 |

## Rating prediction

| model | rmse | mae |
|---|---|---|
| popularity | 0.6289 | 0.4200 |
| mf | 0.5861 | 0.3599 |
| global_mean (reference) | 0.6392 | 0.4368 |

Item-kNN, content and hybrid produce ranking scores only, so RMSE/MAE are not applicable.

## Uncertainty and significance

Bootstrap over evaluation users (1000 resamples). The comparison with the popularity baseline is paired (each resample uses the same users for both models); p is a two-sided bootstrap p-value.

| model | metric | mean | 95% CI | diff vs popularity [95% CI] | p |
|---|---|---|---|---|---|
| popularity | precision_at_10 | 0.0068 | [0.0055, 0.0083] | - | - |
| popularity | recall_at_10 | 0.0246 | [0.0195, 0.0301] | - | - |
| popularity | ndcg_at_10 | 0.0148 | [0.0118, 0.0181] | - | - |
| popularity | hit_rate_at_10 | 0.0591 | [0.0492, 0.0689] | - | - |
| item_knn | precision_at_10 | 0.0069 | [0.0056, 0.0085] | +0.0001 [-0.0006, +0.0009] | 0.692 |
| item_knn | recall_at_10 | 0.0238 | [0.0188, 0.0295] | -0.0007 [-0.0043, +0.0030] | 0.740 |
| item_knn | ndcg_at_10 | 0.0164 | [0.0130, 0.0202] | +0.0016 [-0.0005, +0.0038] | 0.110 |
| item_knn | hit_rate_at_10 | 0.0609 | [0.0510, 0.0712] | +0.0018 [-0.0049, +0.0089] | 0.580 |
| content | precision_at_10 | 0.0006 | [0.0003, 0.0009] | -0.0062 [-0.0077, -0.0049] | <0.001 |
| content | recall_at_10 | 0.0020 | [0.0007, 0.0037] | -0.0225 [-0.0285, -0.0173] | <0.001 |
| content | ndcg_at_10 | 0.0013 | [0.0004, 0.0024] | -0.0135 [-0.0169, -0.0103] | <0.001 |
| content | hit_rate_at_10 | 0.0058 | [0.0031, 0.0089] | -0.0532 [-0.0635, -0.0434] | <0.001 |
| mf | precision_at_10 | 0.0001 | [0.0000, 0.0003] | -0.0067 [-0.0081, -0.0054] | <0.001 |
| mf | recall_at_10 | 0.0003 | [0.0000, 0.0008] | -0.0243 [-0.0297, -0.0192] | <0.001 |
| mf | ndcg_at_10 | 0.0003 | [0.0000, 0.0007] | -0.0146 [-0.0178, -0.0116] | <0.001 |
| mf | hit_rate_at_10 | 0.0013 | [0.0000, 0.0031] | -0.0577 [-0.0676, -0.0479] | <0.001 |
| hybrid | precision_at_10 | 0.0074 | [0.0059, 0.0088] | +0.0006 [+0.0001, +0.0011] | 0.022 |
| hybrid | recall_at_10 | 0.0251 | [0.0200, 0.0306] | +0.0006 [-0.0014, +0.0024] | 0.592 |
| hybrid | ndcg_at_10 | 0.0160 | [0.0129, 0.0193] | +0.0011 [+0.0001, +0.0023] | 0.040 |
| hybrid | hit_rate_at_10 | 0.0640 | [0.0532, 0.0738] | +0.0049 [+0.0004, +0.0098] | 0.040 |

## By user activity (train-period review count)

| model | segment | users | ndcg_at_10 | recall_at_10 |
|---|---|---|---|---|
| popularity | 5-9 reviews | 327 | 0.0092 | 0.0186 |
| popularity | 10-49 reviews | 1110 | 0.0162 | 0.0315 |
| popularity | 50+ reviews | 798 | 0.0152 | 0.0173 |
| item_knn | 5-9 reviews | 327 | 0.0110 | 0.0172 |
| item_knn | 10-49 reviews | 1110 | 0.0197 | 0.0329 |
| item_knn | 50+ reviews | 798 | 0.0141 | 0.0139 |
| content | 5-9 reviews | 327 | 0.0044 | 0.0061 |
| content | 10-49 reviews | 1110 | 0.0011 | 0.0022 |
| content | 50+ reviews | 798 | 0.0003 | 0.0001 |
| mf | 5-9 reviews | 327 | 0.0000 | 0.0000 |
| mf | 10-49 reviews | 1110 | 0.0005 | 0.0006 |
| mf | 50+ reviews | 798 | 0.0001 | 0.0000 |
| hybrid | 5-9 reviews | 327 | 0.0086 | 0.0154 |
| hybrid | 10-49 reviews | 1110 | 0.0183 | 0.0328 |
| hybrid | 50+ reviews | 798 | 0.0158 | 0.0184 |

## Training time

| model | fit_seconds |
|---|---|
| popularity | 0.0 |
| item_knn | 9.2 |
| content | 0.2 |
| mf | 4.8 |
| hybrid | 9.2 |
