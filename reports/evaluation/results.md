# Model Evaluation Results

Time-based split at **2014-02-26**. 2,235 warm test users, 11,453 relevant test interactions (rating >= threshold), 12,078 rated test interactions for RMSE/MAE. Train recipes already reviewed by the user are excluded from the ranking.

## Ranking metrics @ 5

| model | precision@5 | recall@5 | ndcg@5 | hit_rate@5 | coverage@5 | diversity@5 | category_diversity@5 | novelty@5 |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.0070 | 0.0143 | 0.0110 | 0.0318 | 0.0006 | 0.9134 | 0.4001 | 4.4099 |
| item_knn | 0.0073 | 0.0127 | 0.0122 | 0.0322 | 0.0135 | 0.9039 | 0.4746 | 5.1641 |
| content | 0.0004 | 0.0013 | 0.0009 | 0.0022 | 0.0659 | 0.6271 | 0.4414 | 10.3295 |
| mf | 0.0002 | 0.0003 | 0.0004 | 0.0009 | 0.0047 | 0.9551 | 0.8014 | 11.0459 |
| hybrid | 0.0070 | 0.0148 | 0.0138 | 0.0327 | 0.0013 | 0.8955 | 0.3703 | 4.4995 |

## Ranking metrics @ 10

| model | precision@10 | recall@10 | ndcg@10 | hit_rate@10 | coverage@10 | diversity@10 | category_diversity@10 | novelty@10 |
|---|---|---|---|---|---|---|---|---|
| popularity | 0.0068 | 0.0246 | 0.0148 | 0.0591 | 0.0011 | 0.9131 | 0.2983 | 4.5614 |
| item_knn | 0.0063 | 0.0231 | 0.0152 | 0.0559 | 0.0233 | 0.9076 | 0.3684 | 5.3157 |
| content | 0.0006 | 0.0020 | 0.0013 | 0.0058 | 0.1028 | 0.6494 | 0.2886 | 10.3435 |
| mf | 0.0001 | 0.0003 | 0.0004 | 0.0009 | 0.0073 | 0.9534 | 0.6094 | 11.0473 |
| hybrid | 0.0066 | 0.0214 | 0.0164 | 0.0564 | 0.0023 | 0.9023 | 0.2940 | 4.6477 |

## Rating prediction

| model | rmse | mae |
|---|---|---|
| popularity | 0.6289 | 0.4200 |
| mf | 0.5870 | 0.3583 |
| global_mean (reference) | 0.6392 | 0.4368 |

Item-kNN, content and hybrid produce ranking scores only, so RMSE/MAE are not applicable.

## By user activity (train-period review count)

| model | segment | users | ndcg_at_10 | recall_at_10 |
|---|---|---|---|---|
| popularity | 5-9 reviews | 327 | 0.0092 | 0.0186 |
| popularity | 10-49 reviews | 1110 | 0.0162 | 0.0315 |
| popularity | 50+ reviews | 798 | 0.0152 | 0.0173 |
| item_knn | 5-9 reviews | 327 | 0.0089 | 0.0189 |
| item_knn | 10-49 reviews | 1110 | 0.0189 | 0.0332 |
| item_knn | 50+ reviews | 798 | 0.0128 | 0.0106 |
| content | 5-9 reviews | 327 | 0.0044 | 0.0061 |
| content | 10-49 reviews | 1110 | 0.0011 | 0.0022 |
| content | 50+ reviews | 798 | 0.0003 | 0.0001 |
| mf | 5-9 reviews | 327 | 0.0000 | 0.0000 |
| mf | 10-49 reviews | 1110 | 0.0008 | 0.0006 |
| mf | 50+ reviews | 798 | 0.0000 | 0.0000 |
| hybrid | 5-9 reviews | 327 | 0.0105 | 0.0164 |
| hybrid | 10-49 reviews | 1110 | 0.0198 | 0.0285 |
| hybrid | 50+ reviews | 798 | 0.0140 | 0.0136 |

## Training time

| model | fit_seconds |
|---|---|
| popularity | 0.0 |
| item_knn | 9.7 |
| content | 0.2 |
| mf | 4.6 |
| hybrid | 9.7 |
