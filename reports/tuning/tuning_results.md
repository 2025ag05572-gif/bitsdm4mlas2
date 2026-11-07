# Hyper-parameter tuning (validation window inside the train period)

Inner cutoff 2011-08-02 (latest 10% of train interactions), 4,265 validation users. Ranking models are selected on `val_ndcg_at_10`, MF on `val_rmse`. The test set is not used here.

## popularity

| params | val_ndcg_at_10 | val_recall_at_10 | val_precision_at_10 | val_coverage_at_10 |
|---|---|---|---|---|
| score=count | 0.0194 | 0.0268 | 0.0099 | 0.0012 |
| score=recent_count | 0.0188 | 0.0259 | 0.0097 | 0.0012 |
| score=bayesian_rating | 0.0024 | 0.0027 | 0.0015 | 0.0004 |

**Selected:** `{'score': 'count'}`

## item_knn

| params | val_ndcg_at_10 | val_recall_at_10 | val_precision_at_10 | val_coverage_at_10 |
|---|---|---|---|---|
| n_neighbors=500, user_weighting=log, shrinkage=10, history_window=None | 0.0173 | 0.0224 | 0.0092 | 0.0431 |
| n_neighbors=500, user_weighting=log, shrinkage=10, history_window=100 | 0.0172 | 0.0224 | 0.0091 | 0.0438 |
| n_neighbors=200, user_weighting=log, shrinkage=10, history_window=None | 0.0169 | 0.0222 | 0.0088 | 0.0545 |
| n_neighbors=200, user_weighting=log, shrinkage=10, history_window=100 | 0.0169 | 0.0221 | 0.0089 | 0.0555 |
| n_neighbors=500, user_weighting=log, shrinkage=10, history_window=25 | 0.0163 | 0.0218 | 0.0084 | 0.0477 |
| n_neighbors=50, user_weighting=log, shrinkage=10, history_window=100 | 0.0161 | 0.0213 | 0.0088 | 0.0753 |
| n_neighbors=50, user_weighting=log, shrinkage=10, history_window=None | 0.0161 | 0.0214 | 0.0088 | 0.0740 |
| n_neighbors=200, user_weighting=log, shrinkage=10, history_window=25 | 0.0156 | 0.0217 | 0.0083 | 0.0611 |
| n_neighbors=50, user_weighting=log, shrinkage=10, history_window=25 | 0.0153 | 0.0210 | 0.0086 | 0.0833 |
| n_neighbors=50, user_weighting=none, shrinkage=10, history_window=None | 0.0111 | 0.0148 | 0.0063 | 0.2698 |
| n_neighbors=500, user_weighting=none, shrinkage=10, history_window=None | 0.0109 | 0.0138 | 0.0063 | 0.2106 |
| n_neighbors=50, user_weighting=none, shrinkage=10, history_window=100 | 0.0107 | 0.0145 | 0.0061 | 0.2730 |
| n_neighbors=500, user_weighting=none, shrinkage=10, history_window=100 | 0.0105 | 0.0133 | 0.0059 | 0.2114 |
| n_neighbors=200, user_weighting=none, shrinkage=10, history_window=100 | 0.0102 | 0.0129 | 0.0057 | 0.2328 |
| n_neighbors=200, user_weighting=none, shrinkage=10, history_window=None | 0.0102 | 0.0133 | 0.0059 | 0.2307 |
| n_neighbors=500, user_weighting=none, shrinkage=10, history_window=25 | 0.0095 | 0.0125 | 0.0049 | 0.2299 |
| n_neighbors=50, user_weighting=none, shrinkage=10, history_window=25 | 0.0088 | 0.0128 | 0.0047 | 0.3012 |
| n_neighbors=200, user_weighting=none, shrinkage=10, history_window=25 | 0.0086 | 0.0120 | 0.0046 | 0.2554 |

**Selected:** `{'n_neighbors': 500, 'user_weighting': 'log', 'shrinkage': 10, 'history_window': None}`

## mf

| params | val_rmse | val_mae |
|---|---|---|
| n_factors=16, reg=0.1 | 0.5794 | 0.3682 |
| n_factors=64, reg=0.1 | 0.5796 | 0.3700 |
| n_factors=16, reg=0.02 | 0.5872 | 0.3704 |
| n_factors=64, reg=0.02 | 0.5905 | 0.3739 |

**Selected:** `{'n_factors': 16, 'reg': 0.1}`

## hybrid

| params | val_ndcg_at_10 | val_recall_at_10 | val_precision_at_10 | val_coverage_at_10 |
|---|---|---|---|---|
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.0, cold_user_threshold=10, popularity_score=count | 0.0199 | 0.0264 | 0.0102 | 0.0031 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.0, cold_user_threshold=10, popularity_score=count | 0.0197 | 0.0260 | 0.0101 | 0.0038 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.05, cold_user_threshold=10, popularity_score=count | 0.0196 | 0.0260 | 0.0101 | 0.0043 |
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.05, cold_user_threshold=10, popularity_score=count | 0.0195 | 0.0263 | 0.0102 | 0.0037 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.05, cold_user_threshold=10, popularity_score=count | 0.0191 | 0.0255 | 0.0100 | 0.0091 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.0, cold_user_threshold=10, popularity_score=count | 0.0190 | 0.0249 | 0.0100 | 0.0087 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.0, cold_user_threshold=10, popularity_score=count | 0.0185 | 0.0249 | 0.0100 | 0.0230 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.05, cold_user_threshold=10, popularity_score=count | 0.0184 | 0.0240 | 0.0101 | 0.0239 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.0, cold_user_threshold=10, popularity_score=count | 0.0183 | 0.0237 | 0.0097 | 0.0186 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.05, cold_user_threshold=10, popularity_score=count | 0.0181 | 0.0234 | 0.0096 | 0.0193 |

**Selected:** `{'w_knn': 0.3, 'w_content': 0.1, 'w_popularity': 0.6, 'category_boost': 0.0, 'cold_user_threshold': 10, 'popularity_score': 'count'}`
