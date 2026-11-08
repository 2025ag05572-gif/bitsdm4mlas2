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
| n_neighbors=300, user_weighting=log, shrinkage=100, history_window=None | 0.0198 | 0.0261 | 0.0099 | 0.0110 |
| n_neighbors=300, user_weighting=log, shrinkage=100, history_window=50 | 0.0196 | 0.0259 | 0.0098 | 0.0117 |
| n_neighbors=100, user_weighting=log, shrinkage=100, history_window=50 | 0.0194 | 0.0252 | 0.0098 | 0.0142 |
| n_neighbors=100, user_weighting=log, shrinkage=100, history_window=None | 0.0194 | 0.0251 | 0.0098 | 0.0133 |
| n_neighbors=600, user_weighting=log, shrinkage=100, history_window=None | 0.0194 | 0.0254 | 0.0098 | 0.0102 |
| n_neighbors=600, user_weighting=log, shrinkage=100, history_window=50 | 0.0193 | 0.0252 | 0.0098 | 0.0110 |
| n_neighbors=100, user_weighting=log, shrinkage=10, history_window=None | 0.0171 | 0.0233 | 0.0092 | 0.0620 |
| n_neighbors=300, user_weighting=log, shrinkage=10, history_window=None | 0.0171 | 0.0220 | 0.0093 | 0.0487 |
| n_neighbors=600, user_weighting=log, shrinkage=10, history_window=None | 0.0171 | 0.0220 | 0.0091 | 0.0418 |
| n_neighbors=600, user_weighting=log, shrinkage=10, history_window=50 | 0.0169 | 0.0219 | 0.0089 | 0.0437 |
| n_neighbors=600, user_weighting=none, shrinkage=100, history_window=None | 0.0169 | 0.0221 | 0.0095 | 0.0215 |
| n_neighbors=600, user_weighting=none, shrinkage=100, history_window=50 | 0.0169 | 0.0225 | 0.0094 | 0.0228 |
| n_neighbors=100, user_weighting=none, shrinkage=100, history_window=None | 0.0168 | 0.0223 | 0.0092 | 0.0287 |
| n_neighbors=300, user_weighting=none, shrinkage=100, history_window=None | 0.0168 | 0.0215 | 0.0093 | 0.0231 |
| n_neighbors=300, user_weighting=log, shrinkage=10, history_window=50 | 0.0167 | 0.0216 | 0.0090 | 0.0510 |
| n_neighbors=100, user_weighting=none, shrinkage=100, history_window=50 | 0.0166 | 0.0222 | 0.0092 | 0.0309 |
| n_neighbors=100, user_weighting=log, shrinkage=10, history_window=50 | 0.0166 | 0.0230 | 0.0088 | 0.0653 |
| n_neighbors=300, user_weighting=none, shrinkage=100, history_window=50 | 0.0165 | 0.0213 | 0.0089 | 0.0247 |
| n_neighbors=300, user_weighting=none, shrinkage=10, history_window=None | 0.0110 | 0.0137 | 0.0062 | 0.2208 |
| n_neighbors=600, user_weighting=none, shrinkage=10, history_window=None | 0.0109 | 0.0142 | 0.0062 | 0.2062 |
| n_neighbors=100, user_weighting=none, shrinkage=10, history_window=None | 0.0109 | 0.0141 | 0.0061 | 0.2494 |
| n_neighbors=600, user_weighting=none, shrinkage=10, history_window=50 | 0.0102 | 0.0135 | 0.0056 | 0.2115 |
| n_neighbors=100, user_weighting=none, shrinkage=10, history_window=50 | 0.0101 | 0.0139 | 0.0053 | 0.2593 |
| n_neighbors=300, user_weighting=none, shrinkage=10, history_window=50 | 0.0101 | 0.0128 | 0.0056 | 0.2272 |

**Selected:** `{'n_neighbors': 300, 'user_weighting': 'log', 'shrinkage': 100, 'history_window': None}`

## mf

| params | val_rmse | val_mae |
|---|---|---|
| n_factors=16, reg=0.2 | 0.5783 | 0.3698 |
| n_factors=64, reg=0.2 | 0.5785 | 0.3715 |
| n_factors=32, reg=0.2 | 0.5789 | 0.3706 |
| n_factors=16, reg=0.1 | 0.5794 | 0.3682 |
| n_factors=64, reg=0.1 | 0.5796 | 0.3700 |
| n_factors=32, reg=0.1 | 0.5800 | 0.3689 |
| n_factors=16, reg=0.05 | 0.5815 | 0.3682 |
| n_factors=64, reg=0.05 | 0.5826 | 0.3706 |
| n_factors=32, reg=0.05 | 0.5827 | 0.3692 |

**Selected:** `{'n_factors': 16, 'reg': 0.2}`

## hybrid

| params | val_ndcg_at_10 | val_recall_at_10 | val_precision_at_10 | val_coverage_at_10 |
|---|---|---|---|---|
| w_knn=0.2, w_content=0.1, w_popularity=0.7, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0201 | 0.0271 | 0.0103 | 0.0016 |
| w_knn=0.3, w_content=0.0, w_popularity=0.7, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0201 | 0.0277 | 0.0104 | 0.0015 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0264 | 0.0102 | 0.0051 |
| w_knn=0.2, w_content=0.1, w_popularity=0.7, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0263 | 0.0101 | 0.0018 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0260 | 0.0102 | 0.0096 |
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0265 | 0.0101 | 0.0030 |
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0264 | 0.0102 | 0.0028 |
| w_knn=0.3, w_content=0.0, w_popularity=0.7, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0200 | 0.0272 | 0.0102 | 0.0020 |
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0199 | 0.0266 | 0.0103 | 0.0017 |
| w_knn=0.2, w_content=0.1, w_popularity=0.7, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0199 | 0.0259 | 0.0102 | 0.0021 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0199 | 0.0265 | 0.0101 | 0.0031 |
| w_knn=0.5, w_content=0.0, w_popularity=0.5, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0199 | 0.0267 | 0.0101 | 0.0042 |
| w_knn=0.3, w_content=0.1, w_popularity=0.6, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0199 | 0.0265 | 0.0102 | 0.0019 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0198 | 0.0263 | 0.0100 | 0.0053 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0198 | 0.0254 | 0.0102 | 0.0054 |
| w_knn=0.5, w_content=0.0, w_popularity=0.5, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0198 | 0.0265 | 0.0100 | 0.0042 |
| w_knn=0.5, w_content=0.1, w_popularity=0.4, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0198 | 0.0257 | 0.0101 | 0.0031 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0197 | 0.0263 | 0.0101 | 0.0033 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0197 | 0.0265 | 0.0100 | 0.0049 |
| w_knn=0.2, w_content=0.1, w_popularity=0.7, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0197 | 0.0264 | 0.0102 | 0.0016 |
| w_knn=0.3, w_content=0.0, w_popularity=0.7, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0197 | 0.0267 | 0.0103 | 0.0017 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0197 | 0.0264 | 0.0102 | 0.0033 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0197 | 0.0266 | 0.0100 | 0.0084 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.0, cold_user_threshold=5, popularity_score=count | 0.0197 | 0.0260 | 0.0099 | 0.0081 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0196 | 0.0253 | 0.0098 | 0.0058 |
| w_knn=0.5, w_content=0.0, w_popularity=0.5, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0196 | 0.0264 | 0.0102 | 0.0026 |
| w_knn=0.6, w_content=0.25, w_popularity=0.15, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0196 | 0.0253 | 0.0098 | 0.0098 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.05, cold_user_threshold=20, popularity_score=count | 0.0196 | 0.0265 | 0.0102 | 0.0021 |
| w_knn=0.3, w_content=0.0, w_popularity=0.7, category_boost=0.05, cold_user_threshold=5, popularity_score=count | 0.0196 | 0.0263 | 0.0101 | 0.0023 |
| w_knn=0.4, w_content=0.0, w_popularity=0.6, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0195 | 0.0263 | 0.0101 | 0.0020 |
| w_knn=0.7, w_content=0.1, w_popularity=0.2, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0195 | 0.0251 | 0.0099 | 0.0047 |
| w_knn=0.5, w_content=0.0, w_popularity=0.5, category_boost=0.0, cold_user_threshold=20, popularity_score=count | 0.0194 | 0.0259 | 0.0099 | 0.0025 |

**Selected:** `{'w_knn': 0.2, 'w_content': 0.1, 'w_popularity': 0.7, 'category_boost': 0.0, 'cold_user_threshold': 20, 'popularity_score': 'count'}`
