import numpy as np
import pytest
import scipy.sparse as sp

from src.evaluation import metrics as M

RECS = np.array([[0, 1, 2], [3, 4, 5]])
REL = [{0, 2, 9}, {7}]


def test_precision_recall_hit():
    assert M.precision_at_k(RECS, REL, 3) == pytest.approx((2 / 3 + 0) / 2)
    assert M.recall_at_k(RECS, REL, 3) == pytest.approx((2 / 3 + 0) / 2)
    assert M.hit_rate_at_k(RECS, REL, 3) == pytest.approx(0.5)


def test_ndcg():
    d = 1 / np.log2(np.arange(2, 5))
    expected_user0 = (d[0] + d[2]) / d[:3].sum()
    assert M.ndcg_at_k(RECS, REL, 3) == pytest.approx((expected_user0 + 0) / 2)
    assert M.ndcg_at_k(np.array([[1, 2]]), [{1, 2}], 2) == pytest.approx(1.0)


def test_coverage_and_diversity():
    assert M.catalog_coverage(RECS, 12, 3) == pytest.approx(0.5)
    V = sp.csr_matrix(np.eye(6))
    assert M.intra_list_diversity(RECS, V, 3) == pytest.approx(1.0)       # orthogonal items
    V_same = sp.csr_matrix(np.ones((6, 1)))
    assert M.intra_list_diversity(RECS, V_same, 3) == pytest.approx(0.0)  # identical items
    cats = np.array(["a", "a", "b", "c", "c", "c"])
    assert M.category_diversity(RECS, cats, 3) == pytest.approx((2 / 3 + 1 / 3) / 2)


def test_rating_errors():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_per_user_scores_match_aggregates():
    per = M.per_user_scores(RECS[:1], REL[:1], 3)
    assert per["precision_at_3"].mean() == pytest.approx(M.precision_at_k(RECS[:1], REL[:1], 3))
    assert per["ndcg_at_3"].mean() == pytest.approx(M.ndcg_at_k(RECS[:1], REL[:1], 3))


def test_bootstrap_ci_and_paired_test():
    from src.evaluation.stats import bootstrap_ci, paired_bootstrap
    rng = np.random.default_rng(0)
    x = rng.binomial(1, 0.3, 2000).astype(float)
    lo, hi = bootstrap_ci(x, 500)
    assert lo < x.mean() < hi and hi - lo < 0.06
    better = np.clip(x + rng.binomial(1, 0.2, 2000), 0, 1)
    res = paired_bootstrap(better, x, 500)
    assert res["diff"] > 0 and res["diff_ci_low"] > 0 and res["p_value"] < 0.01
    same = paired_bootstrap(x, x, 500)
    assert same["diff"] == 0 and same["p_value"] == 1.0
