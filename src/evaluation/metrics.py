"""Ranking and rating metrics (binary relevance).

All ranking metrics take `recs` (list of top-K item indices per user, best first) and `relevant`
(set of relevant item indices per user) and return the mean over users.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp


def precision_at_k(recs: np.ndarray, relevant: Sequence[set], k: int) -> float:
    return float(np.mean([len(set(r[:k]) & rel) / k for r, rel in zip(recs, relevant)]))


def recall_at_k(recs: np.ndarray, relevant: Sequence[set], k: int) -> float:
    return float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in zip(recs, relevant) if rel]))


def hit_rate_at_k(recs: np.ndarray, relevant: Sequence[set], k: int) -> float:
    return float(np.mean([bool(set(r[:k]) & rel) for r, rel in zip(recs, relevant)]))


def ndcg_at_k(recs: np.ndarray, relevant: Sequence[set], k: int) -> float:
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for r, rel in zip(recs, relevant):
        if not rel:
            continue
        gains = np.array([1.0 if i in rel else 0.0 for i in r[:k]])
        dcg = float(gains @ discounts[: len(gains)])
        idcg = float(discounts[: min(len(rel), k)].sum())
        scores.append(dcg / idcg)
    return float(np.mean(scores))


def catalog_coverage(recs: np.ndarray, n_items: int, k: int) -> float:
    """Share of the train catalog that appears in at least one top-K list."""
    return float(len(np.unique(np.asarray(recs)[:, :k])) / n_items)


def intra_list_diversity(recs: np.ndarray, item_vectors: sp.csr_matrix, k: int) -> float:
    """Mean over users of 1 - average pairwise cosine similarity inside the top-K list.

    item_vectors must be L2-normalised (rows = items)."""
    vals = []
    for r in np.asarray(recs)[:, :k]:
        V = item_vectors[r]
        G = (V @ V.T).toarray()
        n = len(r)
        off = (G.sum() - np.trace(G)) / (n * (n - 1))
        vals.append(1.0 - off)
    return float(np.mean(vals))


def category_diversity(recs: np.ndarray, item_category: np.ndarray, k: int) -> float:
    """Mean share of distinct categories in the top-K list (1/k = all same, 1 = all different)."""
    return float(np.mean([len(set(item_category[r[:k]])) / k for r in np.asarray(recs)]))


def novelty(recs: np.ndarray, item_popularity: np.ndarray, n_users: int, k: int) -> float:
    """Mean self-information -log2(p(i)) of recommended items (higher = less mainstream)."""
    p = np.clip(item_popularity / n_users, 1e-12, 1)
    return float(np.mean(-np.log2(p[np.asarray(recs)[:, :k]])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
