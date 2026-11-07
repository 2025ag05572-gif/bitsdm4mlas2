"""Recommender models.

All models share one interface over the train interaction matrix (users x items):
    fit(data)            -> self
    score(uidx)          -> dense (len(uidx), n_items) relevance scores (higher = better)
    predict(uidx, iidx)  -> predicted star ratings (only if predicts_ratings is True)

Models
* PopularityRecommender  - baseline; ranks recipes by (all-time or recent) review count from the feature store
* ItemKNNRecommender     - item-based collaborative filtering (cosine with shrinkage, top-N neighbours)
* ContentRecommender     - TF-IDF profile of each user's reviewed recipes (tags + ingredients)
* BiasedMFRecommender    - matrix factorisation of star ratings (Funk-SVD style, mini-batch SGD)
* HybridRecommender      - item-kNN + content + popularity blend, gated by user activity features
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.features.store import FeatureStore


# --------------------------------------------------------------------------- data container
@dataclass
class TrainData:
    R: sp.csr_matrix            # implicit (1 = reviewed) users x items
    ratings: sp.csr_matrix      # explicit stars (only rated reviews stored)
    user_ids: np.ndarray        # uidx -> user_id
    item_ids: np.ndarray        # iidx -> recipe_id
    fs: FeatureStore            # training feature view
    dates: sp.csr_matrix | None = None   # day number of each train review (recency features)

    @property
    def n_users(self) -> int:
        return self.R.shape[0]

    @property
    def n_items(self) -> int:
        return self.R.shape[1]


def load_train_data(cfg: dict) -> TrainData:
    pdir = Path(cfg["paths"]["processed_dir"])
    return TrainData(
        R=sp.load_npz(pdir / "interaction_matrix.npz").tocsr(),
        ratings=sp.load_npz(pdir / "rating_matrix.npz").tocsr(),
        user_ids=pd.read_parquet(pdir / "user_index.parquet")["user_id"].values,
        item_ids=pd.read_parquet(pdir / "item_index.parquet")["recipe_id"].values,
        fs=FeatureStore(cfg, view="training"),
        dates=sp.load_npz(pdir / "interaction_dates.npz").tocsr() if (pdir / "interaction_dates.npz").exists() else None,
    )


def most_recent(R: sp.csr_matrix, dates: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """Keep only each user's n most recent interactions (R and dates share the same sparsity pattern)."""
    R, dates = R.tocsr(), dates.tocsr()
    R.sort_indices(); dates.sort_indices()
    keep = np.ones(R.nnz, dtype=bool)
    for u in range(R.shape[0]):
        a, b = R.indptr[u], R.indptr[u + 1]
        if b - a > n:
            d = dates.data[a:b]
            drop = np.argsort(d, kind="stable")[: (b - a) - n]
            keep[a + drop] = False
    out = R.copy()
    out.data = np.where(keep, out.data, 0).astype(out.dtype)
    out.eliminate_zeros()
    return out


# numpy 2.x linked against Apple Accelerate raises spurious divide/overflow/invalid warnings
# from float32 matmul on macOS even when results are finite (checked against float64), so they
# are silenced for scoring only.
def _quiet():
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


def _row_max_normalize(S: np.ndarray) -> np.ndarray:
    m = S.max(axis=1, keepdims=True)
    m[m <= 0] = 1.0
    return S / m


class Recommender:
    name = "base"
    predicts_ratings = False

    def __init__(self, **params) -> None:
        self.params = params
        self.fit_seconds = None

    def fit(self, data: TrainData, **kwargs) -> "Recommender":
        t0 = time.time()
        self._fit(data, **kwargs)
        self.fit_seconds = time.time() - t0
        return self

    def _fit(self, data: TrainData) -> None:
        raise NotImplementedError

    def score(self, uidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, uidx: np.ndarray, iidx: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not predict ratings")


# --------------------------------------------------------------------------- baseline
class PopularityRecommender(Recommender):
    """Same ranked list for everyone. Uses features from the store, not the raw matrix."""

    name = "popularity"
    predicts_ratings = True

    def _fit(self, data: TrainData) -> None:
        f = data.fs.get_item_features(data.item_ids, ["item_popularity", "item_recent_popularity",
                                                       "item_bayesian_rating"])
        pop = f["item_popularity"].fillna(0).values.astype(np.float64)
        recent = f["item_recent_popularity"].fillna(0).values.astype(np.float64)
        bayes = f["item_bayesian_rating"].fillna(f["item_bayesian_rating"].mean()).values
        mode = self.params.get("score", "count")
        if mode == "bayesian_rating":
            self.scores_ = bayes + 1e-6 * pop
        elif mode == "recent_count":       # reviews in the last features.recent_window_days
            self.scores_ = recent + 1e-6 * pop
        else:
            self.scores_ = pop + 1e-3 * bayes  # tie-break by rating
        self.scores_ = self.scores_.astype(np.float32)
        self.rating_ = bayes.astype(np.float32)

    def score(self, uidx):
        return np.tile(self.scores_, (len(uidx), 1))

    def predict(self, uidx, iidx):
        return self.rating_[iidx]


# --------------------------------------------------------------------------- item-kNN CF
class ItemKNNRecommender(Recommender):
    """Item-based CF.

    params: n_neighbors, shrinkage,
            user_weighting  none | log   - down-weight very active reviewers in the similarity
            history_window  int | None   - score with only the user's N most recent reviews
    """

    name = "item_knn"

    def _fit(self, data: TrainData) -> None:
        k = int(self.params.get("n_neighbors", 200))
        shrink = float(self.params.get("shrinkage", 10))
        X = data.R.tocsr().astype(np.float32)
        if self.params.get("user_weighting", "none") == "log":
            n_u = np.asarray(X.sum(axis=1)).ravel()
            X = sp.diags(np.sqrt(1.0 / np.log2(1.0 + n_u)).astype(np.float32)) @ X
        X = X.tocsc()
        n_items = X.shape[1]
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel()).astype(np.float32)
        XT = X.T.tocsr()
        rows, cols, vals = [], [], []
        block = 1000
        for start in range(0, n_items, block):
            stop = min(start + block, n_items)
            co = (XT[start:stop] @ X).toarray()
            sim = co / (norms[start:stop, None] * norms[None, :] + shrink)
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0
            kk = min(k, n_items - 1)
            top = np.argpartition(-sim, kk, axis=1)[:, :kk] if kk < n_items else np.argsort(-sim, axis=1)
            tv = np.take_along_axis(sim, top, axis=1)
            keep = tv > 0
            r = np.repeat(np.arange(start, stop), kk).reshape(-1, kk)
            rows.append(r[keep]); cols.append(top[keep]); vals.append(tv[keep])
        self.S_ = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items)
        )
        self.set_history_window(data, self.params.get("history_window"))

    def set_history_window(self, data: TrainData, n) -> None:
        """Which part of each user's history is used for scoring (cheap - no re-fit of S)."""
        self.params["history_window"] = n
        if n and data.dates is not None:
            self.R_ = most_recent(data.R, data.dates, int(n))
        else:
            self.R_ = data.R

    def score(self, uidx):
        return (self.R_[uidx] @ self.S_).toarray()


# --------------------------------------------------------------------------- content-based
class ContentRecommender(Recommender):
    name = "content"

    def _fit(self, data: TrainData) -> None:
        self.V_ = normalize(data.fs.get_item_content(data.item_ids))   # items x vocab
        # weight reviewed items by stars (unrated reviews count as 3 = neutral-positive signal)
        rated_mask = data.ratings.copy()
        rated_mask.data[:] = 1.0
        W = (3.0 * data.R - 3.0 * rated_mask + data.ratings).tocsr()
        W.eliminate_zeros()
        self.P_ = normalize(W @ self.V_)                                 # users x vocab

    def score(self, uidx):
        with _quiet():
            return (self.P_[uidx] @ self.V_.T).toarray()


# --------------------------------------------------------------------------- matrix factorisation
class BiasedMFRecommender(Recommender):
    """r_ui ~ mu + b_u + b_i + p_u . q_i, trained with mini-batch SGD on observed star ratings."""

    name = "mf"
    predicts_ratings = True

    def _fit(self, data: TrainData) -> None:
        p = self.params
        f, epochs = int(p.get("n_factors", 32)), int(p.get("n_epochs", 25))
        lr, reg, bs = float(p.get("learning_rate", 0.01)), float(p.get("reg", 0.05)), int(p.get("batch_size", 1024))
        rng = np.random.default_rng(int(p.get("seed", 42)))
        coo = data.ratings.tocoo()
        u, i, r = coo.row, coo.col, coo.data.astype(np.float64)
        self.mu_ = float(r.mean())
        self.bu_ = np.zeros(data.n_users); self.bi_ = np.zeros(data.n_items)
        self.P_ = rng.normal(0, 0.1, (data.n_users, f)); self.Q_ = rng.normal(0, 0.1, (data.n_items, f))
        self.history_ = []
        for _ in range(epochs):
            order = rng.permutation(len(r))
            sq = 0.0
            for s in range(0, len(r), bs):
                b = order[s:s + bs]
                ub, ib, rb = u[b], i[b], r[b]
                pu, qi = self.P_[ub], self.Q_[ib]
                e = rb - (self.mu_ + self.bu_[ub] + self.bi_[ib] + np.einsum("ij,ij->i", pu, qi))
                sq += float(np.sum(e * e))
                np.add.at(self.bu_, ub, lr * (e - reg * self.bu_[ub]))
                np.add.at(self.bi_, ib, lr * (e - reg * self.bi_[ib]))
                np.add.at(self.P_, ub, lr * (e[:, None] * qi - reg * pu))
                np.add.at(self.Q_, ib, lr * (e[:, None] * pu - reg * qi))
            self.history_.append(float(np.sqrt(sq / len(r))))
        self.P_ = self.P_.astype(np.float32); self.Q_ = self.Q_.astype(np.float32)

    def score(self, uidx):
        with _quiet():
            return (self.P_[uidx] @ self.Q_.T + self.bi_[None, :]).astype(np.float32)

    def predict(self, uidx, iidx):
        pred = self.mu_ + self.bu_[uidx] + self.bi_[iidx] + np.einsum("ij,ij->i", self.P_[uidx], self.Q_[iidx])
        return np.clip(pred, 1.0, 5.0)


# --------------------------------------------------------------------------- hybrid
class HybridRecommender(Recommender):
    """Weighted blend of item-kNN, content and popularity scores.

    Uses features from the store at training *and* prediction time:
      user_activity_count     -> users with little history lean more on popularity
      user_preferred_category -> small boost for recipes in the user's favourite category
      item_category           -> category of each candidate recipe
    """

    name = "hybrid"

    def __init__(self, knn_params: dict | None = None, **params) -> None:
        super().__init__(**params)
        self.knn_params = knn_params or {}

    def _fit(self, data: TrainData, components: dict | None = None) -> None:
        components = components or {}
        self.knn_ = components.get("knn") or ItemKNNRecommender(**self.knn_params).fit(data)
        self.content_ = components.get("content") or ContentRecommender().fit(data)
        self.pop_ = components.get("pop") or PopularityRecommender(score=self.params.get("popularity_score", "count")).fit(data)
        pop = np.log1p(self.pop_.scores_)
        self.pop_norm_ = (pop / pop.max()).astype(np.float32)
        uf = data.fs.get_user_features(data.user_ids, ["user_activity_count", "user_preferred_category"])
        self.activity_ = uf["user_activity_count"].fillna(0).values
        cats = data.fs.get_item_features(data.item_ids, ["item_category"])["item_category"].fillna("other")
        self.cat_codes_, self.cat_names_ = pd.factorize(cats)
        lookup = {c: k for k, c in enumerate(self.cat_names_)}
        self.user_cat_ = uf["user_preferred_category"].map(lookup).fillna(-1).astype(int).values

    def score(self, uidx):
        p = self.params
        alpha = np.clip(self.activity_[uidx] / float(p.get("cold_user_threshold", 10)), 0, 1)[:, None]
        wk, wc, wp = float(p.get("w_knn", 0.6)), float(p.get("w_content", 0.25)), float(p.get("w_popularity", 0.15))
        s = (wk * alpha) * _row_max_normalize(self.knn_.score(uidx))
        s += (wc * alpha) * _row_max_normalize(self.content_.score(uidx))
        s += (wp + (1 - alpha) * (wk + wc)) * self.pop_norm_[None, :]
        s += float(p.get("category_boost", 0.05)) * (self.cat_codes_[None, :] == self.user_cat_[uidx][:, None])
        return s.astype(np.float32)


MODELS = {
    "popularity": PopularityRecommender,
    "item_knn": ItemKNNRecommender,
    "content": ContentRecommender,
    "mf": BiasedMFRecommender,
    "hybrid": HybridRecommender,
}


def tuned_params(cfg: dict) -> dict:
    """Best parameters from the tuning stage (empty if disabled or not run yet)."""
    path = Path(cfg["paths"]["models_dir"]) / "best_params.json"
    if not cfg["training"].get("use_tuned_params") or not path.exists():
        return {}
    import json
    return json.loads(path.read_text())["params"]


def model_params(name: str, cfg: dict) -> dict:
    p = {k: v for k, v in cfg["training"]["models"][name].items() if k != "enabled"}
    p.update(tuned_params(cfg).get(name, {}))
    return p


def build_model(name: str, cfg: dict) -> Recommender:
    mcfg = model_params(name, cfg)
    if name == "hybrid":
        return HybridRecommender(knn_params=model_params("item_knn", cfg), **mcfg)
    if name == "mf":
        mcfg.setdefault("seed", cfg["project"]["seed"])
    return MODELS[name](**mcfg)
