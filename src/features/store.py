"""Read API of the custom feature store.

Training, evaluation and serving code all fetch features through this class, so the feature
definitions are shared, and the only thing that changes between training and prediction is the *view*.

    fs = FeatureStore(cfg, view="training")
    fs.get_user_features([123, 456], ["user_activity_count", "user_preferred_category"])
    fs.get_item_content(recipe_ids)          # TF-IDF rows aligned with recipe_ids
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.features.registry import FEATURES
from src.utils.common import read_json

_REGISTERED = {f.name: f for f in FEATURES}


class FeatureStore:
    def __init__(self, cfg: dict, view: str = "training") -> None:
        self.dir = Path(cfg["paths"]["features_dir"]) / view
        if not self.dir.exists():
            raise FileNotFoundError(f"Feature view '{view}' not materialised at {self.dir} - run the features stage")
        self.view = view

    @cached_property
    def metadata(self) -> dict:
        return read_json(self.dir / "_metadata.json")

    @cached_property
    def _users(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "user_features.parquet").set_index("user_id")

    @cached_property
    def _items(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "item_features.parquet").set_index("recipe_id")

    @cached_property
    def _content(self) -> tuple[sp.csr_matrix, pd.Series]:
        X = sp.load_npz(self.dir / "item_content_tfidf.npz").tocsr()
        idx = pd.read_parquet(self.dir / "item_content_index.parquet")["recipe_id"]
        return X, pd.Series(np.arange(len(idx)), index=idx.values)

    @staticmethod
    def _select(df: pd.DataFrame, ids: Iterable | None, features: Sequence[str] | None) -> pd.DataFrame:
        if features is not None:
            unknown = [f for f in features if f not in _REGISTERED]
            if unknown:
                raise KeyError(f"Not in feature registry: {unknown}")
            df = df[list(features)]
        return df if ids is None else df.reindex(list(ids))

    def get_user_features(self, user_ids: Iterable | None = None, features: Sequence[str] | None = None) -> pd.DataFrame:
        return self._select(self._users, user_ids, features)

    def get_item_features(self, recipe_ids: Iterable | None = None, features: Sequence[str] | None = None) -> pd.DataFrame:
        return self._select(self._items, recipe_ids, features)

    def get_interaction_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.dir / "interaction_features.parquet")

    def get_item_content(self, recipe_ids: Sequence[int]) -> sp.csr_matrix:
        """TF-IDF rows for recipe_ids (all-zero row for unknown recipes)."""
        X, pos = self._content
        p = pos.reindex(list(recipe_ids))
        known = p.notna().values
        rows = np.zeros(len(p), dtype=np.int64)
        rows[known] = p[known].astype(np.int64).values
        out = X[rows]
        if not known.all():
            mask = sp.diags(known.astype(np.float32))
            out = mask @ out
        return sp.csr_matrix(out)
