"""Produce recommendations for a user with a trained model + the *serving* feature view.

    python -m src.serving.recommend --user-id 424680 --model hybrid --k 10

* Known users (in the model's train index) are scored by the model. Every recipe they have ever
  reviewed (train and later) is excluded.
* Unknown / new users get the cold-start fallback: the most popular recipes over the last
  `features.recent_window_days`, taken from the serving feature view.
Display columns (category, minutes, rating) come from the same feature store used in training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.store import FeatureStore
from src.utils.common import load_config, read_json

DISPLAY = ["item_category", "item_minutes", "item_bayesian_rating", "item_popularity"]


def recommend(cfg: dict, user_id: int, model_name: str = "hybrid", k: int = 10) -> pd.DataFrame:
    pdir = Path(cfg["paths"]["processed_dir"])
    fs = FeatureStore(cfg, view="serving")
    item_ids = pd.read_parquet(pdir / "item_index.parquet")["recipe_id"].values
    user_ids = pd.read_parquet(pdir / "user_index.parquet")["user_id"].values
    names = pd.read_parquet(pdir / "clean" / "recipes.parquet", columns=["recipe_id", "name"]).set_index("recipe_id")["name"]
    history = pd.read_parquet(pdir / "clean" / "interactions.parquet", columns=["user_id", "recipe_id"])
    seen = set(history.loc[history["user_id"] == user_id, "recipe_id"])

    pos = np.where(user_ids == user_id)[0]
    if len(pos):
        model = joblib.load(read_json(Path(cfg["paths"]["models_dir"]) / "registry.json")[model_name]["path"])
        scores = model.score(pos)[0]
        mask = np.isin(item_ids, list(seen))
        scores[mask] = -np.inf
        top = np.argsort(-scores)[:k]
        rec_ids, rec_scores, source = item_ids[top], scores[top], model_name
    else:
        itf = fs.get_item_features(features=["item_recent_popularity", "item_bayesian_rating"])
        itf = itf[~itf.index.isin(seen)]
        itf = itf.sort_values(["item_recent_popularity", "item_bayesian_rating"], ascending=False).head(k)
        rec_ids, rec_scores, source = itf.index.values, itf["item_recent_popularity"].values, "cold-start popularity"

    out = fs.get_item_features(rec_ids, DISPLAY).reset_index()
    out.insert(1, "name", names.reindex(rec_ids).values)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["score"] = np.round(rec_scores.astype(float), 4)
    out.attrs["source"] = source
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--model", default="hybrid")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    fs = FeatureStore(cfg, view="serving")
    uf = fs.get_user_features([a.user_id], ["user_activity_count", "user_preferred_category", "user_avg_rating"])
    print(f"User {a.user_id} profile (serving view):\n{uf.to_string()}\n")
    recs = recommend(cfg, a.user_id, a.model, a.k)
    print(f"Top-{a.k} recommendations ({recs.attrs['source']}):")
    with pd.option_context("display.max_colwidth", 45, "display.width", 160):
        print(recs.to_string(index=False))
