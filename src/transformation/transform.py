"""Stage 4 - Data transformation.

Input : data/processed/clean/*.parquet
Output: data/processed/
          train.parquet / test.parquet          time-based split (global cutoff date)
          user_activity_summary.parquet          per-user activity in the train window
          item_popularity.parquet                per-recipe popularity in the train window
          interaction_matrix.npz / rating_matrix.npz / interaction_dates.npz
                                                 sparse user x item matrices (train)
          user_index.parquet / item_index.parquet      id <-> matrix row/column mapping
          split_summary.json

Split protocol
1. cutoff = date quantile (1 - test_fraction) of all clean interactions: train < cutoff <= test.
2. k-core filter on TRAIN only (users and recipes with >= min interactions), repeated until stable.
3. test keeps only users and recipes that exist in the filtered train set (warm-start evaluation);
   how much of the test window is cold-start is recorded in split_summary.json.
All derived datasets are computed from train only, so nothing from the test period leaks into models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils.common import ensure_dir, get_logger, load_config, write_json, write_parquet

STAGE = "transformation"


def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    cutoff = df["date"].quantile(1 - test_fraction).normalize()
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy(), cutoff


def k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    while True:
        u = df["user_id"].map(df["user_id"].value_counts())
        i = df["recipe_id"].map(df["recipe_id"].value_counts())
        keep = (u >= min_user) & (i >= min_item)
        if keep.all():
            return df
        df = df[keep]


def user_activity_summary(train: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    t = train.merge(recipes[["recipe_id", "category"]], on="recipe_id", how="left")
    g = t.groupby("user_id")
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "n_rated": g["has_rating"].sum(),
        "avg_rating": g["rating"].mean(),
        "first_interaction": g["date"].min(),
        "last_interaction": g["date"].max(),
        "n_categories": g["category"].nunique(),
        "top_category": g["category"].agg(lambda s: s.value_counts().index[0]),
    })
    out["active_span_days"] = (out["last_interaction"] - out["first_interaction"]).dt.days
    return out.reset_index()


def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("recipe_id")
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "n_rated": g["has_rating"].sum(),
        "avg_rating": g["rating"].mean(),
        "first_interaction": g["date"].min(),
        "last_interaction": g["date"].max(),
    }).reset_index()
    out["popularity_rank"] = out["n_interactions"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def build_matrices(train: pd.DataFrame):
    users = np.sort(train["user_id"].unique())
    items = np.sort(train["recipe_id"].unique())
    uidx = pd.Series(np.arange(len(users)), index=users)
    iidx = pd.Series(np.arange(len(items)), index=items)
    rows, cols = uidx[train["user_id"]].values, iidx[train["recipe_id"]].values
    shape = (len(users), len(items))
    implicit = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=shape)
    rated = train["has_rating"].values
    ratings = sp.csr_matrix(
        (train["rating"].values[rated].astype(np.float32), (rows[rated], cols[rated])), shape=shape
    )
    days = ((train["date"] - pd.Timestamp("1999-01-01")).dt.days + 1).values.astype(np.float32)
    dates = sp.csr_matrix((days, (rows, cols)), shape=shape)   # day number of each review (for recency)
    user_index = pd.DataFrame({"user_id": users, "uidx": np.arange(len(users))})
    item_index = pd.DataFrame({"recipe_id": items, "iidx": np.arange(len(items))})
    return implicit, ratings, dates, user_index, item_index


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    tcfg = cfg["transformation"]
    pdir = Path(cfg["paths"]["processed_dir"])
    cdir = pdir / "clean"
    interactions = pd.read_parquet(cdir / "interactions.parquet")
    recipes = pd.read_parquet(cdir / "recipes.parquet")

    train_all, test_all, cutoff = time_split(interactions, tcfg["split"]["test_fraction"])
    train = k_core(train_all, tcfg["min_user_interactions"], tcfg["min_item_interactions"])
    warm = test_all["user_id"].isin(train["user_id"]) & test_all["recipe_id"].isin(train["recipe_id"])
    test = test_all[warm].copy()
    thr = tcfg["relevance_threshold"]
    test["is_relevant"] = test["rating"] >= thr

    implicit, ratings, dates, user_index, item_index = build_matrices(train)
    ua = user_activity_summary(train, recipes)
    ip = item_popularity(train)

    write_parquet(train.reset_index(drop=True), pdir / "train.parquet")
    write_parquet(test.reset_index(drop=True), pdir / "test.parquet")
    write_parquet(ua, pdir / "user_activity_summary.parquet")
    write_parquet(ip, pdir / "item_popularity.parquet")
    write_parquet(user_index, pdir / "user_index.parquet")
    write_parquet(item_index, pdir / "item_index.parquet")
    sp.save_npz(pdir / "interaction_matrix.npz", implicit)
    sp.save_npz(pdir / "rating_matrix.npz", ratings)
    sp.save_npz(pdir / "interaction_dates.npz", dates)

    summary = {
        "cutoff_date": cutoff,
        "train_window": [train_all["date"].min(), train_all["date"].max()],
        "test_window": [test_all["date"].min(), test_all["date"].max()],
        "train_interactions_before_kcore": len(train_all),
        "train_interactions": len(train),
        "train_users": int(train["user_id"].nunique()),
        "train_items": int(train["recipe_id"].nunique()),
        "matrix_density": float(implicit.nnz / (implicit.shape[0] * implicit.shape[1])),
        "test_interactions_total": len(test_all),
        "test_interactions_warm": len(test),
        "test_relevant_interactions": int(test["is_relevant"].sum()),
        "test_users_warm": int(test["user_id"].nunique()),
        "test_users_with_relevant": int(test.loc[test["is_relevant"], "user_id"].nunique()),
        "test_share_cold_user": float((~test_all["user_id"].isin(train["user_id"])).mean()),
        "test_share_cold_item": float((~test_all["recipe_id"].isin(train["recipe_id"])).mean()),
        "relevance_threshold": thr,
    }
    write_json(summary, pdir / "split_summary.json")
    log.info("Cutoff %s | train %d interactions (%d users x %d items, density %.5f) | warm test %d (%d users)",
             cutoff.date(), summary["train_interactions"], summary["train_users"], summary["train_items"],
             summary["matrix_density"], summary["test_interactions_warm"], summary["test_users_warm"])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
