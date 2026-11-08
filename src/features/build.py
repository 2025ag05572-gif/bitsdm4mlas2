"""Stage 5 - Feature engineering and materialisation (custom offline feature store).

The same `materialize(view, ..., as_of)` code path (compute_user/item/interaction_features) produces two *views*:

* training - as_of = split cutoff, built only from interactions before the cutoff
             (point-in-time correct, used for model training and evaluation)
* serving  - as_of = day after the last interaction, built from all data
             (used by src/serving/recommend.py when producing live recommendations)

Layout: data/features/<view>/{user,item,interaction}_features.parquet,
        item_content_tfidf.npz (+ item_content_index.parquet) and _metadata.json.
The feature catalog (reports/feature_catalog.{md,csv}) is generated from the registry plus
statistics of the training view.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.registry import FEATURES, by_entity
from src.utils.common import (
    df_to_markdown,
    ensure_dir,
    get_logger,
    load_config,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_parquet,
)

STAGE = "features"


def _identity(tokens):
    return tokens


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return max(0.0, float(-(p * np.log2(p)).sum()))


def compute_user_features(inter: pd.DataFrame, users: pd.DataFrame, recipes: pd.DataFrame,
                          as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    recent = as_of - pd.Timedelta(days=cfg["features"]["recent_window_days"])
    global_mean = inter["rating"].mean()
    t = inter.merge(recipes[["recipe_id", "category", "cuisine", "minutes", "calories"]], on="recipe_id", how="left")
    g = t.groupby("user_id")
    f = pd.DataFrame({
        "user_activity_count": g.size(),
        "user_rated_count": g["has_rating"].sum().astype(int),
        "user_avg_rating": g["rating"].mean(),
        "user_rating_std": g["rating"].std(),
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "user_interactions_last_365d": t[t["date"] >= recent].groupby("user_id").size(),
        "user_avg_recipe_minutes": g["minutes"].mean(),
        "user_avg_recipe_calories": g["calories"].mean(),
        "user_avg_review_length": g["review_length"].mean(),
    })
    cat_counts = t.groupby(["user_id", "category"]).size()
    top = cat_counts.sort_values(ascending=False).groupby(level=0).head(1).reset_index(level=1)
    f["user_preferred_category"] = top["category"]
    f["user_preferred_category_share"] = top[0] / f["user_activity_count"]
    f["user_category_entropy"] = cat_counts.groupby(level=0).apply(_entropy)
    cui = t.groupby(["user_id", "cuisine"]).size().sort_values(ascending=False).groupby(level=0).head(1)
    f["user_preferred_cuisine"] = cui.reset_index(level=1)["cuisine"]

    contributed = recipes[recipes["submitted"] < as_of].groupby("contributor_id").size()
    f["user_recipes_contributed"] = contributed.reindex(f.index).fillna(0).astype(int)
    member_since = users.set_index("user_id")["member_since"].reindex(f.index)
    first_seen = pd.concat([member_since, f["first_date"]], axis=1).min(axis=1)

    f["user_avg_rating"] = f["user_avg_rating"].fillna(global_mean)
    f["user_rating_std"] = f["user_rating_std"].fillna(0.0)
    f["user_interactions_last_365d"] = f["user_interactions_last_365d"].fillna(0).astype(int)
    f["user_days_since_last_interaction"] = (as_of - f["last_date"]).dt.days
    f["user_tenure_days"] = (as_of - first_seen).dt.days
    f = f.drop(columns=["first_date", "last_date"])
    f.index.name = "user_id"
    return f.reset_index()


def compute_item_features(inter: pd.DataFrame, recipes: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    recent = as_of - pd.Timedelta(days=cfg["features"]["recent_window_days"])
    m = cfg["features"]["bayesian_prior_weight"]
    known = recipes[(recipes["submitted"] < as_of) | recipes["recipe_id"].isin(inter["recipe_id"])]
    g = inter.groupby("recipe_id")
    stats = pd.DataFrame({
        "item_popularity": g.size(),
        "item_rated_count": g["has_rating"].sum(),
        "item_avg_rating": g["rating"].mean(),
        "last_date": g["date"].max(),
        "item_recent_popularity": inter[inter["date"] >= recent].groupby("recipe_id").size(),
    })
    f = known.set_index("recipe_id")[["submitted", "category", "cuisine", "minutes", "n_steps", "n_ingredients",
                                      "calories", "n_tags"]].join(stats)
    gm = inter["rating"].mean()
    n = f["item_rated_count"].fillna(0)
    f["item_bayesian_rating"] = (n * f["item_avg_rating"].fillna(gm) + m * gm) / (n + m)
    for c in ["item_popularity", "item_rated_count", "item_recent_popularity"]:
        f[c] = f[c].fillna(0).astype(int)
    f["item_days_since_last_interaction"] = (as_of - f["last_date"]).dt.days
    f["item_age_days"] = (as_of - f["submitted"]).dt.days.clip(lower=0)
    f = f.rename(columns={"category": "item_category", "cuisine": "item_cuisine", "minutes": "item_minutes",
                          "n_steps": "item_n_steps", "n_ingredients": "item_n_ingredients",
                          "calories": "item_calories", "n_tags": "item_n_tags"})
    f = f.drop(columns=["submitted", "last_date"])
    f.index.name = "recipe_id"
    return f.reset_index()


def compute_interaction_features(inter: pd.DataFrame, recipes: pd.DataFrame, uf: pd.DataFrame,
                                 itf: pd.DataFrame) -> pd.DataFrame:
    t = inter[["user_id", "recipe_id", "date", "rating"]].sort_values(["user_id", "date"]).copy()
    t = t.merge(uf[["user_id", "user_avg_rating", "user_preferred_category"]], on="user_id", how="left")
    t = t.merge(itf[["recipe_id", "item_avg_rating", "item_category"]], on="recipe_id", how="left")
    t = t.merge(recipes[["recipe_id", "submitted"]], on="recipe_id", how="left")
    t["ui_rating_deviation_user"] = t["rating"] - t["user_avg_rating"]
    t["ui_rating_deviation_item"] = t["rating"] - t["item_avg_rating"]
    t["ui_category_match"] = t["item_category"] == t["user_preferred_category"]
    t["ui_recipe_age_at_interaction_days"] = (t["date"] - t["submitted"]).dt.days
    t["ui_user_sequence_number"] = t.groupby("user_id").cumcount() + 1
    t["ui_days_since_prev_interaction"] = t.groupby("user_id")["date"].diff().dt.days
    cols = ["user_id", "recipe_id", "date"] + [f.name for f in by_entity("user_item")]
    return t[cols].reset_index(drop=True)


def compute_item_content(recipes: pd.DataFrame, recipe_ids: np.ndarray, min_df: int):
    r = recipes.set_index("recipe_id").reindex(recipe_ids)
    docs = [["tag=" + t for t in tags] + ["ing=" + i for i in ings]
            for tags, ings in zip(r["tags"], r["ingredients"])]
    vec = TfidfVectorizer(analyzer=_identity, min_df=min_df, max_df=0.5, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(docs)
    return sp.csr_matrix(X), len(vec.vocabulary_)


def _check_registry(frames: dict[str, pd.DataFrame]) -> None:
    for entity, df in frames.items():
        missing = [f.name for f in by_entity(entity) if f.dtype != "sparse vector" and f.name not in df.columns]
        if missing:
            raise ValueError(f"Registered {entity} features not produced: {missing}")


def materialize(view: str, inter: pd.DataFrame, users: pd.DataFrame, recipes: pd.DataFrame,
                as_of: pd.Timestamp, cfg: dict, log) -> dict:
    out = ensure_dir(Path(cfg["paths"]["features_dir"]) / view)
    inter = inter[inter["date"] < as_of]
    uf = compute_user_features(inter, users, recipes, as_of, cfg)
    itf = compute_item_features(inter, recipes, as_of, cfg)
    uif = compute_interaction_features(inter, recipes, uf, itf)
    _check_registry({"user": uf, "item": itf, "user_item": uif})
    content, vocab = compute_item_content(recipes, itf["recipe_id"].values, min_df=5 if len(itf) > 20000 else 2)

    for name, df in (("user_features", uf), ("item_features", itf), ("interaction_features", uif)):
        df = df.copy()
        df["feature_timestamp"] = as_of
        write_parquet(df, out / f"{name}.parquet")
    sp.save_npz(out / "item_content_tfidf.npz", content)
    write_parquet(pd.DataFrame({"recipe_id": itf["recipe_id"].values}), out / "item_content_index.parquet")
    meta = {
        "view": view, "as_of": as_of, "created_at": utc_now(),
        "rows": {"user": len(uf), "item": len(itf), "user_item": len(uif)},
        "content_vocabulary_size": vocab,
        "features": [f.name for f in FEATURES],
        "source_checksum": sha256_file(Path(cfg["paths"]["processed_dir"]) / "clean" / "interactions.parquet"),
    }
    write_json(meta, out / "_metadata.json")
    log.info("[%s] as_of=%s users=%d items=%d interactions=%d vocab=%d",
             view, as_of.date(), len(uf), len(itf), len(uif), vocab)
    return {"user": uf, "item": itf, "user_item": uif, "meta": meta}


def write_catalog(train_view: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        df = train_view[f.entity]
        if f.name in df.columns:
            s = df[f.name]
            null_pct = float(s.isna().mean())
            if pd.api.types.is_bool_dtype(s) or s.dtype == object:
                example = f"top: {s.mode().iloc[0]}" if s.notna().any() else ""
            else:
                example = f"mean {s.mean():.4g} (min {s.min():.4g}, max {s.max():.4g})"
        else:
            null_pct, example = 0.0, f"{train_view['meta']['content_vocabulary_size']} dims"
        rows.append({"feature": f.name, "entity": f.entity, "dtype": f.dtype, "description": f.description,
                     "transformation": f.transformation, "source": f.source, "used_by": f.used_by,
                     "null_pct": round(100 * null_pct, 2), "training_view_stats": example})
    cat = pd.DataFrame(rows)
    rep = ensure_dir(cfg["paths"]["reports_dir"])
    cat.to_csv(rep / "feature_catalog.csv", index=False)
    meta = train_view["meta"]
    md = ["# Feature Catalog", "",
          f"Generated {meta['created_at']} from the **training** view (as_of = {pd.Timestamp(meta['as_of']).date()}).",
          "Entity keys: `user_id` (user), `recipe_id` (item), `(user_id, recipe_id, date)` (user_item).",
          "Every feature is computed by `src/features/build.py` and served through `src/features/store.py`.", "",
          df_to_markdown(cat.rename(columns={"null_pct": "null %"}), ".2f")]
    (rep / "feature_catalog.md").write_text("\n".join(md) + "\n")
    return cat


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    pdir = Path(cfg["paths"]["processed_dir"])
    cdir = pdir / "clean"
    inter = pd.read_parquet(cdir / "interactions.parquet")
    recipes = pd.read_parquet(cdir / "recipes.parquet")
    users = pd.read_parquet(cdir / "users.parquet")
    split = read_json(pdir / "split_summary.json")

    cutoff = pd.Timestamp(split["cutoff_date"])
    train_view = materialize("training", inter, users, recipes, cutoff, cfg, log)
    serving_as_of = inter["date"].max().normalize() + pd.Timedelta(days=1)
    materialize("serving", inter, users, recipes, serving_as_of, cfg, log)
    cat = write_catalog(train_view, cfg)
    return {"training_as_of": cutoff, "serving_as_of": serving_as_of, "n_features": len(cat)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
