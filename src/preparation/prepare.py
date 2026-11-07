"""Stage 3 - Data preparation (cleaning).

Input : data/processed/validated/{recipes,users,interactions}.parquet  (typed, errors removed)
Output: data/processed/clean/{recipes,users,interactions}.parquet + reports/preparation/summary.json

What happens here
* recipes      - text normalisation, nutrition list -> 7 numeric columns, implausible values
                 (minutes == 0 or > 30 days, calories > limit) set to missing and imputed with
                 the category median, primary category / cuisine derived from tags.
* interactions - rating 0 ("review without stars") -> missing rating, the interaction is kept as an
                 implicit signal; repeated (user, recipe) reviews collapsed to the latest one.
* users        - keep static attributes only; anything that counts activity is recomputed
                 point-in-time by the feature layer (so nothing leaks from the test period).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import ensure_dir, get_logger, load_config, write_json, write_parquet

STAGE = "preparation"
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv",
                  "saturated_fat_pdv", "carbohydrates_pdv"]


def _norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def _norm_list(values) -> list[str]:
    if not isinstance(values, (list, np.ndarray)):
        return []
    seen, out = set(), []
    for v in values:
        v = " ".join(str(v).strip().lower().split())
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _first_match(tags: list[str], priority: list[str], default: str) -> str:
    tagset = set(tags)
    for p in priority:
        if p in tagset:
            return p
    return default


def prepare_recipes(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    pcfg, vcfg = cfg["preparation"], cfg["validation"]
    out = pd.DataFrame({"recipe_id": df["id"].astype("int64")})
    out["name"] = _norm_text(df["name"])
    stats["recipes_missing_name_filled"] = int((out["name"] == "").sum())
    out.loc[out["name"] == "", "name"] = "unnamed recipe " + out["recipe_id"].astype(str)
    out["description"] = _norm_text(df["description"])
    out["contributor_id"] = df["contributor_id"].astype("int64")
    out["submitted"] = pd.to_datetime(df["submitted"])
    out["n_steps"] = df["n_steps"].astype("int64")
    out["n_ingredients"] = df["n_ingredients"].astype("int64")

    out["tags"] = df["tags"].map(_norm_list)
    out["ingredients"] = df["ingredients"].map(_norm_list)
    out["n_tags"] = out["tags"].map(len)
    out["category"] = out["tags"].map(lambda t: _first_match(t, pcfg["category_priority"], "other"))
    out["cuisine"] = out["tags"].map(lambda t: _first_match(t, pcfg["cuisine_tags"], "unspecified"))

    nut = pd.DataFrame(df["nutrition"].tolist(), columns=NUTRITION_COLS, index=df.index).astype(float)
    out[NUTRITION_COLS] = nut.values

    # implausible values -> missing -> category median
    minutes = df["minutes"].astype("float")
    bad_minutes = (minutes > vcfg["max_recipe_minutes"]) | (minutes == 0)
    out["minutes"] = minutes.where(~bad_minutes)
    bad_cal = out["calories"] > vcfg["max_calories"]
    out.loc[bad_cal, "calories"] = np.nan
    out["minutes_imputed"] = out["minutes"].isna()
    out["calories_imputed"] = out["calories"].isna()
    for col in ["minutes", "calories"]:
        med = out.groupby("category")[col].transform("median")
        out[col] = out[col].fillna(med).fillna(out[col].median())
    stats["recipes_minutes_imputed"] = int(out["minutes_imputed"].sum())
    stats["recipes_calories_imputed"] = int(out["calories_imputed"].sum())
    stats["recipes_missing_description"] = int((out["description"] == "").sum())
    stats["recipes_without_category"] = int((out["category"] == "other").sum())
    return out


def prepare_interactions(df: pd.DataFrame, recipes: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    pcfg = cfg["preparation"]
    out = pd.DataFrame({
        "user_id": df["user_id"].astype("int64"),
        "recipe_id": df["recipe_id"].astype("int64"),
        "date": pd.to_datetime(df["date"]),
        "rating": df["rating"].astype(float),
    })
    review = df["review"].fillna("").astype(str)
    out["review_length"] = review.str.split().str.len().fillna(0).astype(int)
    if pcfg["zero_rating_as_missing"]:
        stats["interactions_zero_rating_to_missing"] = int((out["rating"] == 0).sum())
        out.loc[out["rating"] == 0, "rating"] = np.nan
    out["has_rating"] = out["rating"].notna()

    # keep only interactions whose recipe survived preparation
    before = len(out)
    out = out[out["recipe_id"].isin(recipes["recipe_id"])]
    stats["interactions_dropped_unknown_recipe"] = before - len(out)

    # repeated (user, recipe) pairs -> keep the latest, remember how many times it happened
    out = out.sort_values(["user_id", "recipe_id", "date"])
    out["n_reviews_of_pair"] = out.groupby(["user_id", "recipe_id"])["date"].transform("size")
    before = len(out)
    keep = "last" if pcfg["duplicate_user_item"] == "keep_latest" else "first"
    out = out.drop_duplicates(["user_id", "recipe_id"], keep=keep)
    stats["interactions_repeat_pairs_collapsed"] = before - len(out)
    return out.sort_values("date").reset_index(drop=True)


def prepare_users(df: pd.DataFrame, interactions: pd.DataFrame, recipes: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "user_id": df["user_id"].astype("int64"),
        "member_since": pd.to_datetime(df["member_since"]),
        "first_contribution_date": pd.to_datetime(df["first_contribution_date"]),
    })
    active = set(interactions["user_id"]) | set(recipes["contributor_id"])
    before = len(out)
    out = out[out["user_id"].isin(active)].reset_index(drop=True)
    stats["users_dropped_no_valid_activity"] = before - len(out)
    return out


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    vdir = Path(cfg["paths"]["processed_dir"]) / "validated"
    cdir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / "clean")
    stats: dict = {}

    recipes = prepare_recipes(pd.read_parquet(vdir / "recipes.parquet"), cfg, stats)
    interactions = prepare_interactions(pd.read_parquet(vdir / "interactions.parquet"), recipes, cfg, stats)
    users = prepare_users(pd.read_parquet(vdir / "users.parquet"), interactions, recipes, stats)

    write_parquet(recipes, cdir / "recipes.parquet")
    write_parquet(interactions, cdir / "interactions.parquet")
    write_parquet(users, cdir / "users.parquet")

    stats.update({
        "recipes": len(recipes), "interactions": len(interactions), "users": len(users),
        "interactions_rated": int(interactions["has_rating"].sum()),
        "date_min": interactions["date"].min(), "date_max": interactions["date"].max(),
        "category_distribution": recipes["category"].value_counts().to_dict(),
    })
    write_json(stats, ensure_dir(Path(cfg["paths"]["reports_dir"]) / "preparation") / "summary.json")
    log.info("Clean data: %d recipes, %d users, %d interactions (%d rated)",
             len(recipes), len(users), len(interactions), stats["interactions_rated"])
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
