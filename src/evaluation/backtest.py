"""Stage 8 - Rolling-origin backtest (several time-based train/test splits, not just one).

Fold i (i = 0 .. n_folds-1):
    cutoff_i = main split cutoff + i * horizon_days
    train    = every clean review before cutoff_i (5-core filtered)   - expanding window
    test     = reviews in [cutoff_i, cutoff_i + horizon_days)          - warm users/recipes, rating >= threshold
    features = point-in-time view materialised as of cutoff_i (data/features/backtest_<i>)
Every enabled model is re-fitted per fold with the tuned hyper-parameters. Those were selected on a
validation window that ends at the main cutoff, so all backtest test windows lie after the tuning
data (no look-ahead in model selection).

Outputs: reports/evaluation/backtest.{csv,md} and an MLflow run "backtest" (per-fold metrics as steps).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.evaluate import top_k
from src.features.build import materialize
from src.features.store import FeatureStore
from src.training.models import TrainData, build_model
from src.training.train import enabled_models
from src.transformation.transform import build_matrices, k_core
from src.utils.common import (
    df_to_markdown,
    ensure_dir,
    get_logger,
    load_config,
    pipeline_run_id,
    read_json,
    set_seed,
    setup_mlflow,
)

STAGE = "backtest"


def fold_data(i: int, cutoff: pd.Timestamp, end: pd.Timestamp, inter, users, recipes, cfg, log):
    tcfg = cfg["transformation"]
    train = k_core(inter[inter["date"] < cutoff], tcfg["min_user_interactions"], tcfg["min_item_interactions"])
    R, ratings, dates, uidx, iidx = build_matrices(train)
    view = f"backtest_{i}"
    materialize(view, inter, users, recipes, cutoff, cfg, log)
    data = TrainData(R=R.tocsr(), ratings=ratings.tocsr(), user_ids=uidx["user_id"].values,
                     item_ids=iidx["recipe_id"].values, fs=FeatureStore(cfg, view), dates=dates.tocsr())
    test = inter[(inter["date"] >= cutoff) & (inter["date"] < end)]
    u = pd.Series(np.arange(len(uidx)), index=uidx["user_id"].values)
    it = pd.Series(np.arange(len(iidx)), index=iidx["recipe_id"].values)
    test = test[test["user_id"].isin(u.index) & test["recipe_id"].isin(it.index)
                & (test["rating"] >= tcfg["relevance_threshold"])]
    rel = pd.Series(it[test["recipe_id"]].values, index=u[test["user_id"]].values).groupby(level=0).apply(set)
    return data, rel.index.values.astype(np.int64), list(rel.values)


def run(cfg: dict) -> pd.DataFrame:
    log = get_logger(STAGE, cfg)
    set_seed(cfg["project"]["seed"])
    bcfg = cfg["evaluation"]["backtest"]
    k, batch = cfg["training"]["k"], cfg["evaluation"]["batch_size"]
    pdir = Path(cfg["paths"]["processed_dir"])
    inter = pd.read_parquet(pdir / "clean" / "interactions.parquet")
    users = pd.read_parquet(pdir / "clean" / "users.parquet")
    recipes = pd.read_parquet(pdir / "clean" / "recipes.parquet")
    cutoff0 = pd.Timestamp(read_json(pdir / "split_summary.json")["cutoff_date"])
    horizon = pd.Timedelta(days=int(bcfg["horizon_days"]))

    rows = []
    for i in range(int(bcfg["n_folds"])):
        cutoff, end = cutoff0 + i * horizon, cutoff0 + (i + 1) * horizon
        if end > inter["date"].max() + pd.Timedelta(days=1):
            log.info("Fold %d skipped: test window %s-%s goes past the end of the data", i, cutoff.date(), end.date())
            break
        data, users_i, relevant = fold_data(i, cutoff, end, inter, users, recipes, cfg, log)
        if len(users_i) == 0:
            log.info("Fold %d skipped: no warm test users", i)
            continue
        log.info("Fold %d: train < %s (%d reviews, %d users x %d recipes), test to %s: %d users",
                 i, cutoff.date(), data.R.nnz, data.n_users, data.n_items, end.date(), len(users_i))
        for name in enabled_models(cfg):
            model = build_model(name, cfg).fit(data)
            recs = top_k(model, users_i, data.R, k, batch)
            rows.append({"fold": i, "cutoff": cutoff.date(), "test_end": end.date(), "users": len(users_i),
                         "model": name, f"ndcg_at_{k}": M.ndcg_at_k(recs, relevant, k),
                         f"recall_at_{k}": M.recall_at_k(recs, relevant, k),
                         f"precision_at_{k}": M.precision_at_k(recs, relevant, k),
                         f"coverage_at_{k}": M.catalog_coverage(recs, data.n_items, k)})
            log.info("  %-10s NDCG@%d=%.4f", name, k, rows[-1][f"ndcg_at_{k}"])

    res = pd.DataFrame(rows)
    out = ensure_dir(Path(cfg["paths"]["reports_dir"]) / "evaluation")
    res.to_csv(out / "backtest.csv", index=False)
    summary = _summary(res, k, cfg["evaluation"]["bootstrap"].get("baseline", "popularity"))
    summary.to_csv(out / "backtest_summary.csv", index=False)
    _write_markdown(res, summary, k, bcfg, out / "backtest.md")

    mlflow = setup_mlflow(cfg)
    with mlflow.start_run(run_name="backtest"):
        mlflow.set_tags({"stage": "backtest", "pipeline_run_id": pipeline_run_id()})
        mlflow.log_params({"n_folds": res["fold"].nunique(), "horizon_days": bcfg["horizon_days"],
                           "first_cutoff": str(cutoff0.date())})
        for r in res.itertuples():
            mlflow.log_metric(f"{r.model}.ndcg_at_{k}", getattr(r, f"ndcg_at_{k}"), step=int(r.fold))
        for r in summary.itertuples():
            mlflow.log_metric(f"{r.model}.mean_ndcg_at_{k}", r.mean)
        mlflow.log_artifact(str(out / "backtest.csv"))
        mlflow.log_artifact(str(out / "backtest.md"))
    return res


def _summary(res: pd.DataFrame, k: int, baseline: str) -> pd.DataFrame:
    col = f"ndcg_at_{k}"
    wide = res.pivot(index="fold", columns="model", values=col)
    rows = []
    for m in wide.columns:
        rows.append({"model": m, "mean": wide[m].mean(), "std": wide[m].std(ddof=1), "min": wide[m].min(),
                     "max": wide[m].max(),
                     "folds_beating_baseline": "-" if m == baseline else f"{int((wide[m] > wide[baseline]).sum())}/{len(wide)}"})
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def _write_markdown(res, summary, k, bcfg, path: Path) -> None:
    col = f"ndcg_at_{k}"
    wide = res.pivot(index="model", columns="fold", values=col)
    folds = res.drop_duplicates("fold").set_index("fold")
    wide.columns = [f"fold {f}: {folds.at[f, 'cutoff']} (+{bcfg['horizon_days']}d, {folds.at[f, 'users']} users)"
                    for f in wide.columns]
    lines = ["# Rolling-origin backtest", "",
             f"{len(folds)} expanding-window folds; each test window is the {bcfg['horizon_days']} days after its "
             "cutoff. Models are re-fitted per fold with the tuned hyper-parameters (chosen on data before the first "
             "cutoff).", "", f"## NDCG@{k} per fold", "", df_to_markdown(wide.reset_index()), "",
             f"## Summary over folds (NDCG@{k})", "", df_to_markdown(summary), "",
             "## All metrics", "", df_to_markdown(res), ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
