"""Stage 6a - Hyper-parameter tuning on a validation window (the test set is never used).

1. The train period is split again by time: the latest `training.tuning.val_fraction` of train
   interactions become a validation window and the rest is the inner-train set (re-filtered to a k-core).
2. Point-in-time features are materialised as of the inner cutoff (feature view "tuning"),
   so no information from the validation window leaks into the candidates.
3. Every candidate is scored on validation NDCG@K (RMSE for MF) and logged as a nested MLflow run.
4. The best parameters are written to models/best_params.json. With training.use_tuned_params the
   training stage merges them over configs/config.yaml.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import metrics as M
from src.evaluation.evaluate import top_k
from src.features.build import materialize
from src.features.store import FeatureStore
from src.training.models import (
    BiasedMFRecommender,
    ContentRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    TrainData,
)
from src.transformation.transform import build_matrices, k_core, time_split
from src.utils.common import (
    df_to_markdown,
    ensure_dir,
    get_logger,
    load_config,
    pipeline_run_id,
    set_seed,
    setup_mlflow,
    utc_now,
    write_json,
)

STAGE = "tuning"


def validation_split(cfg: dict, log):
    tcfg, pdir = cfg["transformation"], Path(cfg["paths"]["processed_dir"])
    train = pd.read_parquet(pdir / "train.parquet")
    inner, val, cutoff = time_split(train, cfg["training"]["tuning"]["val_fraction"])
    inner = k_core(inner, tcfg["min_user_interactions"], tcfg["min_item_interactions"])
    R, ratings, dates, uidx, iidx = build_matrices(inner)

    clean = pdir / "clean"
    materialize("tuning", pd.read_parquet(clean / "interactions.parquet"), pd.read_parquet(clean / "users.parquet"),
                pd.read_parquet(clean / "recipes.parquet"), cutoff, cfg, log)
    data = TrainData(R=R.tocsr(), ratings=ratings.tocsr(), user_ids=uidx["user_id"].values,
                     item_ids=iidx["recipe_id"].values, fs=FeatureStore(cfg, "tuning"), dates=dates.tocsr())

    u = pd.Series(np.arange(len(uidx)), index=uidx["user_id"].values)
    i = pd.Series(np.arange(len(iidx)), index=iidx["recipe_id"].values)
    val = val[val["user_id"].isin(u.index) & val["recipe_id"].isin(i.index)].copy()
    val["uidx"], val["iidx"] = u[val["user_id"]].values, i[val["recipe_id"]].values
    rel = val[val["rating"] >= tcfg["relevance_threshold"]].groupby("uidx")["iidx"].apply(set)
    log.info("Validation window from %s: inner-train %d interactions (%d users x %d items), %d validation users",
             cutoff.date(), R.nnz, R.shape[0], R.shape[1], len(rel))
    return data, rel.index.values.astype(np.int64), list(rel.values), val[val["rating"].notna()], cutoff


def _score(model, data, users, relevant, k, batch) -> dict:
    recs = top_k(model, users, data.R, k, batch)
    return {f"val_ndcg_at_{k}": M.ndcg_at_k(recs, relevant, k), f"val_recall_at_{k}": M.recall_at_k(recs, relevant, k),
            f"val_precision_at_{k}": M.precision_at_k(recs, relevant, k),
            f"val_coverage_at_{k}": M.catalog_coverage(recs, data.n_items, k)}


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    set_seed(cfg["project"]["seed"])
    tcfg = cfg["training"]["tuning"]
    grid = tcfg["grid"]
    k, batch = cfg["training"]["k"], cfg["evaluation"]["batch_size"]
    data, users, relevant, rated, cutoff = validation_split(cfg, log)
    mlflow = setup_mlflow(cfg)
    rows: list[dict] = []
    key = f"val_ndcg_at_{k}"

    def record(model: str, params: dict, metrics: dict) -> None:
        rows.append({"model": model, "params": params, **metrics})
        with mlflow.start_run(run_name=f"tune-{model}", nested=True):
            mlflow.set_tags({"stage": "tuning", "model": model, "pipeline_run_id": pipeline_run_id()})
            mlflow.log_params({f"model.{p}": v for p, v in params.items()})
            mlflow.log_metrics(metrics)
        log.info("  %-10s %s -> %s", model, params, {m: round(v, 4) for m, v in metrics.items()})

    best: dict[str, dict] = {}
    with mlflow.start_run(run_name="tuning") as parent:
        mlflow.set_tags({"stage": "tuning", "pipeline_run_id": pipeline_run_id()})
        mlflow.log_params({"val_fraction": tcfg["val_fraction"], "inner_cutoff": str(cutoff.date()),
                           "val_users": len(users), "selection_metric": key})

        # popularity variants
        pops = {}
        for score in grid["popularity"]["score"]:
            m = PopularityRecommender(score=score).fit(data)
            pops[score] = m
            record("popularity", {"score": score}, _score(m, data, users, relevant, k, batch))
        best["popularity"] = max((r for r in rows if r["model"] == "popularity"), key=lambda r: r[key])["params"]

        # item-kNN: fit S once per (neighbours, weighting, shrinkage), vary history window at scoring time
        g = grid["item_knn"]
        knn_models = {}
        for nn, uw, sh in itertools.product(g["n_neighbors"], g["user_weighting"], g["shrinkage"]):
            m = ItemKNNRecommender(n_neighbors=nn, user_weighting=uw, shrinkage=sh).fit(data)
            for hw in g["history_window"]:
                m.set_history_window(data, hw)
                params = {"n_neighbors": nn, "user_weighting": uw, "shrinkage": sh, "history_window": hw}
                record("item_knn", params, _score(m, data, users, relevant, k, batch))
            knn_models[(nn, uw, sh)] = m
        best["item_knn"] = max((r for r in rows if r["model"] == "item_knn"), key=lambda r: r[key])["params"]

        # MF: selected on validation RMSE
        g = grid["mf"]
        base_mf = {kk: v for kk, v in cfg["training"]["models"]["mf"].items() if kk != "enabled"}
        for nf, reg in itertools.product(g["n_factors"], g["reg"]):
            params = {**base_mf, "n_factors": nf, "reg": reg, "seed": cfg["project"]["seed"]}
            m = BiasedMFRecommender(**params).fit(data)
            pred = m.predict(rated["uidx"].values, rated["iidx"].values)
            record("mf", {"n_factors": nf, "reg": reg},
                   {"val_rmse": M.rmse(rated["rating"], pred), "val_mae": M.mae(rated["rating"], pred)})
        best["mf"] = min((r for r in rows if r["model"] == "mf"), key=lambda r: r["val_rmse"])["params"]

        # hybrid: re-use the best kNN, the content model and the best popularity variant
        bk = best["item_knn"]
        knn = knn_models[(bk["n_neighbors"], bk["user_weighting"], bk["shrinkage"])]
        knn.set_history_window(data, bk["history_window"])
        content = ContentRecommender().fit(data)
        pop = pops[best["popularity"]["score"]]
        g = grid["hybrid"]
        cold = g.get("cold_user_threshold", [cfg["training"]["models"]["hybrid"]["cold_user_threshold"]])
        for (wk, wc, wp), cb, ct in itertools.product(g["weights"], g["category_boost"], cold):
            params = {"w_knn": wk, "w_content": wc, "w_popularity": wp, "category_boost": cb,
                      "cold_user_threshold": ct, "popularity_score": best["popularity"]["score"]}
            m = HybridRecommender(**params).fit(data, components={"knn": knn, "content": content, "pop": pop})
            record("hybrid", params, _score(m, data, users, relevant, k, batch))
        best["hybrid"] = max((r for r in rows if r["model"] == "hybrid"), key=lambda r: r[key])["params"]
        mlflow.log_dict(best, "best_params.json")

    out = ensure_dir(Path(cfg["paths"]["reports_dir"]) / "tuning")
    res = pd.DataFrame(rows)
    res["params"] = res["params"].map(lambda p: ", ".join(f"{a}={b}" for a, b in p.items()))
    res.to_csv(out / "tuning_results.csv", index=False)
    lines = ["# Hyper-parameter tuning (validation window inside the train period)", "",
             f"Inner cutoff {cutoff.date()} (latest {tcfg['val_fraction']:.0%} of train interactions), "
             f"{len(users):,} validation users. Ranking models are selected on `{key}`, MF on `val_rmse`. "
             "The test set is not used here.", ""]
    for model, grp in res.groupby("model", sort=False):
        cols = ["params"] + [c for c in grp.columns if c.startswith("val_") and grp[c].notna().any()]
        sort = "val_rmse" if model == "mf" else key
        lines += [f"## {model}", "", df_to_markdown(grp[cols].sort_values(sort, ascending=(model == "mf"))), "",
                  f"**Selected:** `{best[model]}`", ""]
    (out / "tuning_results.md").write_text("\n".join(lines))
    best_path = Path(cfg["paths"]["models_dir"]) / "best_params.json"
    write_json({"generated_at": utc_now(), "inner_cutoff": cutoff, "selection_metric": key,
                "mlflow_run_id": parent.info.run_id, "params": best}, best_path)
    log.info("Best params -> %s", best_path)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
