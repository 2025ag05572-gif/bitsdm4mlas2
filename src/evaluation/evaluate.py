"""Stage 7 - Model evaluation.

Protocol (time-based, warm-start):
  * evaluation users = test users (seen in train) with at least one relevant test interaction
    (rating >= transformation.relevance_threshold)
  * every model ranks the full train catalog; recipes the user already reviewed in train are excluded
  * ranking metrics @K for K in evaluation.k_values: Precision, Recall, NDCG, HitRate
  * beyond-accuracy @K: catalog Coverage, Diversity (intra-list, TF-IDF cosine), category diversity, novelty
  * rating metrics (models that predict stars): RMSE and MAE on all rated warm test interactions,
    next to a global-mean reference
Results go to reports/evaluation/ and are logged to each model's MLflow training run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from src.evaluation import metrics as M
from src.features.store import FeatureStore
from src.training.models import load_train_data
from src.utils.common import (
    df_to_markdown,
    ensure_dir,
    get_logger,
    load_config,
    pipeline_run_id,
    read_json,
    set_seed,
    setup_mlflow,
    utc_now,
    write_json,
)

STAGE = "evaluation"
SEGMENTS = [(0, 10, "5-9 reviews"), (10, 50, "10-49 reviews"), (50, np.inf, "50+ reviews")]


def top_k(model, uidx: np.ndarray, R, k: int, batch: int) -> np.ndarray:
    out = np.empty((len(uidx), k), dtype=np.int64)
    for s in range(0, len(uidx), batch):
        b = uidx[s:s + batch]
        S = np.array(model.score(b), dtype=np.float32, copy=True)
        rows, cols = R[b].nonzero()
        S[rows, cols] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1, kind="stable")
        out[s:s + len(b)] = np.take_along_axis(part, order, axis=1)
    return out


def ranking_metrics(recs, relevant, ks, n_items, V, cats, pop, n_users) -> dict:
    res = {}
    for k in ks:
        res[f"precision_at_{k}"] = M.precision_at_k(recs, relevant, k)
        res[f"recall_at_{k}"] = M.recall_at_k(recs, relevant, k)
        res[f"ndcg_at_{k}"] = M.ndcg_at_k(recs, relevant, k)
        res[f"hit_rate_at_{k}"] = M.hit_rate_at_k(recs, relevant, k)
        res[f"coverage_at_{k}"] = M.catalog_coverage(recs, n_items, k)
        res[f"diversity_at_{k}"] = M.intra_list_diversity(recs, V, k)
        res[f"category_diversity_at_{k}"] = M.category_diversity(recs, cats, k)
        res[f"novelty_at_{k}"] = M.novelty(recs, pop, n_users, k)
    return res


def _sample_recommendations(cfg, data, test, eval_users, recs_by_model, relevant, n=3) -> str:
    recipes = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "clean" / "recipes.parquet",
                              columns=["recipe_id", "name", "category"]).set_index("recipe_id")
    train = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "train.parquet")
    rng = np.random.default_rng(cfg["project"]["seed"])
    picks = rng.choice(len(eval_users), size=min(n, len(eval_users)), replace=False)
    lines = ["# Sample recommendations (top-5)", "",
             "Recipes marked **(hit)** were reviewed with >= threshold stars by the user in the test window.", ""]
    for p in picks:
        u = eval_users[p]
        uid = int(data.user_ids[u])
        hist = train[train["user_id"] == uid].sort_values("date").tail(5)["recipe_id"]
        lines += [f"## user {uid}", "", "Last 5 train reviews: " +
                  "; ".join(f"{recipes.at[r, 'name']} ({recipes.at[r, 'category']})" for r in hist), ""]
        for name, recs in recs_by_model.items():
            items = []
            for i in recs[p][:5]:
                rid = int(data.item_ids[i])
                tag = " **(hit)**" if i in relevant[p] else ""
                items.append(f"{recipes.at[rid, 'name']}{tag}")
            lines.append(f"- **{name}**: " + "; ".join(items))
        lines.append("")
    return "\n".join(lines)


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    set_seed(cfg["project"]["seed"])
    ecfg = cfg["evaluation"]
    ks = sorted(ecfg["k_values"])
    kmax = max(ks)
    pdir = Path(cfg["paths"]["processed_dir"])
    out_dir = ensure_dir(Path(cfg["paths"]["reports_dir"]) / "evaluation")

    data = load_train_data(cfg)
    test = pd.read_parquet(pdir / "test.parquet")
    uix = pd.Series(np.arange(data.n_users), index=data.user_ids)
    iix = pd.Series(np.arange(data.n_items), index=data.item_ids)
    test["uidx"] = uix.reindex(test["user_id"]).values
    test["iidx"] = iix.reindex(test["recipe_id"]).values
    test = test.dropna(subset=["uidx", "iidx"]).astype({"uidx": int, "iidx": int})

    rel = test[test["is_relevant"]].groupby("uidx")["iidx"].apply(set)
    eval_users = rel.index.values.astype(np.int64)
    relevant = list(rel.values)
    log.info("Evaluating on %d users (%d relevant test interactions), K=%s", len(eval_users),
             sum(map(len, relevant)), ks)

    fs: FeatureStore = data.fs
    V = normalize(fs.get_item_content(data.item_ids))
    cats = fs.get_item_features(data.item_ids, ["item_category"])["item_category"].fillna("other").values
    pop = np.asarray(data.R.sum(axis=0)).ravel()
    activity = fs.get_user_features(data.user_ids[eval_users], ["user_activity_count"])["user_activity_count"].values

    rated = test[test["rating"].notna()]
    registry = read_json(Path(cfg["paths"]["models_dir"]) / "registry.json")
    rows, seg_rows, recs_by_model = [], [], {}
    mlflow = setup_mlflow(cfg)
    for name, entry in registry.items():
        if not cfg["training"]["models"].get(name, {}).get("enabled", True):
            continue
        model = joblib.load(entry["path"])
        recs = top_k(model, eval_users, data.R, kmax, ecfg["batch_size"])
        recs_by_model[name] = recs
        res = {"model": name, **ranking_metrics(recs, relevant, ks, data.n_items, V, cats, pop, data.n_users)}
        if model.predicts_ratings:
            pred = model.predict(rated["uidx"].values, rated["iidx"].values)
            res["rmse"], res["mae"] = M.rmse(rated["rating"], pred), M.mae(rated["rating"], pred)
        else:
            res["rmse"] = res["mae"] = np.nan
        res["fit_seconds"] = entry.get("fit_seconds", np.nan)
        rows.append(res)

        for lo, hi, label in SEGMENTS:
            m = (activity >= lo) & (activity < hi)
            if m.any():
                seg_rows.append({"model": name, "segment": label, "users": int(m.sum()),
                                 f"ndcg_at_{kmax}": M.ndcg_at_k(recs[m], [relevant[j] for j in np.where(m)[0]], kmax),
                                 f"recall_at_{kmax}": M.recall_at_k(recs[m], [relevant[j] for j in np.where(m)[0]], kmax)})

        with mlflow.start_run(run_id=entry["run_id"]):
            mlflow.log_metrics({k: float(v) for k, v in res.items() if k != "model" and pd.notna(v)})
            mlflow.set_tag("evaluated_at", utc_now())
        log.info("  %-10s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f%s", name, kmax,
                 res[f"precision_at_{kmax}"], kmax, res[f"recall_at_{kmax}"], kmax, res[f"ndcg_at_{kmax}"],
                 res[f"coverage_at_{kmax}"], res[f"diversity_at_{kmax}"],
                 f" RMSE={res['rmse']:.4f} MAE={res['mae']:.4f}" if pd.notna(res["rmse"]) else "")

    results = pd.DataFrame(rows)
    gm = float(data.ratings.data.mean())
    reference = {"model": "global_mean (reference)", "rmse": M.rmse(rated["rating"], np.full(len(rated), gm)),
                 "mae": M.mae(rated["rating"], np.full(len(rated), gm))}
    segments = pd.DataFrame(seg_rows)
    results.to_csv(out_dir / "results.csv", index=False)
    segments.to_csv(out_dir / "results_by_user_segment.csv", index=False)

    split = read_json(pdir / "split_summary.json")
    summary = {
        "generated_at": utc_now(), "pipeline_run_id": pipeline_run_id(), "k_values": ks,
        "eval_users": int(len(eval_users)), "relevant_interactions": int(sum(map(len, relevant))),
        "rated_test_interactions": int(len(rated)), "cutoff_date": split["cutoff_date"],
        "results": results.to_dict("records"), "rating_reference": reference,
    }
    write_json(summary, out_dir / "results.json")
    _write_markdown(results, segments, reference, summary, ks, out_dir / "results.md")
    (out_dir / "sample_recommendations.md").write_text(
        _sample_recommendations(cfg, data, test, eval_users, recs_by_model, relevant))

    try:
        from src.evaluation.plots import plot_results
        plot_results(results, segments, kmax, out_dir)
    except Exception as exc:  # plotting must never fail the pipeline
        log.warning("Plotting skipped: %s", exc)

    with mlflow.start_run(run_name="evaluation-summary"):
        mlflow.set_tags({"stage": "evaluation", "pipeline_run_id": pipeline_run_id()})
        best = results.loc[results[f"ndcg_at_{kmax}"].idxmax()]
        mlflow.log_params({"eval_users": len(eval_users), "k_values": str(ks), "best_model": best["model"]})
        mlflow.log_artifacts(str(out_dir), artifact_path="evaluation")
    log.info("Best model by NDCG@%d: %s (%.4f)", kmax, best["model"], best[f"ndcg_at_{kmax}"])
    return summary


def _write_markdown(results, segments, reference, summary, ks, path: Path) -> None:
    lines = ["# Model Evaluation Results", "",
             f"Time-based split at **{pd.Timestamp(summary['cutoff_date']).date()}**. "
             f"{summary['eval_users']:,} warm test users, {summary['relevant_interactions']:,} relevant test "
             f"interactions (rating >= threshold), {summary['rated_test_interactions']:,} rated test interactions "
             "for RMSE/MAE. Train recipes already reviewed by the user are excluded from the ranking.", ""]
    for k in ks:
        cols = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"hit_rate_at_{k}",
                f"coverage_at_{k}", f"diversity_at_{k}", f"category_diversity_at_{k}", f"novelty_at_{k}"]
        t = results[cols].rename(columns=lambda c: c.replace(f"_at_{k}", f"@{k}"))
        lines += [f"## Ranking metrics @ {k}", "", df_to_markdown(t), ""]
    rt = results.loc[results["rmse"].notna(), ["model", "rmse", "mae"]]
    rt = pd.concat([rt, pd.DataFrame([reference])], ignore_index=True)
    lines += ["## Rating prediction", "", df_to_markdown(rt), "",
              "Item-kNN, content and hybrid produce ranking scores only, so RMSE/MAE are not applicable.", ""]
    if len(segments):
        lines += ["## By user activity (train-period review count)", "", df_to_markdown(segments), ""]
    lines += ["## Training time", "", df_to_markdown(results[["model", "fit_seconds"]], ".1f"), ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
