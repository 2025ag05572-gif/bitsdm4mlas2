"""Prefect orchestration of the Food.com recommendation pipeline.

DAG
    ingest -> validate (quality gate) -> prepare -> transform -> build_features -> tune
           -> train[popularity | item_knn | content | mf | hybrid] -> evaluate -> publish_report

Every task wraps the `run()` function of one stage module, so the orchestrated run and
`python -m src.pipeline` execute exactly the same code. Stages exchange data through files in
data/ (Parquet / NPZ), and each task's return value carries a small summary, which is what Prefect
uses to draw the dependency edges.

Usage
    python dags/recommendation_pipeline.py                       # one run (uses PREFECT_API_URL if set)
    python dags/recommendation_pipeline.py --config configs/config.sample.yaml
    python dags/recommendation_pipeline.py --serve               # register a weekly schedule and serve it
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact, create_table_artifact  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402

from src.evaluation import evaluate  # noqa: E402
from src.features import build as features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train, tune  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.utils.common import load_config  # noqa: E402
from src.validation import validate  # noqa: E402


@task(name="ingest", retries=2, retry_delay_seconds=15)
def ingest_task(cfg: dict) -> dict:
    m = ingest.run(cfg)
    return {"outputs": m["outputs"], "sha256": {t: v["sha256"][:12] for t, v in m["sources"].items()}}


@task(name="validate")
def validate_task(cfg: dict, ingested: dict) -> dict:
    rep = validate.run(cfg)  # raises if the quality gate fails -> downstream tasks do not run
    rows = [{"table": t, "rows_in": s["rows_in"], "valid": s["rows_valid"], "quarantined": s["rows_quarantined"],
             "pass_rate": round(s["pass_rate"], 4)} for t, s in rep["tables"].items()]
    create_table_artifact(key="validation-summary", table=rows, description="Validation pass rates per table")
    return {"tables": rows, "quality_gate": rep["quality_gate"]}


@task(name="prepare")
def prepare_task(cfg: dict, validated: dict) -> dict:
    s = prepare.run(cfg)
    return {k: s[k] for k in ("recipes", "users", "interactions", "interactions_rated")}


@task(name="transform")
def transform_task(cfg: dict, prepared: dict) -> dict:
    s = transform.run(cfg)
    return {k: s[k] for k in ("cutoff_date", "train_interactions", "train_users", "train_items",
                              "test_interactions_warm")}


@task(name="build-features")
def features_task(cfg: dict, split: dict) -> dict:
    return features.run(cfg)


@task(name="tune")
def tune_task(cfg: dict, feature_meta: dict) -> dict:
    if not cfg["training"]["tuning"]["enabled"]:
        return {"skipped": True}
    best = tune.run(cfg)
    create_markdown_artifact(key="tuning-results",
                             markdown=(Path(cfg["paths"]["reports_dir"]) / "tuning" / "tuning_results.md").read_text())
    return best


@task(name="train")
def train_task(cfg: dict, model_name: str, tuned: dict) -> dict:
    return {"model": model_name, **train.train_model(model_name, cfg)}


@task(name="evaluate")
def evaluate_task(cfg: dict, trained: list[dict]) -> dict:
    s = evaluate.run(cfg)
    k = max(s["k_values"])
    keep = ["model", f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", f"coverage_at_{k}",
            f"diversity_at_{k}", "rmse", "mae"]
    table = [{c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in keep} for r in s["results"]]
    create_table_artifact(key="model-comparison", table=table, description=f"Ranking metrics @ {k}")
    return {"k": k, "results": table}


@task(name="publish-report")
def publish_task(cfg: dict, evaluated: dict) -> str:
    md = (Path(cfg["paths"]["reports_dir"]) / "evaluation" / "results.md").read_text()
    create_markdown_artifact(key="evaluation-report", markdown=md, description="Model evaluation results")
    return md


@flow(name="foodcom-recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None) -> dict:
    os.environ["PIPELINE_RUN_ID"] = str(flow_run.id or "local")
    cfg = load_config(config_path)
    logger = get_run_logger()
    logger.info("Config: %s", cfg["_config_path"])

    ingested = ingest_task(cfg)
    validated = validate_task(cfg, ingested)
    prepared = prepare_task(cfg, validated)
    split = transform_task(cfg, prepared)
    feats = features_task(cfg, split)
    tuned = tune_task(cfg, feats)
    trained = [train_task.with_options(task_run_name=f"train-{m}")(cfg, m, tuned)
               for m in train.enabled_models(cfg)]
    evaluated = evaluate_task(cfg, trained)
    publish_task(cfg, evaluated)
    return evaluated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--serve", action="store_true", help="serve a deployment with a weekly cron schedule")
    a = ap.parse_args()
    if a.serve:
        recommendation_pipeline.serve(name="weekly-retrain", cron="0 3 * * 1",
                                      parameters={"config_path": a.config})
    else:
        recommendation_pipeline(a.config)
