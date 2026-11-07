"""Stage 6 - Model training with MLflow experiment tracking.

For every enabled model in configs/config.yaml:
  * fit on the train interaction matrix + training feature view
  * log parameters, data statistics, fit time (and the MF learning curve) to MLflow
  * save the fitted model to models/<name>.joblib and log it, the config, the split summary and
    the feature-view metadata as MLflow artifacts
  * record the MLflow run id in models/registry.json so evaluation can log metrics to the same run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.training.models import TrainData, build_model, load_train_data, tuned_params
from src.utils.common import (
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

STAGE = "training"


def enabled_models(cfg: dict) -> list[str]:
    return [m for m, c in cfg["training"]["models"].items() if c.get("enabled", True)]


def _update_registry(cfg: dict, name: str, entry: dict) -> None:
    path = Path(cfg["paths"]["models_dir"]) / "registry.json"
    reg = read_json(path) if path.exists() else {}
    reg[name] = entry
    write_json(reg, path)


def train_model(name: str, cfg: dict, data: TrainData | None = None) -> dict:
    log = get_logger(STAGE, cfg)
    set_seed(cfg["project"]["seed"])
    mlflow = setup_mlflow(cfg)
    data = data or load_train_data(cfg)
    pdir = Path(cfg["paths"]["processed_dir"])
    split = read_json(pdir / "split_summary.json")
    model = build_model(name, cfg)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model": name, "stage": "training", "pipeline_run_id": pipeline_run_id(),
                         "dataset": "food.com (Kaggle shuyangli94)", "feature_view": "training",
                         "params_source": "tuned (models/best_params.json)" if name in tuned_params(cfg) else "config"})
        params = {f"model.{k}": v for k, v in model.params.items()}
        if name == "hybrid":
            params.update({f"model.knn.{k}": v for k, v in model.knn_params.items()})
        params.update({
            "data.cutoff_date": split["cutoff_date"], "data.train_users": data.n_users,
            "data.train_items": data.n_items, "data.train_interactions": int(data.R.nnz),
            "data.train_ratings": int(data.ratings.nnz), "data.relevance_threshold": split["relevance_threshold"],
            "data.min_user_interactions": cfg["transformation"]["min_user_interactions"],
            "data.min_item_interactions": cfg["transformation"]["min_item_interactions"],
            "seed": cfg["project"]["seed"],
        })
        mlflow.log_params(params)
        log.info("Training %s ...", name)
        model.fit(data)
        mlflow.log_metric("fit_seconds", model.fit_seconds)
        for step, v in enumerate(getattr(model, "history_", [])):
            mlflow.log_metric("train_rmse", v, step=step)

        path = ensure_dir(cfg["paths"]["models_dir"]) / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        mlflow.log_artifact(str(path), artifact_path="model")
        mlflow.log_artifact(cfg["_config_path"], artifact_path="config")
        mlflow.log_artifact(str(pdir / "split_summary.json"), artifact_path="data")
        mlflow.log_artifact(str(Path(cfg["paths"]["features_dir"]) / "training" / "_metadata.json"),
                            artifact_path="features")
        entry = {"run_id": run.info.run_id, "path": str(path), "trained_at": utc_now(),
                 "fit_seconds": round(model.fit_seconds, 2), "pipeline_run_id": pipeline_run_id()}
    _update_registry(cfg, name, entry)
    log.info("  %s done in %.1fs (MLflow run %s)", name, model.fit_seconds, entry["run_id"])
    return entry


def run(cfg: dict, models: list[str] | None = None) -> dict:
    data = load_train_data(cfg)
    return {m: train_model(m, cfg, data) for m in (models or enabled_models(cfg))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.models)
