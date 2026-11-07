"""Shared helpers: config loading, logging, paths and small I/O utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve every entry under `paths` to an absolute path."""
    path = Path(path or os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "base" in cfg:  # small override files, e.g. configs/config.sample.yaml
        with open(path.parent / cfg.pop("base")) as fh:
            cfg = _deep_merge(yaml.safe_load(fh), cfg)
    cfg["paths"] = {k: str((PROJECT_ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_logger(name: str, cfg: dict | None = None) -> logging.Logger:
    """Logger that writes to stdout and to logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logs_dir = Path(cfg["paths"]["logs_dir"]) if cfg else PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / "pipeline.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(obj: Any, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_parquet(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small markdown table renderer (avoids the optional `tabulate` dependency)."""
    cols = list(df.columns)

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def setup_mlflow(cfg: dict):
    """Point MLflow at the project-local tracking store and select the experiment."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(PROJECT_ROOT / uri[len("sqlite:///"):])
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(name)
    return mlflow


def pipeline_run_id() -> str:
    """Id shared by every stage of one pipeline execution (set by the orchestrator)."""
    rid = os.environ.get("PIPELINE_RUN_ID")
    if not rid:
        rid = datetime.now(timezone.utc).strftime("manual-%Y%m%dT%H%M%S")
        os.environ["PIPELINE_RUN_ID"] = rid
    return rid
