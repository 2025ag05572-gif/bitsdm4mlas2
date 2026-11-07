"""Run every stage in order without an orchestrator (same code the Prefect flow calls).

    python -m src.pipeline [--config configs/config.yaml] [--from-stage features]
"""
from __future__ import annotations

import argparse
import time

from src.evaluation import evaluate
from src.features import build as features
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train, tune
from src.transformation import transform
from src.validation import validate
from src.utils.common import get_logger, load_config, pipeline_run_id

STAGES = [
    ("ingestion", ingest.run),
    ("validation", validate.run),
    ("preparation", prepare.run),
    ("transformation", transform.run),
    ("features", features.run),
    ("tuning", lambda cfg: tune.run(cfg) if cfg["training"]["tuning"]["enabled"] else None),
    ("training", train.run),
    ("evaluation", evaluate.run),
]


def main(config: str | None = None, from_stage: str | None = None) -> None:
    cfg = load_config(config)
    log = get_logger("pipeline", cfg)
    log.info("Pipeline run %s with %s", pipeline_run_id(), cfg["_config_path"])
    names = [n for n, _ in STAGES]
    start = names.index(from_stage) if from_stage else 0
    t_all = time.time()
    for name, fn in STAGES[start:]:
        t0 = time.time()
        log.info("=== %s ===", name)
        fn(cfg)
        log.info("=== %s finished in %.1fs ===", name, time.time() - t0)
    log.info("Pipeline finished in %.1fs", time.time() - t_all)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--from-stage", default=None, choices=[n for n, _ in STAGES])
    a = ap.parse_args()
    main(a.config, a.from_stage)
