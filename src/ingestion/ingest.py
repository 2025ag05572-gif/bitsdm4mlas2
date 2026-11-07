"""Stage 1 - Data ingestion.

Reads the Food.com Kaggle export (RAW_recipes.csv, RAW_interactions.csv), either from the
downloaded zip or from extracted CSVs, and lands everything in data/raw as Parquet.

Design notes
* All columns are read as *strings*. Typing happens in validation, so badly typed values
  (e.g. rating="five") survive ingestion and can be caught and quarantined instead of
  crashing the reader or being silently coerced.
* Food.com has no user-profile table. A user dimension is reconstructed from every account
  that appears as a reviewer or as a recipe contributor.
* Lineage columns (_row_id, _source_file) are added so any record can be traced back to its
  source file and position.
* The data is landed exactly as delivered - nothing is modified, filtered or added here.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common import (
    ensure_dir,
    get_logger,
    load_config,
    set_seed,
    sha256_file,
    utc_now,
    write_json,
    write_parquet,
)

STAGE = "ingestion"


# --------------------------------------------------------------------------- source handling
def _download_from_kaggle(dataset: str, dest: Path, log) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # optional dependency
    except ImportError as exc:  # pragma: no cover - depends on user setup
        raise RuntimeError("source=kaggle needs `pip install kaggle` and ~/.kaggle/kaggle.json") from exc
    api = KaggleApi()
    api.authenticate()
    log.info("Downloading %s from Kaggle into %s", dataset, dest)
    api.dataset_download_files(dataset, path=str(dest), unzip=False, quiet=False)


def _locate_file(source_dir: Path, filename: str) -> tuple[Path, str | None]:
    """Return (path, zip_member). zip_member is None when the CSV sits directly on disk."""
    direct = list(source_dir.rglob(filename))
    if direct:
        return direct[0], None
    for zp in sorted(source_dir.glob("*.zip")):
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():
                if Path(member).name == filename:
                    return zp, member
    raise FileNotFoundError(
        f"{filename} not found in {source_dir}. Put the Kaggle zip (archive.zip) or the CSVs there."
    )


def _read_csv_as_strings(path: Path, member: str | None) -> pd.DataFrame:
    kwargs = dict(dtype=str, keep_default_na=False, na_values=[""])
    if member is None:
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
        return pd.read_csv(fh, **kwargs)


# --------------------------------------------------------------------------- user dimension
def build_user_dimension(recipes: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    """One row per account seen as reviewer or contributor."""
    rev = interactions[["user_id", "date"]].copy()
    rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
    rev_agg = rev.groupby("user_id").agg(first_review_date=("date", "min"), n_reviews=("date", "size"))

    con = recipes[["contributor_id", "submitted"]].rename(columns={"contributor_id": "user_id"}).copy()
    con["submitted"] = pd.to_datetime(con["submitted"], errors="coerce")
    con_agg = con.groupby("user_id").agg(
        first_contribution_date=("submitted", "min"), n_recipes_contributed=("submitted", "size")
    )

    users = rev_agg.join(con_agg, how="outer").reset_index()
    users["is_reviewer"] = users["n_reviews"].notna()
    users["is_contributor"] = users["n_recipes_contributed"].notna()
    users["n_reviews"] = users["n_reviews"].fillna(0).astype(int)
    users["n_recipes_contributed"] = users["n_recipes_contributed"].fillna(0).astype(int)
    users["member_since"] = users[["first_review_date", "first_contribution_date"]].min(axis=1)
    users = users[
        ["user_id", "member_since", "is_reviewer", "is_contributor", "n_reviews",
         "n_recipes_contributed", "first_review_date", "first_contribution_date"]
    ]
    # keep the raw layer string-typed, like the other two tables
    for c in ["member_since", "first_review_date", "first_contribution_date"]:
        users[c] = users[c].dt.strftime("%Y-%m-%d")
    return users.astype(str).replace({"NaT": np.nan, "nan": np.nan})


# --------------------------------------------------------------------------- main
def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    set_seed(cfg["project"]["seed"])
    icfg = cfg["ingestion"]
    source_dir = ensure_dir(cfg["paths"]["source_dir"])
    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])

    if icfg["source"] == "kaggle":
        _download_from_kaggle(icfg["kaggle_dataset"], source_dir, log)

    frames, sources = {}, {}
    for table, filename in icfg["files"].items():
        path, member = _locate_file(source_dir, filename)
        log.info("Reading %s from %s%s", table, path.name, f"::{member}" if member else "")
        df = _read_csv_as_strings(path, member)
        df.insert(0, "_row_id", np.arange(len(df)))
        df["_source_file"] = filename
        frames[table] = df
        sources[table] = {
            "file": str(path),
            "zip_member": member,
            "sha256": sha256_file(path),
            "rows": int(len(df)),
            "columns": [c for c in df.columns if not c.startswith("_")],
        }
        log.info("  %s: %d rows, %d columns", table, len(df), len(sources[table]["columns"]))

    recipes, interactions = frames["recipes"], frames["interactions"]
    users = build_user_dimension(recipes, interactions)
    log.info("Reconstructed user dimension: %d accounts", len(users))

    write_parquet(recipes, Path(raw_dir) / "recipes.parquet")
    write_parquet(interactions, Path(raw_dir) / "interactions.parquet")
    write_parquet(users, Path(raw_dir) / "users.parquet")

    manifest = {
        "stage": STAGE,
        "ingested_at": utc_now(),
        "source": icfg["source"],
        "kaggle_dataset": icfg["kaggle_dataset"],
        "sources": sources,
        "outputs": {
            "recipes": int(len(recipes)),
            "interactions": int(len(interactions)),
            "users": int(len(users)),
        },
    }
    write_json(manifest, Path(raw_dir) / "ingestion_manifest.json")
    log.info("Ingestion done -> %s", raw_dir)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
