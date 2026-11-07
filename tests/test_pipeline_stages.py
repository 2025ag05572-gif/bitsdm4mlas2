"""Stage-level tests on a tiny generated dataset in Food.com format."""
from pathlib import Path

import pandas as pd
import pytest

from scripts.make_sample_data import main as make_sample
from src.ingestion import ingest
from src.transformation.transform import k_core, time_split
from src.utils.common import load_config
from src.validation import validate


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    make_sample(root / "source", n_recipes=300, n_users=120, n_interactions=3000, seed=1)
    c = load_config("configs/config.sample.yaml")
    for k in ("source_dir", "raw_dir", "processed_dir", "features_dir", "errors_dir", "models_dir", "reports_dir"):
        c["paths"][k] = str(root / ("source" if k == "source_dir" else k))
    return c


def _raw_recipe(**overrides):
    row = {"_row_id": 0, "id": "1", "name": "soup", "minutes": "30", "contributor_id": "5",
           "submitted": "2005-01-01", "tags": "['soups-stews']", "nutrition": "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]",
           "n_steps": "3", "steps": "['a', 'b', 'c']", "description": "x", "ingredients": "['salt']",
           "n_ingredients": "1", "_source_file": "unit-test"}
    row.update(overrides)
    return row


def test_validation_rules_flag_bad_rows(cfg):
    """Hand-written rows for the validator unit test only - the pipeline itself never alters the data."""
    raw_r = pd.DataFrame([_raw_recipe(), _raw_recipe(_row_id=1, id="2", minutes="-5"),
                          _raw_recipe(_row_id=2, id="3", nutrition="[1.0]"), _raw_recipe(_row_id=3, id="abc"),
                          _raw_recipe(_row_id=4, id="5", submitted="2008-02-30"), _raw_recipe(_row_id=5)])
    typed_r, rules_r = validate.validate_recipes(raw_r, cfg)
    flagged = rules_r.frame().groupby("idx")["rule"].apply(set)
    assert "greater_than_or_equal_to:minutes" in flagged[1] or "non_negative_minutes:minutes" in flagged[1]
    assert "malformed_list:nutrition" in flagged[2]
    assert "type_error:id" in flagged[3]
    assert "type_error:submitted" in flagged[4]
    assert "duplicate_row" in flagged[5]
    assert 0 not in rules_r.error_index()

    recipes_ok = typed_r.drop(index=rules_r.error_index())
    users_ok = pd.DataFrame({"user_id": pd.array([7], dtype="Int64")})
    raw_i = pd.DataFrame({
        "_row_id": range(5), "_source_file": "unit-test",
        "user_id": ["7", "7", "7", "8", "7"], "recipe_id": ["1", "1", "999", "1", "1"],
        "date": ["2010-01-01", "2010-01-01", "2010-01-02", "2010-01-03", "2031-01-01"],
        "rating": ["5", "5", "4", "3", "9"], "review": ["ok", "ok", "ok", "ok", "ok"],
    })
    _, rules_i = validate.validate_interactions(raw_i, cfg, recipes_ok, users_ok)
    fi = rules_i.frame().groupby("idx")["rule"].apply(set)
    assert "duplicate_row" in fi[1]
    assert "referential_integrity:recipe_id" in fi[2]
    assert "referential_integrity:user_id" in fi[3]
    assert {"rating_in_allowed_set:rating", "date_before_max:date"} <= fi[4]
    assert 0 not in rules_i.error_index()


def test_ingest_and_validate_sample(cfg):
    ingest.run(cfg)
    report = validate.run(cfg)
    assert report["quality_gate"]["passed"]
    assert (Path(cfg["paths"]["errors_dir"]) / "interactions_errors.csv").exists()


def test_quality_gate_stops_the_pipeline(cfg):
    strict = {**cfg, "validation": {**cfg["validation"], "min_pass_rate": 1.01}}
    with pytest.raises(ValueError, match="quality gate"):
        validate.run(strict)


def test_time_split_has_no_leakage():
    df = pd.DataFrame({"user_id": range(10), "recipe_id": range(10),
                       "date": pd.date_range("2010-01-01", periods=10, freq="30D")})
    train, test, cutoff = time_split(df, 0.2)
    assert train["date"].max() < cutoff <= test["date"].min()
    assert len(train) + len(test) == len(df)


def test_k_core_is_stable():
    df = pd.DataFrame({"user_id": [1, 1, 1, 2, 2, 3], "recipe_id": [10, 11, 12, 10, 11, 12]})
    out = k_core(df, min_user=2, min_item=2)
    assert out["user_id"].value_counts().min() >= 2
    assert out["recipe_id"].value_counts().min() >= 2
