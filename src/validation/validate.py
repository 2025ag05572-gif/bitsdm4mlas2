"""Stage 2 - Data validation.

For each raw (string-typed) table:
  1. coerce every column to its target type, recording rows where coercion fails (type errors)
  2. validate the typed frame with a Pandera schema (nullability, ranges, allowed values)
  3. run custom checks: exact duplicates, duplicate keys, referential integrity
  4. run soft checks (warnings) that do not reject a row but are fixed/handled in preparation

Rows with at least one *error* are written to data/errors/<table>_errors.{parquet,csv} together
with the list of rules they broke. Valid rows are written typed to data/processed/validated/.
A JSON + Markdown validation report is written to reports/validation/.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from src.utils.common import (
    df_to_markdown,
    ensure_dir,
    get_logger,
    load_config,
    utc_now,
    write_json,
    write_parquet,
)
from src.validation.schemas import (
    COERCION,
    CUSTOM_RULES,
    DUPLICATE_KEYS,
    interactions_schema,
    recipes_schema,
    users_schema,
)

STAGE = "validation"
SCHEMAS = {"recipes": recipes_schema, "users": users_schema, "interactions": interactions_schema}


class RuleLog:
    """Collects (row index, rule, severity) triples."""

    def __init__(self) -> None:
        self._parts: list[pd.DataFrame] = []

    def add(self, mask_or_index, rule: str, severity: str) -> None:
        idx = mask_or_index[mask_or_index].index if isinstance(mask_or_index, pd.Series) else pd.Index(mask_or_index)
        if len(idx):
            self._parts.append(pd.DataFrame({"idx": idx, "rule": rule, "severity": severity}))

    def frame(self) -> pd.DataFrame:
        if not self._parts:
            return pd.DataFrame(columns=["idx", "rule", "severity"])
        return pd.concat(self._parts, ignore_index=True).drop_duplicates()

    def error_index(self) -> pd.Index:
        f = self.frame()
        return pd.Index(f.loc[f["severity"] == "error", "idx"].unique())


# --------------------------------------------------------------------------- coercion
def _parse_list(value, elem_type):
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return None
    try:
        out = ast.literal_eval(v)  # literal_eval only evaluates Python literals - safe on untrusted text
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(out, list):
        return None
    try:
        return [elem_type(x) for x in out]
    except (TypeError, ValueError):
        return None


def coerce(raw: pd.DataFrame, table: str, rules: RuleLog, cfg: dict) -> pd.DataFrame:
    typed = raw.copy()
    for col, kind in COERCION[table].items():
        if col not in raw.columns:
            rules.add(raw.index, f"missing_column:{col}", "error")
            continue
        s = raw[col]
        present = s.notna()
        if kind == "int":
            num = pd.to_numeric(s.str.strip(), errors="coerce")
            bad = present & (num.isna() | (num % 1 != 0))
            typed[col] = num.where(~bad).astype("Int64")
        elif kind == "float":
            num = pd.to_numeric(s.str.strip(), errors="coerce")
            bad = present & num.isna()
            typed[col] = num.astype("float64")
        elif kind == "date":
            dt = pd.to_datetime(s.str.strip(), format="%Y-%m-%d", errors="coerce")
            bad = present & dt.isna()
            typed[col] = dt
        elif kind == "bool":
            mapped = s.map({"True": True, "False": False})
            bad = present & mapped.isna()
            typed[col] = mapped.astype("boolean")
        elif kind in ("str_list", "float_list"):
            elem = str if kind == "str_list" else float
            parsed = s.map(lambda v: _parse_list(v, elem))
            bad = present & parsed.isna()
            if kind == "float_list":
                n = cfg["validation"]["nutrition_length"]
                wrong_len = parsed.map(lambda x: x is not None and len(x) != n)
                bad = bad | wrong_len
            parsed = parsed.where(~bad, None)
            typed[col] = parsed
            rules.add(bad, f"malformed_list:{col}", "error")
            continue
        else:  # str
            typed[col] = s
            continue
        rules.add(bad, f"type_error:{col}", "error")
    return typed


def apply_schema(typed: pd.DataFrame, schema: pa.DataFrameSchema, rules: RuleLog) -> None:
    try:
        schema.validate(typed, lazy=True)
        return
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
    already = rules.frame()
    type_err = set(zip(already["idx"], already["rule"].str.split(":").str[-1]))
    fc = fc[fc["index"].notna()].copy()
    fc["rule"] = [_rule_name(schema, col, chk, num) for col, chk, num in zip(fc["column"], fc["check"], fc["check_number"])]
    for (rule, column), grp in fc.groupby(["rule", "column"]):
        idx = [i for i in grp["index"].astype(int) if not (rule.startswith("not_null") and (i, column) in type_err)]
        rules.add(idx, rule, "error")


def _rule_name(schema: pa.DataFrameSchema, column: str, check: str, check_number) -> str:
    if check == "not_nullable":
        return f"not_null:{column}"
    if check == "field_uniqueness":
        return f"unique:{column}"
    try:
        return f"{schema.columns[column].checks[int(check_number)].name}:{column}"
    except (KeyError, IndexError, TypeError, ValueError):
        return f"{check.split('(')[0]}:{column}"


def rule_catalog(table: str, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Every rule applied to a table (so the report also lists rules with zero violations)."""
    rows = []
    for col, kind in COERCION[table].items():
        if kind in ("int", "float", "date", "bool"):
            rows.append((f"type_error:{col}", "error", f"{col} is present but cannot be parsed as {kind}"))
        elif kind.endswith("list"):
            rows.append((f"malformed_list:{col}", "error", f"{col} is not a valid list literal"
                         + (" of 7 numbers" if kind == "float_list" else "")))
    for col, c in schema.columns.items():
        if not c.nullable:
            rows.append((f"not_null:{col}", "error", f"{col} is missing"))
        if c.unique:
            rows.append((f"unique:{col}", "error", f"{col} must be unique"))
        for ch in c.checks:
            rows.append((f"{ch.name}:{col}", "error", f"{col} must satisfy {ch.error}"))
    rows += [(name, sev, desc) for name, (sev, desc) in CUSTOM_RULES[table].items()]
    return pd.DataFrame(rows, columns=["rule", "severity", "description"]).drop_duplicates("rule")


def _not_errored(rules: RuleLog, index: pd.Index) -> pd.Series:
    return pd.Series(~index.isin(rules.error_index()), index=index)


# --------------------------------------------------------------------------- per table
def validate_recipes(raw: pd.DataFrame, cfg: dict):
    rules = RuleLog()
    typed = coerce(raw, "recipes", rules, cfg)
    apply_schema(typed, recipes_schema(cfg), rules)

    ok = _not_errored(rules, typed.index)
    exact_dup = ok & typed.duplicated(subset=DUPLICATE_KEYS["recipes"], keep="first")
    rules.add(exact_dup, "duplicate_row", "error")
    ok = _not_errored(rules, typed.index)
    rules.add(typed.loc[ok, "id"].duplicated(keep="first"), "duplicate_key:id", "error")

    v = cfg["validation"]
    rules.add(typed["name"].isna(), "missing_name", "warning")
    rules.add(typed["description"].isna(), "missing_description", "warning")
    rules.add(typed["minutes"].fillna(0) > v["max_recipe_minutes"], "implausible_minutes", "warning")
    rules.add(typed["minutes"].fillna(-1) == 0, "zero_minutes", "warning")
    rules.add(typed["n_steps"].fillna(-1) == 0, "zero_steps", "warning")
    n_ing = typed["ingredients"].map(lambda x: len(x) if isinstance(x, list) else np.nan)
    rules.add((n_ing != typed["n_ingredients"].astype("float")) & n_ing.notna(), "ingredient_count_mismatch", "warning")
    cal = typed["nutrition"].map(lambda x: x[0] if isinstance(x, list) and x else np.nan)
    rules.add(cal > v["max_calories"], "extreme_calories", "warning")
    rules.add(typed["tags"].map(lambda x: isinstance(x, list) and len(x) == 0), "empty_tags", "warning")
    return typed, rules


def validate_users(raw: pd.DataFrame, cfg: dict):
    rules = RuleLog()
    typed = coerce(raw, "users", rules, cfg)
    apply_schema(typed, users_schema(cfg), rules)
    return typed, rules


def validate_interactions(raw: pd.DataFrame, cfg: dict, recipes_ok: pd.DataFrame, users_ok: pd.DataFrame):
    rules = RuleLog()
    typed = coerce(raw, "interactions", rules, cfg)
    apply_schema(typed, interactions_schema(cfg), rules)

    ok = _not_errored(rules, typed.index)
    rules.add(ok & typed.duplicated(subset=DUPLICATE_KEYS["interactions"], keep="first"), "duplicate_row", "error")

    ok = _not_errored(rules, typed.index)
    rules.add(ok & ~typed["recipe_id"].isin(recipes_ok["id"]), "referential_integrity:recipe_id", "error")
    rules.add(ok & ~typed["user_id"].isin(users_ok["user_id"]), "referential_integrity:user_id", "error")

    submitted = typed["recipe_id"].map(recipes_ok.set_index("id")["submitted"])
    rules.add(ok & (typed["date"] < submitted), "invalid_timestamp:review_before_recipe_submitted", "error")

    rules.add(typed["rating"] == 0, "rating_zero_no_stars", "warning")
    rules.add(typed["review"].isna(), "missing_review", "warning")
    ok = _not_errored(rules, typed.index)
    rules.add(typed.loc[ok, ["user_id", "recipe_id"]].duplicated(keep=False), "repeat_user_recipe_pair", "warning")
    return typed, rules


# --------------------------------------------------------------------------- outputs
def _split_and_save(table, raw, typed, rules: RuleLog, cfg, log) -> tuple[pd.DataFrame, dict]:
    frame = rules.frame()
    err_idx = rules.error_index()
    valid = typed.drop(index=err_idx)

    errors = raw.loc[err_idx].copy()
    if len(errors):
        reasons = frame[frame["severity"] == "error"].groupby("idx")["rule"].apply(lambda r: ";".join(sorted(r)))
        errors["error_rules"] = reasons.reindex(errors.index).values
    else:
        errors["error_rules"] = pd.Series(dtype=str)
    errors["validated_at"] = utc_now()

    errors_dir = ensure_dir(cfg["paths"]["errors_dir"])
    write_parquet(errors, errors_dir / f"{table}_errors.parquet")
    errors.to_csv(errors_dir / f"{table}_errors.csv", index=False)
    warn = frame[frame["severity"] == "warning"].copy()
    warn["_row_id"] = raw.loc[warn["idx"], "_row_id"].values if len(warn) else []
    write_parquet(warn[["_row_id", "rule"]], errors_dir / f"{table}_warnings.parquet")

    out_dir = ensure_dir(Path(cfg["paths"]["processed_dir"]) / "validated")
    write_parquet(valid, out_dir / f"{table}.parquet")

    counts = frame.groupby("rule").size()
    rule_counts = rule_catalog(table, SCHEMAS[table](cfg))
    extra = sorted(set(counts.index) - set(rule_counts["rule"]))
    if extra:
        rule_counts = pd.concat([rule_counts, pd.DataFrame({"rule": extra, "severity": "error", "description": ""})])
    rule_counts["rows"] = rule_counts["rule"].map(counts).fillna(0).astype(int)
    rule_counts["pct_of_rows"] = 100 * rule_counts["rows"] / max(len(raw), 1)
    rule_counts = rule_counts.sort_values(["severity", "rows"], ascending=[True, False])
    summary = {
        "rows_in": int(len(raw)),
        "rows_valid": int(len(valid)),
        "rows_quarantined": int(len(err_idx)),
        "pass_rate": float(len(valid) / max(len(raw), 1)),
        "rows_with_warnings": int(warn["idx"].nunique()) if len(warn) else 0,
        "rules": rule_counts.to_dict("records"),
    }
    log.info("%s: %d in -> %d valid, %d quarantined, %d rows with warnings",
             table, summary["rows_in"], summary["rows_valid"], summary["rows_quarantined"], summary["rows_with_warnings"])
    return valid, summary


def _write_markdown(report: dict, path: Path) -> None:
    lines = ["# Data Validation Report", "", f"Generated: {report['generated_at']}", "",
             "Rows breaking an **error** rule are quarantined to `data/errors/<table>_errors.csv`. "
             "**Warning** rules keep the row and are handled in the preparation stage.", "",
             "## Summary", ""]
    summ = pd.DataFrame([
        {"table": t, "rows_in": s["rows_in"], "valid": s["rows_valid"], "quarantined": s["rows_quarantined"],
         "pass_rate": s["pass_rate"], "rows_with_warnings": s["rows_with_warnings"]}
        for t, s in report["tables"].items()
    ])
    lines += [df_to_markdown(summ), ""]
    g = report.get("quality_gate", {})
    lines += [f"Quality gate (every table pass rate >= {g.get('min_pass_rate')}): "
              f"**{'PASSED' if g.get('passed') else 'FAILED'}**", ""]
    for t, s in report["tables"].items():
        lines += [f"## {t}", "", f"{len(s['rules'])} rules checked.", ""]
        rc = pd.DataFrame(s["rules"]).rename(columns={"pct_of_rows": "% of rows"})
        lines += [df_to_markdown(rc, ".3f"), ""]
    path.write_text("\n".join(lines))


def run(cfg: dict) -> dict:
    log = get_logger(STAGE, cfg)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raws = {t: pd.read_parquet(raw_dir / f"{t}.parquet") for t in ("recipes", "users", "interactions")}

    report = {"stage": STAGE, "generated_at": utc_now(), "tables": {}}
    typed_r, rules_r = validate_recipes(raws["recipes"], cfg)
    recipes_ok, report["tables"]["recipes"] = _split_and_save("recipes", raws["recipes"], typed_r, rules_r, cfg, log)

    typed_u, rules_u = validate_users(raws["users"], cfg)
    users_ok, report["tables"]["users"] = _split_and_save("users", raws["users"], typed_u, rules_u, cfg, log)

    typed_i, rules_i = validate_interactions(raws["interactions"], cfg, recipes_ok, users_ok)
    _, report["tables"]["interactions"] = _split_and_save("interactions", raws["interactions"], typed_i, rules_i, cfg, log)

    gate = cfg["validation"].get("min_pass_rate", 0.0)
    failing = {t: s["pass_rate"] for t, s in report["tables"].items() if s["pass_rate"] < gate}
    report["quality_gate"] = {"min_pass_rate": gate, "passed": not failing, "failing_tables": failing}
    rep_dir = ensure_dir(Path(cfg["paths"]["reports_dir"]) / "validation")
    write_json(report, rep_dir / "validation_report.json")
    _write_markdown(report, rep_dir / "validation_report.md")
    if failing:
        raise ValueError(f"Data quality gate failed (pass rate < {gate}): {failing}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
