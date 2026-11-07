"""Pandera schemas for the typed Food.com tables.

The raw layer is all strings; `validate.py` coerces each column first (recording type errors)
and then checks the typed frame against these schemas for nullability and value ranges.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


def interactions_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    lo, hi = pd.Timestamp(v["min_date"]), pd.Timestamp(v["max_date"])
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("Int64", nullable=False, checks=pa.Check.gt(0, name="positive_id")),
            "recipe_id": pa.Column("Int64", nullable=False, checks=pa.Check.gt(0, name="positive_id")),
            "date": pa.Column(
                "datetime64[ns]",
                nullable=False,
                checks=[pa.Check.ge(lo, name="date_after_min"), pa.Check.le(hi, name="date_before_max")],
            ),
            "rating": pa.Column(
                "float64", nullable=False, checks=pa.Check.isin(v["rating_values"], name="rating_in_allowed_set")
            ),
            "review": pa.Column(object, nullable=True),
        },
        strict=False,
        name="interactions",
    )


def recipes_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    lo, hi = pd.Timestamp(v["min_date"]), pd.Timestamp(v["max_date"])
    return pa.DataFrameSchema(
        {
            "id": pa.Column("Int64", nullable=False, checks=pa.Check.gt(0, name="positive_id")),
            "name": pa.Column(object, nullable=True),
            "minutes": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0, name="non_negative_minutes")),
            "contributor_id": pa.Column("Int64", nullable=False, checks=pa.Check.gt(0, name="positive_id")),
            "submitted": pa.Column(
                "datetime64[ns]",
                nullable=False,
                checks=[pa.Check.ge(lo, name="date_after_min"), pa.Check.le(hi, name="date_before_max")],
            ),
            "n_steps": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0, name="non_negative_n_steps")),
            "n_ingredients": pa.Column("Int64", nullable=False, checks=pa.Check.ge(1, name="at_least_one_ingredient")),
            "tags": pa.Column(object, nullable=False),
            "ingredients": pa.Column(object, nullable=False),
            "steps": pa.Column(object, nullable=False),
            "nutrition": pa.Column(object, nullable=False),
        },
        strict=False,
        name="recipes",
    )


def users_schema(cfg: dict) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column("Int64", nullable=False, unique=True, checks=pa.Check.gt(0, name="positive_id")),
            "member_since": pa.Column("datetime64[ns]", nullable=False),
            "n_reviews": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0, name="non_negative")),
            "n_recipes_contributed": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0, name="non_negative")),
        },
        strict=False,
        name="users",
    )


# Column -> coercion kind used before schema validation.
COERCION = {
    "interactions": {"user_id": "int", "recipe_id": "int", "date": "date", "rating": "float", "review": "str"},
    "recipes": {
        "id": "int",
        "name": "str",
        "minutes": "int",
        "contributor_id": "int",
        "submitted": "date",
        "n_steps": "int",
        "n_ingredients": "int",
        "tags": "str_list",
        "ingredients": "str_list",
        "steps": "str",
        "nutrition": "float_list",
        "description": "str",
    },
    "users": {
        "user_id": "int",
        "member_since": "date",
        "n_reviews": "int",
        "n_recipes_contributed": "int",
        "is_reviewer": "bool",
        "is_contributor": "bool",
        "first_review_date": "date",
        "first_contribution_date": "date",
    },
}

# Business columns used for exact-duplicate detection (lineage columns excluded).
DUPLICATE_KEYS = {
    "interactions": ["user_id", "recipe_id", "date", "rating", "review"],
    "recipes": ["id", "name", "minutes", "contributor_id", "submitted", "n_steps", "n_ingredients"],
    "users": ["user_id"],
}


# Rules implemented in validate.py on top of the schema (name -> (severity, description)).
CUSTOM_RULES = {
    "recipes": {
        "duplicate_row": ("error", "exact duplicate of an earlier row"),
        "duplicate_key:id": ("error", "recipe id already used by an earlier row"),
        "missing_name": ("warning", "recipe name is empty"),
        "missing_description": ("warning", "description is empty"),
        "implausible_minutes": ("warning", "minutes > validation.max_recipe_minutes (30 days)"),
        "zero_minutes": ("warning", "minutes == 0 (treated as unknown and imputed)"),
        "zero_steps": ("warning", "n_steps == 0"),
        "ingredient_count_mismatch": ("warning", "len(ingredients) != n_ingredients"),
        "extreme_calories": ("warning", "calories > validation.max_calories"),
        "empty_tags": ("warning", "tag list is empty"),
    },
    "users": {},
    "interactions": {
        "duplicate_row": ("error", "exact duplicate of an earlier row (user, recipe, date, rating, review)"),
        "referential_integrity:recipe_id": ("error", "recipe_id not found among valid recipes"),
        "referential_integrity:user_id": ("error", "user_id not found among valid users"),
        "rating_zero_no_stars": ("warning", "rating == 0, i.e. review written without a star rating"),
        "missing_review": ("warning", "review text is empty"),
        "invalid_timestamp:review_before_recipe_submitted": (
            "error", "review date is earlier than the recipe's submitted date (impossible ordering)"),
        "repeat_user_recipe_pair": ("warning", "user reviewed the same recipe more than once"),
    },
}
