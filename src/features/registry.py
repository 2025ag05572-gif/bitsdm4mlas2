"""Feature registry: the single place where every feature is defined and documented.

The registry drives three things:
* which columns `build.py` must produce (it fails if a registered feature is missing),
* the generated feature catalog (reports/feature_catalog.md),
* look-ups through `FeatureStore`, which only serves registered features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    entity: str          # user | item | user_item
    dtype: str
    description: str
    transformation: str
    source: str
    used_by: str = "-"


U, I, UI = "user", "item", "user_item"
INTER, REC, USR = "interactions", "recipes", "users"

FEATURES: list[Feature] = [
    # ------------------------------------------------------------------ user features
    Feature("user_activity_count", U, "int", "Number of recipes the user reviewed before as_of",
            "count(interactions) per user", INTER, "hybrid (cold-user gating), eval segments"),
    Feature("user_rated_count", U, "int", "Number of reviews that carry a 1-5 star rating",
            "count(rating not null)", INTER),
    Feature("user_avg_rating", U, "float", "Mean star rating given by the user (global mean if none)",
            "mean(rating), fillna(global mean)", INTER, "mf cold-start fallback"),
    Feature("user_rating_std", U, "float", "Std-dev of the user's ratings (0 if < 2 ratings)",
            "std(rating), fillna(0)", INTER),
    Feature("user_days_since_last_interaction", U, "int", "Days between the user's last review and as_of",
            "as_of - max(date)", INTER),
    Feature("user_tenure_days", U, "int", "Days since the user first appeared (review or recipe upload)",
            "as_of - min(member_since, first interaction)", f"{INTER}, {USR}"),
    Feature("user_interactions_last_365d", U, "int", "Reviews written in the 365 days before as_of",
            "count(date >= as_of - 365d)", INTER),
    Feature("user_preferred_category", U, "category", "Most frequent recipe category the user reviewed",
            "mode(recipe.category)", f"{INTER}, {REC}", "hybrid (category boost)"),
    Feature("user_preferred_category_share", U, "float", "Share of the user's reviews in the preferred category",
            "count(mode)/count(*)", f"{INTER}, {REC}"),
    Feature("user_category_entropy", U, "float", "Shannon entropy (bits) of the user's category distribution",
            "-sum(p log2 p) over categories", f"{INTER}, {REC}"),
    Feature("user_preferred_cuisine", U, "category", "Most frequent cuisine tag among reviewed recipes",
            "mode(recipe.cuisine)", f"{INTER}, {REC}"),
    Feature("user_avg_recipe_minutes", U, "float", "Average preparation time of recipes the user reviewed",
            "mean(recipe.minutes)", f"{INTER}, {REC}"),
    Feature("user_avg_recipe_calories", U, "float", "Average calories of recipes the user reviewed",
            "mean(recipe.calories)", f"{INTER}, {REC}"),
    Feature("user_avg_review_length", U, "float", "Average number of words per review",
            "mean(review_length)", INTER),
    Feature("user_recipes_contributed", U, "int", "Recipes the user uploaded before as_of",
            "count(recipes where contributor_id = user)", REC),
    # ------------------------------------------------------------------ item features
    Feature("item_popularity", I, "int", "Number of reviews of the recipe before as_of",
            "count(interactions) per recipe", INTER, "popularity, hybrid"),
    Feature("item_rated_count", I, "int", "Number of star-rated reviews", "count(rating not null)", INTER),
    Feature("item_avg_rating", I, "float", "Mean star rating of the recipe (NaN if never rated)",
            "mean(rating)", INTER),
    Feature("item_bayesian_rating", I, "float", "Average rating shrunk towards the global mean",
            "(n*avg + m*global)/(n+m), m = features.bayesian_prior_weight", INTER, "popularity (rating prediction)"),
    Feature("item_recent_popularity", I, "int", "Reviews in the 365 days before as_of",
            "count(date >= as_of - 365d)", INTER),
    Feature("item_days_since_last_interaction", I, "float", "Days since the recipe was last reviewed (NaN if never)",
            "as_of - max(date)", INTER),
    Feature("item_age_days", I, "int", "Days since the recipe was submitted", "as_of - submitted", REC),
    Feature("item_category", I, "category", "Primary course category derived from tags",
            "first tag found in preparation.category_priority", REC, "hybrid, diversity analysis"),
    Feature("item_cuisine", I, "category", "Cuisine derived from tags", "first tag in preparation.cuisine_tags", REC),
    Feature("item_minutes", I, "float", "Preparation time in minutes (implausible values imputed)",
            "clean minutes", REC),
    Feature("item_n_steps", I, "int", "Number of preparation steps", "n_steps", REC),
    Feature("item_n_ingredients", I, "int", "Number of ingredients", "n_ingredients", REC),
    Feature("item_calories", I, "float", "Calories per serving (implausible values imputed)", "nutrition[0]", REC),
    Feature("item_n_tags", I, "int", "Number of distinct tags", "len(tags)", REC),
    Feature("item_content_tfidf", I, "sparse vector", "TF-IDF vector over tags and ingredients",
            "TfidfVectorizer(tags + ingredients, min_df, max_df)", REC, "content, hybrid, diversity metric"),
    # ------------------------------------------------------------- interaction features
    Feature("ui_rating_deviation_user", UI, "float", "Rating minus the user's average rating",
            "rating - user_avg_rating", INTER),
    Feature("ui_rating_deviation_item", UI, "float", "Rating minus the recipe's average rating",
            "rating - item_avg_rating", INTER),
    Feature("ui_category_match", UI, "bool", "Recipe category equals the user's preferred category",
            "item_category == user_preferred_category", f"{INTER}, {REC}"),
    Feature("ui_recipe_age_at_interaction_days", UI, "int", "Age of the recipe when it was reviewed",
            "date - submitted", f"{INTER}, {REC}"),
    Feature("ui_user_sequence_number", UI, "int", "n-th review of this user (chronological, 1-based)",
            "cumcount per user ordered by date", INTER),
    Feature("ui_days_since_prev_interaction", UI, "float", "Days since the user's previous review (NaN for first)",
            "date - lag(date) per user", INTER),
]


def by_entity(entity: str) -> list[Feature]:
    return [f for f in FEATURES if f.entity == entity]


def as_records() -> list[dict]:
    return [asdict(f) for f in FEATURES]
