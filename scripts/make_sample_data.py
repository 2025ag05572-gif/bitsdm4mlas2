"""Generate a small synthetic sample in the exact Food.com Kaggle format (for smoke tests / CI).

Users get a latent preference over recipe categories so personalised models have signal to learn.
Usage: python -m scripts.make_sample_data --out data/sample/source
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = ["desserts", "main-dish", "side-dishes", "appetizers", "breakfast", "salads", "soups-stews", "breads", "beverages"]
CUISINES = ["american", "italian", "mexican", "asian", "indian", "greek", "french"]
INGREDIENTS = ["salt", "butter", "sugar", "flour", "eggs", "garlic", "onion", "milk", "olive oil", "pepper",
               "chicken", "beef", "rice", "tomato", "cheese", "lemon", "basil", "cumin", "ginger", "potato"]


def main(out: Path, n_recipes: int, n_users: int, n_interactions: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    cats = rng.choice(len(CATEGORIES), n_recipes)
    submitted = pd.to_datetime("2000-06-01") + pd.to_timedelta(rng.integers(0, 15 * 365, n_recipes), unit="D")
    recipes = []
    for i in range(n_recipes):
        tags = ["time-to-make", CATEGORIES[cats[i]], str(rng.choice(CUISINES))] + [
            str(t) for t in rng.choice(["easy", "healthy", "vegetarian", "low-fat", "kid-friendly"], 2, replace=False)]
        ings = [str(x) for x in rng.choice(INGREDIENTS, int(rng.integers(3, 10)), replace=False)]
        steps = [f"step {k + 1}" for k in range(int(rng.integers(2, 12)))]
        nutrition = [round(float(rng.gamma(4, 90)), 1)] + [round(float(x), 1) for x in rng.gamma(2, 12, 6)]
        recipes.append({
            "name": f"sample recipe {i}",
            "id": 1000 + i,
            "minutes": int(rng.choice([10, 15, 20, 30, 45, 60, 90, 120, 240])),
            "contributor_id": int(rng.integers(1, n_users + 50)),
            "submitted": submitted[i].strftime("%Y-%m-%d"),
            "tags": str(tags),
            "nutrition": str(nutrition),
            "n_steps": len(steps),
            "steps": str(steps),
            "description": "" if rng.random() < 0.03 else f"a tasty {CATEGORIES[cats[i]]} dish",
            "ingredients": str(ings),
            "n_ingredients": len(ings),
        })
    recipes = pd.DataFrame(recipes)

    pop = rng.zipf(1.6, n_recipes).astype(float)
    pref = rng.dirichlet(np.full(len(CATEGORIES), 0.3), n_users)
    activity = np.clip(rng.zipf(1.7, n_users), 1, 300).astype(float)
    activity = activity / activity.sum()
    start = pd.to_datetime("2002-01-01") + pd.to_timedelta(rng.integers(0, 13 * 365, n_users), unit="D")

    rows = []
    users = rng.choice(n_users, n_interactions, p=activity)
    for u in users:
        c = rng.choice(len(CATEGORIES), p=pref[u])
        cand = np.where(cats == c)[0]
        r = int(rng.choice(cand, p=pop[cand] / pop[cand].sum()))
        base = max(start[u], submitted[r])
        date = min(base + pd.Timedelta(days=int(rng.integers(0, 4 * 365))), pd.Timestamp("2018-12-20"))
        rating = int(rng.choice([5, 4, 3, 2, 1, 0], p=[0.70, 0.17, 0.04, 0.015, 0.015, 0.06]))
        rows.append({"user_id": 10 + int(u), "recipe_id": int(recipes.at[r, "id"]),
                     "date": date.strftime("%Y-%m-%d"), "rating": rating,
                     "review": "" if rng.random() < 0.01 else "great recipe"})
    inter = pd.DataFrame(rows)

    recipes.to_csv(out / "RAW_recipes.csv", index=False)
    inter.to_csv(out / "RAW_interactions.csv", index=False)
    print(f"wrote {len(recipes)} recipes, {len(inter)} interactions to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample/source")
    ap.add_argument("--recipes", type=int, default=1500)
    ap.add_argument("--users", type=int, default=600)
    ap.add_argument("--interactions", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(Path(a.out), a.recipes, a.users, a.interactions, a.seed)
