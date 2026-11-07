# Data Validation Report

Generated: 2026-11-07T16:02:48Z

Rows breaking an **error** rule are quarantined to `data/errors/<table>_errors.csv`. **Warning** rules keep the row and are handled in the preparation stage.

## Summary

| table | rows_in | valid | quarantined | pass_rate | rows_with_warnings |
|---|---|---|---|---|---|
| recipes | 231637 | 231637 | 0 | 1.0000 | 5599 |
| users | 236719 | 236719 | 0 | 1.0000 | 0 |
| interactions | 1132367 | 1131121 | 1246 | 0.9989 | 61012 |

Quality gate (every table pass rate >= 0.95): **PASSED**

## recipes

36 rules checked.

| rule | severity | description | rows | % of rows |
|---|---|---|---|---|
| type_error:id | error | id is present but cannot be parsed as int | 0 | 0.000 |
| type_error:minutes | error | minutes is present but cannot be parsed as int | 0 | 0.000 |
| type_error:contributor_id | error | contributor_id is present but cannot be parsed as int | 0 | 0.000 |
| type_error:submitted | error | submitted is present but cannot be parsed as date | 0 | 0.000 |
| type_error:n_steps | error | n_steps is present but cannot be parsed as int | 0 | 0.000 |
| type_error:n_ingredients | error | n_ingredients is present but cannot be parsed as int | 0 | 0.000 |
| malformed_list:tags | error | tags is not a valid list literal | 0 | 0.000 |
| malformed_list:ingredients | error | ingredients is not a valid list literal | 0 | 0.000 |
| malformed_list:nutrition | error | nutrition is not a valid list literal of 7 numbers | 0 | 0.000 |
| not_null:id | error | id is missing | 0 | 0.000 |
| positive_id:id | error | id must satisfy greater_than(0) | 0 | 0.000 |
| not_null:minutes | error | minutes is missing | 0 | 0.000 |
| non_negative_minutes:minutes | error | minutes must satisfy greater_than_or_equal_to(0) | 0 | 0.000 |
| not_null:contributor_id | error | contributor_id is missing | 0 | 0.000 |
| positive_id:contributor_id | error | contributor_id must satisfy greater_than(0) | 0 | 0.000 |
| not_null:submitted | error | submitted is missing | 0 | 0.000 |
| date_after_min:submitted | error | submitted must satisfy greater_than_or_equal_to(1999-01-01 00:00:00) | 0 | 0.000 |
| date_before_max:submitted | error | submitted must satisfy less_than_or_equal_to(2018-12-31 00:00:00) | 0 | 0.000 |
| not_null:n_steps | error | n_steps is missing | 0 | 0.000 |
| non_negative_n_steps:n_steps | error | n_steps must satisfy greater_than_or_equal_to(0) | 0 | 0.000 |
| not_null:n_ingredients | error | n_ingredients is missing | 0 | 0.000 |
| at_least_one_ingredient:n_ingredients | error | n_ingredients must satisfy greater_than_or_equal_to(1) | 0 | 0.000 |
| not_null:tags | error | tags is missing | 0 | 0.000 |
| not_null:ingredients | error | ingredients is missing | 0 | 0.000 |
| not_null:steps | error | steps is missing | 0 | 0.000 |
| not_null:nutrition | error | nutrition is missing | 0 | 0.000 |
| duplicate_row | error | exact duplicate of an earlier row | 0 | 0.000 |
| duplicate_key:id | error | recipe id already used by an earlier row | 0 | 0.000 |
| missing_description | warning | description is empty | 4979 | 2.149 |
| zero_minutes | warning | minutes == 0 (treated as unknown and imputed) | 1094 | 0.472 |
| extreme_calories | warning | calories > validation.max_calories | 125 | 0.054 |
| implausible_minutes | warning | minutes > validation.max_recipe_minutes (30 days) | 70 | 0.030 |
| missing_name | warning | recipe name is empty | 1 | 0.000 |
| zero_steps | warning | n_steps == 0 | 1 | 0.000 |
| ingredient_count_mismatch | warning | len(ingredients) != n_ingredients | 0 | 0.000 |
| empty_tags | warning | tag list is empty | 0 | 0.000 |

## users

16 rules checked.

| rule | severity | description | rows | % of rows |
|---|---|---|---|---|
| type_error:user_id | error | user_id is present but cannot be parsed as int | 0 | 0.000 |
| type_error:member_since | error | member_since is present but cannot be parsed as date | 0 | 0.000 |
| type_error:n_reviews | error | n_reviews is present but cannot be parsed as int | 0 | 0.000 |
| type_error:n_recipes_contributed | error | n_recipes_contributed is present but cannot be parsed as int | 0 | 0.000 |
| type_error:is_reviewer | error | is_reviewer is present but cannot be parsed as bool | 0 | 0.000 |
| type_error:is_contributor | error | is_contributor is present but cannot be parsed as bool | 0 | 0.000 |
| type_error:first_review_date | error | first_review_date is present but cannot be parsed as date | 0 | 0.000 |
| type_error:first_contribution_date | error | first_contribution_date is present but cannot be parsed as date | 0 | 0.000 |
| not_null:user_id | error | user_id is missing | 0 | 0.000 |
| unique:user_id | error | user_id must be unique | 0 | 0.000 |
| positive_id:user_id | error | user_id must satisfy greater_than(0) | 0 | 0.000 |
| not_null:member_since | error | member_since is missing | 0 | 0.000 |
| not_null:n_reviews | error | n_reviews is missing | 0 | 0.000 |
| non_negative:n_reviews | error | n_reviews must satisfy greater_than_or_equal_to(0) | 0 | 0.000 |
| not_null:n_recipes_contributed | error | n_recipes_contributed is missing | 0 | 0.000 |
| non_negative:n_recipes_contributed | error | n_recipes_contributed must satisfy greater_than_or_equal_to(0) | 0 | 0.000 |

## interactions

20 rules checked.

| rule | severity | description | rows | % of rows |
|---|---|---|---|---|
| invalid_timestamp:review_before_recipe_submitted | error | review date is earlier than the recipe's submitted date (impossible ordering) | 1246 | 0.110 |
| type_error:user_id | error | user_id is present but cannot be parsed as int | 0 | 0.000 |
| type_error:recipe_id | error | recipe_id is present but cannot be parsed as int | 0 | 0.000 |
| type_error:date | error | date is present but cannot be parsed as date | 0 | 0.000 |
| type_error:rating | error | rating is present but cannot be parsed as float | 0 | 0.000 |
| not_null:user_id | error | user_id is missing | 0 | 0.000 |
| positive_id:user_id | error | user_id must satisfy greater_than(0) | 0 | 0.000 |
| not_null:recipe_id | error | recipe_id is missing | 0 | 0.000 |
| positive_id:recipe_id | error | recipe_id must satisfy greater_than(0) | 0 | 0.000 |
| not_null:date | error | date is missing | 0 | 0.000 |
| date_after_min:date | error | date must satisfy greater_than_or_equal_to(1999-01-01 00:00:00) | 0 | 0.000 |
| date_before_max:date | error | date must satisfy less_than_or_equal_to(2018-12-31 00:00:00) | 0 | 0.000 |
| not_null:rating | error | rating is missing | 0 | 0.000 |
| rating_in_allowed_set:rating | error | rating must satisfy isin([0, 1, 2, 3, 4, 5]) | 0 | 0.000 |
| duplicate_row | error | exact duplicate of an earlier row (user, recipe, date, rating, review) | 0 | 0.000 |
| referential_integrity:recipe_id | error | recipe_id not found among valid recipes | 0 | 0.000 |
| referential_integrity:user_id | error | user_id not found among valid users | 0 | 0.000 |
| rating_zero_no_stars | warning | rating == 0, i.e. review written without a star rating | 60847 | 5.373 |
| missing_review | warning | review text is empty | 165 | 0.015 |
| repeat_user_recipe_pair | warning | user reviewed the same recipe more than once | 0 | 0.000 |
