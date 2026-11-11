"""전처리 결정의 근거가 되는 대안 비교 표를 만든다 (docs/preprocessing.md 에서 인용).

기본 설정의 칼로리 기준과 양성 기준을 적용한 뒤, k 와 분할 방식을 바꿔 가며 결과 규모를 비교한다.

사용법:
    python3 scripts/preprocess_alternatives.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.filtering import k_core  # noqa: E402
from dmrec.preprocess import calorie_mask, parse_recipes, split_leave_last_out  # noqa: E402

K_VALUES = [3, 5, 10]


def main() -> None:
    config = load_config()
    pre = config["preprocess"]
    raw_recipes = pd.read_csv(resolve_path(config, "recipes_csv"))
    inter = pd.read_csv(resolve_path(config, "interactions_csv"), usecols=["user_id", "recipe_id", "date", "rating"])

    recipes = parse_recipes(raw_recipes, pre["nutrition_fields"])
    keep, _ = calorie_mask(recipes, pre["calorie_max_quantile"], pre["remove_zero_calories"])
    recipes = recipes[keep]
    inter = inter[inter["recipe_id"].isin(recipes["recipe_id"])]
    calories = recipes.set_index("recipe_id")["calories"]

    rows = []
    for label, frame in (("all_interactions", inter), (f"rating>={pre['positive_min_rating']}", inter[inter["rating"] >= pre["positive_min_rating"]])):
        for k in K_VALUES:
            core = k_core(frame, k)
            row = {"positives": label, "k": k, "interactions": len(core), "users": core["user_id"].nunique(), "recipes": core["recipe_id"].nunique()}

            # 사용자별 시간순 분할 (마지막 1개 test, 그 앞 1개 valid)
            train, valid, test = split_leave_last_out(core, 1, 1)
            per_user = train["user_id"].value_counts()
            train_recipes = set(train["recipe_id"])
            row["llo_train_per_user_min"] = int(per_user.min())
            row["llo_train_per_user_median"] = float(per_user.median())
            row["llo_test_users"] = int(test["recipe_id"].isin(train_recipes).sum())
            row["llo_test_warm_pct"] = round(float(test["recipe_id"].isin(train_recipes).mean() * 100), 2)

            # 전체 시간 기준 80/10/10 분할
            ordered = core.sort_values("date", kind="stable")
            n = len(ordered)
            g_train, g_test = ordered.iloc[: int(n * 0.8)], ordered.iloc[int(n * 0.9):]
            warm = g_test["user_id"].isin(set(g_train["user_id"])) & g_test["recipe_id"].isin(set(g_train["recipe_id"]))
            row["global_train_end"] = g_train["date"].max()
            row["global_test_start"] = g_test["date"].min()
            row["global_test_warm_pct"] = round(float(warm.mean() * 100), 2)
            row["global_test_warm_users"] = int(g_test[warm]["user_id"].nunique())

            # core 레시피의 칼로리 분포
            core_cal = calories[calories.index.isin(core["recipe_id"].unique())]
            row["core_calories_median"] = round(float(core_cal.median()), 1)
            row["core_calories_p99"] = round(float(core_cal.quantile(0.99)), 1)
            row["core_calories_max"] = round(float(core_cal.max()), 1)
            rows.append(row)

    table = pd.DataFrame(rows)
    out = REPO_ROOT / "reports" / "tables" / "preprocess_alternatives.csv"
    table.to_csv(out, index=False)
    print(table.T.to_string())
    print("저장:", out)


if __name__ == "__main__":
    main()
