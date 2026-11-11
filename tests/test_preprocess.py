import json

import pandas as pd

from dmrec.config import load_config
from dmrec.preprocess import calorie_mask, check_outputs, parse_recipes, preprocess, split_leave_last_out

FIELDS = ["calories", "total_fat", "sugar", "sodium", "protein", "saturated_fat", "carbohydrates"]


def raw_recipes(calories):
    return pd.DataFrame({
        "name": [f"r{i}" for i in range(len(calories))],
        "id": [100 + i for i in range(len(calories))],
        "minutes": 10, "n_steps": 2, "n_ingredients": 2,
        "nutrition": [f"[{c}, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]" for c in calories],
        "ingredients": ["['salt', \"baker's yeast\"]"] * len(calories),
        "tags": ["['easy', 'dinner']"] * len(calories),
    })


def test_parse_recipes_expands_nutrition_and_writes_json_lists():
    parsed = parse_recipes(raw_recipes([51.5]), FIELDS)
    row = parsed.iloc[0]
    assert row["recipe_id"] == 100
    assert [row[f] for f in FIELDS] == [51.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert json.loads(row["ingredients"]) == ["salt", "baker's yeast"]
    assert json.loads(row["tags"]) == ["easy", "dinner"]


def test_calorie_mask_removes_zero_and_values_over_quantile():
    recipes = pd.DataFrame({"calories": [0.0] + [float(v) for v in range(1, 100)] + [1_000_000.0]})
    keep, info = calorie_mask(recipes, 0.99, True)
    assert not keep.iloc[0] and not keep.iloc[-1]
    assert info["removed_zero"] == 1 and info["removed_over_cutoff"] == 1
    keep_all, _ = calorie_mask(recipes, None, False)
    assert keep_all.all()


def test_split_takes_last_rows_and_breaks_date_ties_by_row_order():
    positives = pd.DataFrame({
        "user_id": ["a", "b", "a", "a", "a", "b", "b"],
        "recipe_id": [1, 9, 2, 3, 4, 8, 7],
        "date": ["2010-01-03", "2011-01-01", "2010-01-01", "2010-01-03", "2010-01-02", "2011-01-01", "2011-01-01"],
        "rating": 5,
    })
    train, valid, test = split_leave_last_out(positives, 1, 1)
    # a: 날짜순 2, 4, 1, 3 (1 과 3 은 같은 날, 파일에서 1 이 먼저)
    assert list(train[train.user_id == "a"].recipe_id) == [2, 4]
    assert list(valid[valid.user_id == "a"].recipe_id) == [1]
    assert list(test[test.user_id == "a"].recipe_id) == [3]
    # b: 모두 같은 날 -> 파일 순서 9, 8, 7
    assert list(train[train.user_id == "b"].recipe_id) == [9]
    assert list(valid[valid.user_id == "b"].recipe_id) == [8]
    assert list(test[test.user_id == "b"].recipe_id) == [7]


def small_config():
    config = load_config()
    config["preprocess"].update({"k_core": 3, "calorie_max_quantile": None, "remove_zero_calories": True})
    return config


def test_preprocess_end_to_end_on_small_data():
    # 사용자 4명이 레시피 100~103 을 모두 양성으로 평가. 104 는 칼로리 0, 105 는 저평점만 받음.
    recipes = raw_recipes([10, 20, 30, 40, 0, 50])
    rows = []
    for u in range(4):
        for j, rid in enumerate([100, 101, 102, 103]):
            rows.append((u, rid, f"2010-01-{(j + u) % 4 + 1:02d}", 5))
        rows.append((u, 104, "2010-02-01", 5))   # 칼로리 0 레시피 -> 제거
    rows.append((0, 105, "2010-03-01", 2))        # 비양성, 105 는 core 밖 -> seen 에도 없음
    rows.append((9, 100, "2010-03-01", 5))        # 1건뿐인 사용자 -> k-core 에서 제거
    inter = pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating"])
    config = small_config()
    result = preprocess(recipes, inter, config)

    assert check_outputs(result, config) == []
    assert len(result["recipes"]) == 5                       # 104 제거
    assert set(result["users"]["user_id"]) == {0, 1, 2, 3}
    assert len(result["train"]) == 8 and len(result["valid"]) == 4 and len(result["test"]) == 4
    assert len(result["seen_non_positive"]) == 0
    assert (result["recipes"].set_index("recipe_id").loc[105, "recipe_idx"]) == -1
    steps = result["steps"].set_index("step")["interactions"]
    assert steps["0_raw"] == 22 and steps["1_calorie_filter"] == 18 and steps["2_positive_only"] == 17 and steps["3_k_core"] == 16


def test_seen_keeps_non_positive_rows_inside_core():
    recipes = raw_recipes([10, 20, 30, 40])
    rows = [(u, rid, f"2010-01-0{j + 1}", 5) for u in range(4) for j, rid in enumerate([100, 101, 102])]
    rows += [(u, 103, "2009-12-31", 5) for u in range(1, 4)]   # 103 은 가장 이른 기록이라 train 에 들어간다
    rows.append((0, 103, "2010-01-09", 0))                     # 사용자 0 은 103 에 평점 없이 리뷰만 남김
    inter = pd.DataFrame(rows, columns=["user_id", "recipe_id", "date", "rating"])
    config = small_config()
    result = preprocess(recipes, inter, config)
    assert check_outputs(result, config) == []
    assert 103 in set(result["train"]["recipe_id"])
    seen = result["seen_non_positive"]
    assert list(zip(seen.user_id, seen.recipe_id)) == [(0, 103)]
