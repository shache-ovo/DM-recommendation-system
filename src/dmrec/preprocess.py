"""원본 CSV 를 파싱, 정제, 필터링, 분할한다.

적용 순서:
  1. 레시피 파싱 (nutrition, ingredients, tags)
  2. 칼로리 기준으로 레시피 제거, 그 레시피의 상호작용도 제거
  3. 평점 기준으로 양성 표시
  4. 양성만으로 k-core
  5. 사용자별 시간순 분할 (train / valid / test)
  6. 평가 집합에는 train 에 등장한 레시피만 남김
"""

from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd

from dmrec.filtering import k_core

RECIPE_COLUMNS = ["recipe_id", "name", "minutes", "n_steps", "n_ingredients"]


def parse_recipes(raw: pd.DataFrame, nutrition_fields: list[str]) -> pd.DataFrame:
    """RAW_recipes 를 파싱한다. nutrition 은 컬럼 7개로 펼치고, ingredients 와 tags 는 JSON 리스트 문자열로 바꾼다."""
    out = raw.rename(columns={"id": "recipe_id"})[RECIPE_COLUMNS].copy()
    nutrition = pd.DataFrame(raw["nutrition"].map(json.loads).tolist(), columns=nutrition_fields, index=raw.index)
    out = pd.concat([out, nutrition], axis=1)
    # 원본은 파이썬 리스트 표기(작은따옴표)라 JSON 으로 다시 쓴다. 재료 이름 정규화는 여기서 하지 않는다.
    out["ingredients"] = raw["ingredients"].map(lambda s: json.dumps(ast.literal_eval(s), ensure_ascii=False))
    out["tags"] = raw["tags"].map(lambda s: json.dumps(ast.literal_eval(s), ensure_ascii=False))
    return out


def calorie_mask(recipes: pd.DataFrame, max_quantile: float | None, remove_zero: bool) -> tuple[pd.Series, dict[str, Any]]:
    """남길 레시피의 불리언 마스크와 적용 내역을 돌려준다."""
    keep = pd.Series(True, index=recipes.index)
    info: dict[str, Any] = {"calorie_cutoff": None, "removed_over_cutoff": 0, "removed_zero": 0}
    if max_quantile is not None:
        cutoff = float(recipes["calories"].quantile(max_quantile))
        over = recipes["calories"] > cutoff
        info["calorie_cutoff"] = cutoff
        info["removed_over_cutoff"] = int(over.sum())
        keep &= ~over
    if remove_zero:
        zero = recipes["calories"] == 0
        info["removed_zero"] = int(zero.sum())
        keep &= ~zero
    return keep, info


def split_leave_last_out(positives: pd.DataFrame, n_test: int, n_valid: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """사용자별로 시간순 정렬해 마지막 n_test 개를 test, 그 앞 n_valid 개를 valid, 나머지를 train 으로 나눈다.

    같은 날짜의 순서는 입력 프레임의 행 순서(원본 파일 순서)를 따른다.
    """
    ordered = positives.sort_values("date", kind="stable")
    ordered = ordered.sort_values("user_id", kind="stable")
    from_end = ordered.groupby("user_id").cumcount(ascending=False)
    test = ordered[from_end < n_test]
    valid = ordered[(from_end >= n_test) & (from_end < n_test + n_valid)]
    train = ordered[from_end >= n_test + n_valid]
    return train, valid, test


def preprocess(raw_recipes: pd.DataFrame, raw_interactions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """전체 전처리를 수행한다. 산출 프레임과 단계별 행 수 기록을 돌려준다."""
    pre, split_cfg = config["preprocess"], config["split"]
    if split_cfg["method"] != "per_user_leave_last_out":
        raise ValueError(f"지원하지 않는 분할 방식: {split_cfg['method']}")
    steps: list[dict[str, Any]] = []

    def log(step: str, inter: pd.DataFrame, n_recipes: int | None = None, note: str = "") -> None:
        steps.append({
            "step": step,
            "interactions": len(inter),
            "users": inter["user_id"].nunique(),
            "recipes_with_interactions": inter["recipe_id"].nunique(),
            "recipes_in_catalog": n_recipes,
            "note": note,
        })

    # 1. 파싱
    recipes = parse_recipes(raw_recipes, pre["nutrition_fields"])
    inter = raw_interactions[["user_id", "recipe_id", "date", "rating"]]
    log("0_raw", inter, len(recipes))

    # 2. 칼로리 기준
    keep, calorie_info = calorie_mask(recipes, pre["calorie_max_quantile"], pre["remove_zero_calories"])
    recipes = recipes[keep].reset_index(drop=True)
    inter = inter[inter["recipe_id"].isin(recipes["recipe_id"])]
    cutoff = calorie_info["calorie_cutoff"]
    log("1_calorie_filter", inter, len(recipes),
        f"cutoff={cutoff}, over={calorie_info['removed_over_cutoff']}, zero={calorie_info['removed_zero']}")

    # 3. 양성 표시
    is_positive = inter["rating"] >= pre["positive_min_rating"]
    positives, non_positives = inter[is_positive], inter[~is_positive]
    log("2_positive_only", positives, len(recipes), f"rating>={pre['positive_min_rating']}")

    # 4. k-core
    core = k_core(positives, pre["k_core"])
    log("3_k_core", core, len(recipes), f"k={pre['k_core']}")

    # 5. 분할
    train, valid, test = split_leave_last_out(core, split_cfg["n_test"], split_cfg["n_valid"])

    # 6. 평가 집합에는 train 에 등장한 레시피만
    dropped = {"valid": 0, "test": 0}
    if split_cfg["restrict_eval_to_train"]:
        train_recipes = set(train["recipe_id"])
        for name, frame in (("valid", valid), ("test", test)):
            in_train = frame["recipe_id"].isin(train_recipes)
            dropped[name] = int((~in_train).sum())
        valid = valid[valid["recipe_id"].isin(train_recipes)]
        test = test[test["recipe_id"].isin(train_recipes)]
    log("4_train", train, len(recipes))
    log("4_valid", valid, len(recipes), f"dropped_not_in_train={dropped['valid']}")
    log("4_test", test, len(recipes), f"dropped_not_in_train={dropped['test']}")

    # 인덱스: train 에 등장한 사용자·레시피에 0부터 연속 번호를 준다 (id 오름차순).
    user_ids = sorted(train["user_id"].unique())
    recipe_ids = sorted(train["recipe_id"].unique())
    users = pd.DataFrame({"user_id": user_ids, "user_idx": range(len(user_ids))})
    users["n_train"] = users["user_id"].map(train["user_id"].value_counts()).astype(int)
    recipe_idx = {rid: i for i, rid in enumerate(recipe_ids)}
    recipes.insert(1, "recipe_idx", recipes["recipe_id"].map(recipe_idx).fillna(-1).astype(int))
    recipes.insert(2, "n_train", recipes["recipe_id"].map(train["recipe_id"].value_counts()).fillna(0).astype(int))

    # core 사용자가 core 레시피에 남긴 비양성 기록: 추천 대상과 negative 샘플에서 제외하는 용도
    seen = non_positives[non_positives["user_id"].isin(set(user_ids)) & non_positives["recipe_id"].isin(set(recipe_ids))]
    log("5_seen_non_positive", seen, len(recipes), "core 사용자 x core 레시피의 비양성 기록")

    def finish(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(["user_id", "date"], kind="stable").reset_index(drop=True)

    return {
        "recipes": recipes,
        "users": users,
        "train": finish(train),
        "valid": finish(valid),
        "test": finish(test),
        "seen_non_positive": finish(seen),
        "steps": pd.DataFrame(steps),
        "calorie_info": calorie_info,
    }


def check_outputs(result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """산출물의 불변식을 확인한다. 어긋난 항목의 목록을 돌려준다(없으면 빈 리스트)."""
    train, valid, test, seen = result["train"], result["valid"], result["test"], result["seen_non_positive"]
    recipes, users = result["recipes"], result["users"]
    problems = []
    key = ["user_id", "recipe_id"]

    everything = pd.concat([train[key], valid[key], test[key], seen[key]])
    if everything.duplicated().any():
        problems.append("train/valid/test/seen 사이에 겹치는 (user, recipe) 쌍이 있다")
    train_users, train_recipes = set(train["user_id"]), set(train["recipe_id"])
    for name, frame in (("valid", valid), ("test", test), ("seen", seen)):
        if not set(frame["user_id"]) <= train_users:
            problems.append(f"{name} 에 train 에 없는 사용자가 있다")
        if not set(frame["recipe_id"]) <= train_recipes:
            problems.append(f"{name} 에 train 에 없는 레시피가 있다")
    last_train = train.groupby("user_id")["date"].max()
    for name, frame in (("valid", valid), ("test", test)):
        if (frame["date"] < frame["user_id"].map(last_train)).any():
            problems.append(f"{name} 에 그 사용자의 train 마지막 날짜보다 이른 기록이 있다")
        if frame["user_id"].value_counts().max() > max(config["split"]["n_test"], config["split"]["n_valid"]):
            problems.append(f"{name} 의 사용자당 행 수가 설정보다 많다")
    first_test = test.groupby("user_id")["date"].min()
    both = valid[valid["user_id"].isin(first_test.index)]
    if (both["date"] > both["user_id"].map(first_test)).any():
        problems.append("valid 가 같은 사용자의 test 보다 늦다")
    min_positive = config["preprocess"]["positive_min_rating"]
    if (pd.concat([train, valid, test])["rating"] < min_positive).any():
        problems.append("양성 집합에 기준 미만 평점이 있다")
    if (seen["rating"] >= min_positive).any():
        problems.append("seen 에 양성 평점이 있다")
    if len(users) != len(train_users) or list(users["user_idx"]) != list(range(len(users))):
        problems.append("users 의 인덱스가 train 사용자와 맞지 않다")
    indexed = recipes[recipes["recipe_idx"] >= 0]
    if set(indexed["recipe_id"]) != train_recipes or sorted(indexed["recipe_idx"]) != list(range(len(indexed))):
        problems.append("recipes 의 recipe_idx 가 train 레시피와 맞지 않다")
    if recipes["recipe_id"].duplicated().any():
        problems.append("recipes 에 중복 recipe_id 가 있다")
    if recipes[config["preprocess"]["nutrition_fields"]].isna().any().any():
        problems.append("nutrition 에 결측이 있다")
    return problems
