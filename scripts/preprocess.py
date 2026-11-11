"""전처리를 실행해 data/processed 에 산출물을, reports/tables 에 단계별 행 수를 쓴다.

사용법:
    python3 scripts/preprocess.py
    python3 scripts/preprocess.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.preprocess import check_outputs, preprocess  # noqa: E402

OUTPUTS = {
    "recipes": "recipes.csv",
    "users": "users.csv",
    "train": "interactions_train.csv",
    "valid": "interactions_valid.csv",
    "test": "interactions_test.csv",
    "seen_non_positive": "interactions_seen_non_positive.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    raw_recipes = pd.read_csv(resolve_path(config, "recipes_csv"))
    raw_interactions = pd.read_csv(resolve_path(config, "interactions_csv"), usecols=["user_id", "recipe_id", "date", "rating"])
    result = preprocess(raw_recipes, raw_interactions, config)

    problems = check_outputs(result, config)
    if problems:
        print("산출물 검사 실패:")
        for problem in problems:
            print(" -", problem)
        return 1

    processed = resolve_path(config, "processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUTS.items():
        result[key].to_csv(processed / filename, index=False)

    tables = REPO_ROOT / "reports" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    result["steps"].to_csv(tables / "preprocess_steps.csv", index=False)
    train = result["train"]
    summary = {
        "config": {"preprocess": config["preprocess"], "split": config["split"]},
        "calorie": result["calorie_info"],
        "rows": {key: len(result[key]) for key in OUTPUTS},
        "core_users": len(result["users"]),
        "core_recipes": int((result["recipes"]["recipe_idx"] >= 0).sum()),
        "train_density_pct": round(len(train) / (len(result["users"]) * train["recipe_id"].nunique()) * 100, 4),
        "train_per_user": {"min": int(result["users"]["n_train"].min()), "median": float(result["users"]["n_train"].median()), "max": int(result["users"]["n_train"].max())},
        "users_with_valid": int(result["valid"]["user_id"].nunique()),
        "users_with_test": int(result["test"]["user_id"].nunique()),
        "date_range": {key: [str(result[key]["date"].min()), str(result[key]["date"].max())] for key in ("train", "valid", "test")},
    }
    (tables / "preprocess_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(result["steps"].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))
    print("산출물 검사 통과. 저장 위치:", processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
