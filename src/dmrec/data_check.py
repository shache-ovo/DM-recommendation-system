"""원본 데이터 파일을 source_manifest.json 과 대조한다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(manifest_path: Path, data_dir: Path) -> list[str]:
    """manifest 의 각 파일에 대해 존재, 크기, sha256 을 확인한다. 문제 목록을 돌려준다(없으면 빈 리스트)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = data_dir / entry["name"]
        if not path.exists():
            problems.append(f"{entry['name']}: 파일 없음 ({path})")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            problems.append(f"{entry['name']}: 크기 {size} != 기대 {entry['bytes']}")
            continue
        actual = sha256_of(path)
        if actual != entry["sha256"]:
            problems.append(f"{entry['name']}: sha256 {actual} != 기대 {entry['sha256']}")
    return problems


def check_counts(manifest_path: Path, data_dir: Path) -> list[str]:
    """행 수, 컬럼, 고유 사용자 수를 manifest 와 대조한다. pandas 가 필요하다."""
    import pandas as pd

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in manifest["files"]}
    expected = manifest["expected_counts"]
    problems = []

    recipes = pd.read_csv(data_dir / "RAW_recipes.csv", usecols=["id"])
    interactions = pd.read_csv(data_dir / "RAW_interactions.csv", usecols=["user_id", "recipe_id"])
    observed = {
        "recipes": recipes["id"].nunique(),
        "users": interactions["user_id"].nunique(),
        "interactions": len(interactions),
    }
    for key, value in expected.items():
        if observed[key] != value:
            problems.append(f"{key}: {observed[key]} != 기대 {value}")

    for name in ("RAW_recipes.csv", "RAW_interactions.csv"):
        columns = list(pd.read_csv(data_dir / name, nrows=0).columns)
        if columns != by_name[name]["columns"]:
            problems.append(f"{name}: 컬럼 {columns} != 기대 {by_name[name]['columns']}")

    unmatched = int((~interactions["recipe_id"].isin(recipes["id"])).sum())
    if unmatched:
        problems.append(f"RAW_recipes.csv 에 없는 recipe_id 를 가진 상호작용 {unmatched}건")
    return problems
