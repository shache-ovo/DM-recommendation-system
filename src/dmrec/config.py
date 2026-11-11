"""설정 파일 로딩."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """YAML 설정을 읽어 dict 로 돌려준다. path 가 없으면 configs/default.yaml 을 쓴다."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """값 사이의 제약을 확인한다. 어긋나면 ValueError."""
    num_perm = config["minhash"]["num_perm"]
    bands, rows = config["lsh"]["bands"], config["lsh"]["rows"]
    if bands * rows != num_perm:
        raise ValueError(
            f"lsh.bands * lsh.rows ({bands} * {rows}) 가 minhash.num_perm ({num_perm}) 과 다르다"
        )
    if not 0 <= config["preprocess"]["positive_min_rating"] <= 5:
        raise ValueError("preprocess.positive_min_rating 은 0~5 여야 한다")
    if config["preprocess"]["k_core"] < 1:
        raise ValueError("preprocess.k_core 는 1 이상이어야 한다")


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """paths 의 항목을 저장소 루트 기준 절대 경로로 바꾼다.

    recipes_csv, interactions_csv 는 raw_dir 아래의 파일로 해석한다.
    """
    paths = config["paths"]
    if key in ("recipes_csv", "interactions_csv"):
        return REPO_ROOT / paths["raw_dir"] / paths[key]
    return REPO_ROOT / paths[key]
