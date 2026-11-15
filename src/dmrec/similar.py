"""재료가 비슷한 레시피 검색. 전처리 산출물에서 LSH 인덱스를 만들고 recipe_id 로 조회한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dmrec.lsh import LSHIndex


class SimilarRecipes:
    """recipe_id 기준의 재료 유사 레시피 검색기."""

    def __init__(self, recipe_ids: np.ndarray, signatures: np.ndarray, bands: int, rows: int):
        self.recipe_ids = np.asarray(recipe_ids)
        self._row = {int(rid): i for i, rid in enumerate(self.recipe_ids)}
        self.index = LSHIndex(bands, rows).build(signatures)

    @classmethod
    def from_processed(cls, processed_dir: Path, config: dict[str, Any], scope: str | None = None) -> "SimilarRecipes":
        """data/processed/minhash_signatures.npz 에서 만든다.

        scope 가 "core" 면 추천 대상 레시피만, "all" 이면 전체 레시피를 인덱스에 넣는다. 생략하면 설정값을 쓴다.
        """
        scope = scope or config["lsh"]["index_scope"]
        if scope not in ("core", "all"):
            raise ValueError(f"lsh.index_scope 는 core 또는 all 이어야 한다: {scope}")
        data = np.load(Path(processed_dir) / "minhash_signatures.npz")
        recipe_ids, recipe_idx, signatures = data["recipe_id"], data["recipe_idx"], data["signatures"]
        if scope == "core":
            keep = np.flatnonzero(recipe_idx >= 0)
            keep = keep[np.argsort(recipe_idx[keep])]       # recipe_idx 순서로 정렬: 행 번호 == recipe_idx
            recipe_ids, signatures = recipe_ids[keep], signatures[keep]
        return cls(recipe_ids, signatures, config["lsh"]["bands"], config["lsh"]["rows"])

    def __len__(self) -> int:
        return len(self.recipe_ids)

    def __contains__(self, recipe_id: int) -> bool:
        return int(recipe_id) in self._row

    def similar(self, recipe_id: int, top_n: int | None = None, min_jaccard: float = 0.0) -> list[tuple[int, float]]:
        """recipe_id 와 재료가 비슷한 레시피를 [(recipe_id, 추정 Jaccard), ...] 로 돌려준다. 유사도 내림차순이다.

        인덱스에 없는 recipe_id 면 KeyError 다.
        """
        rows, estimates = self.index.query(self._row[int(recipe_id)], top_n, min_jaccard)
        return [(int(self.recipe_ids[r]), float(e)) for r, e in zip(rows, estimates)]

    def candidates_for(self, recipe_ids: Iterable[int], top_n_each: int | None = None, min_jaccard: float = 0.0,
                       exclude: Iterable[int] = ()) -> dict[int, float]:
        """여러 레시피(예: 사용자가 좋아한 레시피들)의 유사 레시피를 합친다.

        {후보 recipe_id: 입력 레시피들과의 추정 Jaccard 중 최댓값} 을 돌려준다.
        입력 레시피 자신과 exclude 에 든 레시피는 뺀다. 인덱스에 없는 입력은 건너뛴다.
        """
        sources = [int(r) for r in recipe_ids]
        skip = set(sources) | {int(r) for r in exclude}
        best: dict[int, float] = {}
        for rid in sources:
            if rid not in self._row:
                continue
            for other, estimate in self.similar(rid, top_n_each, min_jaccard):
                if other not in skip and estimate > best.get(other, -1.0):
                    best[other] = estimate
        return best

    def neighbor_table(self, top_n: int, min_jaccard: float = 0.0) -> pd.DataFrame:
        """모든 레시피의 상위 top_n 이웃을 표로 만든다."""
        records = []
        for row, rid in enumerate(self.recipe_ids):
            rows, estimates = self.index.query(row, top_n, min_jaccard)
            for rank, (r, e) in enumerate(zip(rows, estimates), start=1):
                records.append((int(rid), int(self.recipe_ids[r]), rank, round(float(e), 6)))
        return pd.DataFrame(records, columns=["recipe_id", "neighbor_recipe_id", "rank", "est_jaccard"])
