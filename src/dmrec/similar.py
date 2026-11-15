"""재료가 비슷한 레시피 검색. 전처리 산출물에서 LSH 인덱스를 만들고 recipe_id 로 조회한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from dmrec.ingredients import to_csr
from dmrec.lsh import LSHIndex
from dmrec.minhash import estimate_jaccard


class SimilarRecipes:
    """recipe_id 기준의 재료 유사 레시피 검색기.

    후보는 LSH 로 찾는다. 재료 집합 행렬(ingredient_matrix)을 주면 후보의 순위를 정확한 Jaccard 로 매기고,
    주지 않으면 MinHash 추정값으로 매긴다.
    """

    def __init__(self, recipe_ids: np.ndarray, signatures: np.ndarray, bands: int, rows: int, ingredient_matrix=None):
        self.recipe_ids = np.asarray(recipe_ids)
        self._row = {int(rid): i for i, rid in enumerate(self.recipe_ids)}
        self.index = LSHIndex(bands, rows).build(signatures)
        self.matrix = None if ingredient_matrix is None else ingredient_matrix.tocsr()
        self.exact = self.matrix is not None
        if self.exact:
            self._sizes = np.asarray(self.matrix.sum(axis=1)).ravel().astype(np.float64)

    @classmethod
    def from_processed(cls, processed_dir: Path, config: dict[str, Any], scope: str | None = None, rank_by: str | None = None) -> "SimilarRecipes":
        """data/processed 의 minhash_signatures.npz 와 ingredient_sets.csv 에서 만든다.

        scope 가 "core" 면 추천 대상 레시피만, "all" 이면 전체 레시피를 인덱스에 넣는다.
        rank_by 가 "exact" 면 후보를 정확한 Jaccard 로, "estimate" 면 MinHash 추정값으로 정렬한다.
        둘 다 생략하면 설정값을 쓴다.
        """
        scope = scope or config["lsh"]["index_scope"]
        rank_by = rank_by or config["lsh"]["rank_by"]
        if scope not in ("core", "all"):
            raise ValueError(f"lsh.index_scope 는 core 또는 all 이어야 한다: {scope}")
        if rank_by not in ("exact", "estimate"):
            raise ValueError(f"lsh.rank_by 는 exact 또는 estimate 여야 한다: {rank_by}")
        data = np.load(Path(processed_dir) / "minhash_signatures.npz")
        recipe_ids, recipe_idx, signatures = data["recipe_id"], data["recipe_idx"], data["signatures"]
        keep = np.arange(len(recipe_ids))
        if scope == "core":
            keep = np.flatnonzero(recipe_idx >= 0)
            keep = keep[np.argsort(recipe_idx[keep])]       # recipe_idx 순서로 정렬: 행 번호 == recipe_idx
        matrix = None
        if rank_by == "exact":
            sets = pd.read_csv(Path(processed_dir) / "ingredient_sets.csv", usecols=["recipe_id", "ingredients_norm"])
            if not np.array_equal(sets["recipe_id"].to_numpy(), recipe_ids):
                raise ValueError("ingredient_sets.csv 와 minhash_signatures.npz 의 행 순서가 다르다. build_minhash.py 를 다시 실행한다")
            matrix, _ = to_csr([json.loads(s) for s in sets["ingredients_norm"].to_numpy()[keep]])
        return cls(recipe_ids[keep], signatures[keep], config["lsh"]["bands"], config["lsh"]["rows"], matrix)

    def _query_row(self, row: int, top_n: int | None, min_jaccard: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(이웃 행 번호, 순위에 쓴 유사도, MinHash 추정값) 을 유사도 내림차순으로 돌려준다."""
        if not self.exact:
            rows, estimates = self.index.query(row, top_n, min_jaccard)
            return rows, estimates, estimates
        found = self.index.candidates(row)
        if len(found) == 0:
            empty = np.empty(0)
            return found, empty, empty
        inter = np.asarray((self.matrix[found] @ self.matrix[row].T).todense()).ravel().astype(np.float64)
        scores = inter / (self._sizes[found] + self._sizes[row] - inter)
        keep = scores >= min_jaccard
        found, scores = found[keep], scores[keep]
        order = np.lexsort((found, -scores))                 # 정확한 Jaccard 내림차순, 같으면 행 번호 오름차순
        if top_n is not None:
            order = order[:top_n]
        found, scores = found[order], scores[order]
        estimates = estimate_jaccard(self.index.signatures[found], self.index.signatures[row][None, :]) if len(found) else np.empty(0)
        return found, scores, estimates

    def __len__(self) -> int:
        return len(self.recipe_ids)

    def __contains__(self, recipe_id: int) -> bool:
        return int(recipe_id) in self._row

    def similar(self, recipe_id: int, top_n: int | None = None, min_jaccard: float = 0.0) -> list[tuple[int, float]]:
        """recipe_id 와 재료가 비슷한 레시피를 [(recipe_id, Jaccard), ...] 로 돌려준다. 유사도 내림차순이다.

        유사도는 정확한 Jaccard 다(rank_by 가 estimate 면 MinHash 추정값). 인덱스에 없는 recipe_id 면 KeyError 다.
        """
        rows, scores, _ = self._query_row(self._row[int(recipe_id)], top_n, min_jaccard)
        return [(int(self.recipe_ids[r]), float(v)) for r, v in zip(rows, scores)]

    def candidates_for(self, recipe_ids: Iterable[int], top_n_each: int | None = None, min_jaccard: float = 0.0,
                       exclude: Iterable[int] = ()) -> dict[int, float]:
        """여러 레시피(예: 사용자가 좋아한 레시피들)의 유사 레시피를 합친다.

        {후보 recipe_id: 입력 레시피들과의 Jaccard 중 최댓값} 을 돌려준다.
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
        """모든 레시피의 상위 top_n 이웃을 표로 만든다.

        jaccard 는 순위에 쓴 유사도(정확값, rank_by 가 estimate 면 추정값), est_jaccard 는 MinHash 추정값이다.
        """
        records = []
        for row, rid in enumerate(self.recipe_ids):
            rows, scores, estimates = self._query_row(row, top_n, min_jaccard)
            for rank, (r, v, e) in enumerate(zip(rows, scores, estimates), start=1):
                records.append((int(rid), int(self.recipe_ids[r]), rank, round(float(v), 6), round(float(e), 6)))
        return pd.DataFrame(records, columns=["recipe_id", "neighbor_recipe_id", "rank", "jaccard", "est_jaccard"])
