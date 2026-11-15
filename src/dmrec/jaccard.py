"""정확한 Jaccard 유사도. MinHash 추정과 LSH 결과를 검증하는 기준값으로 쓴다."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def jaccard(a: set, b: set) -> float:
    """두 집합의 Jaccard 유사도. 둘 다 비어 있으면 0 으로 정의한다."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(matrix: csr_matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """행 번호 쌍 (left[i], right[i]) 의 Jaccard 를 한 번에 계산한다."""
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    inter = np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()
    union = sizes[left] + sizes[right] - inter
    return np.divide(inter, union, out=np.zeros(len(left), dtype=np.float64), where=union > 0)


def jaccard_to_all(matrix: csr_matrix, rows: np.ndarray) -> np.ndarray:
    """rows 의 각 행과 모든 행 사이의 Jaccard 행렬 (len(rows) x n) 을 돌려준다. 메모리를 위해 rows 를 나눠서 부른다."""
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    inter = (matrix[rows] @ matrix.T).toarray()
    union = sizes[rows][:, None] + sizes[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def top_k_neighbors(matrix: csr_matrix, k: int, chunk: int = 1000, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """각 행에 대해 Jaccard 가 가장 큰 k 개의 다른 행을 전수 비교로 찾는다.

    (이웃 행 번호, Jaccard) 를 유사도 내림차순으로 돌려준다. 동률은 행 번호가 작은 쪽이 먼저다.
    """
    rows = np.arange(matrix.shape[0]) if rows is None else np.asarray(rows)
    neighbors = np.empty((len(rows), k), dtype=np.int32)
    scores = np.empty((len(rows), k), dtype=np.float32)
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        sim = jaccard_to_all(matrix, part)
        sim[np.arange(len(part)), part] = -1.0          # 자기 자신 제외
        # 행 전체를 정렬하면 느리므로, k 번째로 큰 값을 먼저 구하고 그 이상인 것만 정렬한다.
        # 유사도 내림차순, 같은 값이면 행 번호 오름차순이다.
        kth = np.partition(sim, sim.shape[1] - k, axis=1)[:, sim.shape[1] - k]
        for r in range(len(part)):
            candidates = np.flatnonzero(sim[r] >= kth[r])
            order = candidates[np.lexsort((candidates, -sim[r, candidates]))][:k]
            neighbors[start + r] = order
            scores[start + r] = sim[r, order]
    return neighbors, scores
