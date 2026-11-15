"""MinHash 시그니처. 직접 구현했다.

집합 S 의 원소 x 마다 해시 함수 h_i(x) = (a_i * x + b_i) mod p 를 적용하고 그 최솟값을 시그니처의 i 번째 값으로 둔다.
두 집합의 시그니처에서 같은 위치의 값이 일치할 확률이 Jaccard 유사도와 같으므로,
일치하는 위치의 비율이 Jaccard 의 추정값이 된다 [Leskovec, Rajaraman, Ullman, Mining of Massive Datasets, 3장].
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from dmrec.ingredients import token_hash

PRIME = 2**31 - 1          # 메르센 소수. a, x < p 이므로 a * x < 2^62 로 uint64 안에 들어간다
EMPTY = np.uint32(PRIME)   # 빈 집합의 시그니처 값 (실제 해시값은 항상 p 미만)


class MinHasher:
    def __init__(self, num_perm: int, seed: int):
        self.num_perm = num_perm
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, tokens) -> np.ndarray:
        """재료 이름 집합 하나의 시그니처 (길이 num_perm, uint32)."""
        tokens = set(tokens)
        if not tokens:
            return np.full(self.num_perm, EMPTY, dtype=np.uint32)
        x = np.array([token_hash(t) for t in tokens], dtype=np.uint64)
        hashed = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(PRIME)
        return hashed.min(axis=1).astype(np.uint32)

    def signatures(self, matrix: csr_matrix, vocabulary: list[str]) -> np.ndarray:
        """(레시피 x 재료) 희소 행렬의 모든 행에 대한 시그니처 (n x num_perm, uint32).

        해시 함수 하나마다 모든 원소의 해시값을 한 번에 계산하고 행별 최솟값을 구한다.
        """
        matrix = matrix.tocsr()
        n = matrix.shape[0]
        x = np.array([token_hash(t) for t in vocabulary], dtype=np.uint64)[matrix.indices]
        empty = np.diff(matrix.indptr) == 0
        out = np.full((n, self.num_perm), EMPTY, dtype=np.uint32)
        # reduceat 의 구간은 "이 시작점부터 다음 시작점 전까지"다.
        # 빈 행의 시작점을 넣으면 구간이 어긋나므로 비어 있지 않은 행의 시작점만 넘긴다.
        starts = matrix.indptr[:-1][~empty]
        if len(starts) == 0:
            return out
        filled = np.empty((len(starts), self.num_perm), dtype=np.uint32)
        for i in range(self.num_perm):
            hashed = (self.a[i] * x + self.b[i]) % np.uint64(PRIME)
            filled[:, i] = np.minimum.reduceat(hashed, starts)
        out[~empty] = filled
        return out


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray | float:
    """시그니처에서 Jaccard 를 추정한다. 마지막 축이 시그니처다. 빈 집합이 끼면 0 이다."""
    match = (sig_a == sig_b) & (sig_a != EMPTY)
    return match.mean(axis=-1)
