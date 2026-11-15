"""MinHash 시그니처에 대한 LSH (banding). 직접 구현했다.

시그니처의 앞 b*r 개 값을 r 개씩 b 개의 band 로 나눈다. 어떤 band 에서든 r 개 값이 모두 같은 두 항목을 후보 쌍으로 본다.
Jaccard 가 s 인 두 항목이 후보가 될 확률은 1 - (1 - s^r)^b 다 [Leskovec, Rajaraman, Ullman, Mining of Massive Datasets, 3.4절].
"""

from __future__ import annotations

import numpy as np

from dmrec.minhash import EMPTY, estimate_jaccard

_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def collision_probability(similarity, bands: int, rows: int):
    """Jaccard 가 similarity 인 쌍이 적어도 한 band 에서 일치할 확률 (S-curve)."""
    similarity = np.asarray(similarity, dtype=np.float64)
    return 1.0 - (1.0 - similarity**rows) ** bands


def approximate_threshold(bands: int, rows: int) -> float:
    """S-curve 가 가파르게 오르는 지점의 근사값 (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows)


def band_keys(signatures: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """각 band 의 r 개 값을 64비트 키 하나로 접는다. 결과는 (n, bands) uint64.

    band 번호를 초깃값에 섞어서 서로 다른 band 의 키가 겹치지 않게 한다.
    값이 다른데 키가 같을 확률은 쌍마다 2^-64 수준이고, 그런 경우에도 후보가 하나 늘 뿐이다.
    """
    signatures = np.atleast_2d(signatures)
    if signatures.shape[1] < bands * rows:
        raise ValueError(f"시그니처 길이 {signatures.shape[1]} 가 bands * rows ({bands} * {rows}) 보다 짧다")
    # 시그니처가 더 길면 앞의 bands * rows 개만 banding 에 쓴다. 순위는 시그니처 전체로 매긴다.
    grouped = signatures[:, :bands * rows].reshape(signatures.shape[0], bands, rows).astype(np.uint64)
    keys = np.broadcast_to(np.arange(1, bands + 1, dtype=np.uint64), (signatures.shape[0], bands)).copy()
    for j in range(rows):
        keys = keys * _MULTIPLIER + grouped[:, :, j]     # uint64 에서 넘치면 2^64 로 나눈 나머지가 된다
    return keys


class LSHIndex:
    """시그니처 행렬의 banding 인덱스.

    항목은 0 부터의 행 번호로 가리킨다. 빈 집합(시그니처가 EMPTY)은 인덱스에 넣지 않는다.
    """

    def __init__(self, bands: int, rows: int):
        if bands < 1 or rows < 1:
            raise ValueError("bands 와 rows 는 1 이상이어야 한다")
        self.bands = bands
        self.rows = rows
        self.signatures: np.ndarray | None = None

    def build(self, signatures: np.ndarray) -> "LSHIndex":
        signatures = np.asarray(signatures)
        keys = band_keys(signatures, self.bands, self.rows)
        self.signatures = signatures
        self._indexed = signatures[:, 0] != EMPTY
        members = np.flatnonzero(self._indexed)
        # band 마다: 키로 정렬한 행 번호(_order), 서로 다른 키(_unique), 각 키의 구간 시작(_starts), 각 행의 버킷 번호(_bucket)
        self._order, self._unique, self._starts = [], [], []
        self._bucket = np.full((signatures.shape[0], self.bands), -1, dtype=np.int64)
        for band in range(self.bands):
            band_key = keys[members, band]
            order = np.argsort(band_key, kind="stable")
            unique, starts, inverse = np.unique(band_key[order], return_index=True, return_inverse=True)
            self._order.append(members[order])
            self._unique.append(unique)
            self._starts.append(np.append(starts, len(order)))
            self._bucket[members[order], band] = inverse
        return self

    # ---------- 후보 ----------

    def _bucket_members(self, band: int, bucket: int) -> np.ndarray:
        starts = self._starts[band]
        return self._order[band][starts[bucket]:starts[bucket + 1]]

    def candidates(self, row: int) -> np.ndarray:
        """인덱스에 있는 행 row 와 적어도 한 band 가 같은 다른 행들 (오름차순)."""
        if not self._indexed[row]:
            return np.empty(0, dtype=np.int64)
        parts = [self._bucket_members(band, self._bucket[row, band]) for band in range(self.bands)]
        found = np.unique(np.concatenate(parts))
        return found[found != row]

    def candidates_for_signature(self, signature: np.ndarray) -> np.ndarray:
        """인덱스에 없는 시그니처에 대한 후보 행들 (오름차순)."""
        signature = np.asarray(signature)
        if signature[0] == EMPTY:
            return np.empty(0, dtype=np.int64)
        keys = band_keys(signature, self.bands, self.rows)[0]
        parts = []
        for band in range(self.bands):
            unique = self._unique[band]
            pos = np.searchsorted(unique, keys[band])
            if pos < len(unique) and unique[pos] == keys[band]:
                parts.append(self._bucket_members(band, pos))
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))

    # ---------- 검색 ----------

    def _rank(self, signature: np.ndarray, found: np.ndarray, top_n: int | None, min_similarity: float):
        estimates = estimate_jaccard(self.signatures[found], signature[None, :])
        keep = estimates >= min_similarity
        found, estimates = found[keep], estimates[keep]
        order = np.lexsort((found, -estimates))          # 추정 유사도 내림차순, 같으면 행 번호 오름차순
        if top_n is not None:
            order = order[:top_n]
        return found[order], estimates[order]

    def query(self, row: int, top_n: int | None = None, min_similarity: float = 0.0):
        """행 row 와 비슷한 행을 (행 번호, 추정 Jaccard) 로 돌려준다. 추정 Jaccard 내림차순이다.

        후보는 LSH 로 찾고, 순위는 시그니처 전체로 추정한 Jaccard 로 매긴다.
        """
        return self._rank(self.signatures[row], self.candidates(row), top_n, min_similarity)

    def query_signature(self, signature: np.ndarray, top_n: int | None = None, min_similarity: float = 0.0):
        """인덱스에 없는 시그니처로 검색한다."""
        signature = np.asarray(signature)
        return self._rank(signature, self.candidates_for_signature(signature), top_n, min_similarity)

    # ---------- 통계 ----------

    def bucket_sizes(self) -> np.ndarray:
        """모든 band 의 버킷 크기를 이어 붙인 배열."""
        return np.concatenate([np.diff(starts) for starts in self._starts])

    def candidate_pair_count(self) -> int:
        """서로 다른 후보 쌍의 수. 여러 band 에서 겹치는 쌍은 한 번만 센다."""
        return int(sum(len(self.candidates(row)) for row in np.flatnonzero(self._indexed)) // 2)
