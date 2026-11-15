"""레시피의 재료 이름을 정규화해 MinHash 입력(재료 집합)을 만든다.

정규화 순서:
  1. 공백 정리 (앞뒤 공백 제거, 연속 공백을 하나로)
  2. 복수형 접기: 이름 끝의 s / es / ies 를 떼거나 바꾼 형태가 어휘에 이미 있을 때만 그 형태로 바꾼다
  3. 별칭: 설정에 적힌 이름을 대표 이름으로 바꾸거나 여러 재료로 나눈다
  4. 고빈도 재료 제거: 문서 빈도가 기준 이상인 재료를 집합에서 뺀다 (선택)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix


def clean(name: str) -> str:
    return " ".join(name.split())


def singular_candidates(name: str) -> list[str]:
    out = []
    if name.endswith("ies"):
        out.append(name[:-3] + "y")
    if name.endswith("es"):
        out.append(name[:-2])
    if name.endswith("s"):
        out.append(name[:-1])
    return out


def build_plural_map(vocabulary: Iterable[str]) -> dict[str, str]:
    """복수형 -> 단수형 사전. 단수형이 어휘에 실제로 있을 때만 만든다 (molasses, asparagus 같은 이름을 건드리지 않기 위해)."""
    vocab = set(vocabulary)
    mapping = {}
    for name in vocab:
        for candidate in singular_candidates(name):
            if candidate in vocab:
                mapping[name] = candidate
                break
    return mapping


class IngredientNormalizer:
    """재료 이름 정규화 규칙. fit 으로 어휘에서 규칙을 만들고 transform 으로 적용한다."""

    def __init__(self, fold_plurals: bool = True, aliases: dict[str, Any] | None = None, max_doc_freq: float | None = None):
        self.fold_plurals = fold_plurals
        # 별칭 값은 문자열(대표 이름) 또는 문자열 리스트(여러 재료로 나눔)
        self.aliases = {k: ([v] if isinstance(v, str) else list(v)) for k, v in (aliases or {}).items()}
        self.max_doc_freq = max_doc_freq
        self.plural_map: dict[str, str] = {}
        self.stop_ingredients: list[str] = []

    def _rename(self, name: str) -> list[str]:
        name = clean(name)
        name = self.plural_map.get(name, name)
        return self.aliases.get(name, [name])

    def _rename_all(self, names: Iterable[str]) -> set[str]:
        return {out for name in names for out in self._rename(name)}

    def fit(self, recipes: Iterable[Iterable[str]]) -> "IngredientNormalizer":
        recipes = list(recipes)
        if self.fold_plurals:
            self.plural_map = build_plural_map(clean(n) for names in recipes for n in names)
        if self.max_doc_freq is not None:
            doc_freq = Counter(n for names in recipes for n in self._rename_all(names))
            limit = self.max_doc_freq * len(recipes)
            self.stop_ingredients = sorted(n for n, c in doc_freq.items() if c >= limit)
        return self

    def transform(self, names: Iterable[str]) -> list[str]:
        """정규화한 재료 집합을 정렬된 리스트로 돌려준다. 고빈도 재료만 있던 레시피는 빈 리스트가 된다."""
        result = self._rename_all(names)
        if self.stop_ingredients:
            result -= set(self.stop_ingredients)
        return sorted(result)


def normalizer_from_config(config: dict[str, Any]) -> IngredientNormalizer:
    cfg = config["ingredients"]
    return IngredientNormalizer(cfg["fold_plurals"], cfg["aliases"], cfg["max_doc_freq"])


def token_hash(token: str) -> int:
    """재료 이름을 0 이상 2^31-1 미만의 정수로 바꾼다. 파이썬 내장 hash 와 달리 실행마다 같은 값이다."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def to_csr(sets: list[list[str]]) -> tuple[csr_matrix, list[str]]:
    """재료 집합들을 (레시피 x 재료) 0/1 희소 행렬로 바꾼다. 열 순서는 재료 이름의 사전순이다."""
    vocabulary = sorted({t for s in sets for t in s})
    index = {t: i for i, t in enumerate(vocabulary)}
    indptr = np.zeros(len(sets) + 1, dtype=np.int64)
    indices = []
    for row, tokens in enumerate(sets):
        cols = sorted(index[t] for t in set(tokens))
        indices.extend(cols)
        indptr[row + 1] = len(indices)
    data = np.ones(len(indices), dtype=np.float32)
    matrix = csr_matrix((data, np.array(indices, dtype=np.int32), indptr), shape=(len(sets), len(vocabulary)))
    return matrix, vocabulary
