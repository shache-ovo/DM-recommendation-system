import numpy as np
import pytest

from dmrec.ingredients import to_csr
from dmrec.jaccard import jaccard
from dmrec.minhash import EMPTY, MinHasher, estimate_jaccard


def test_same_seed_gives_same_signature_and_different_seed_differs():
    tokens = ["salt", "sugar", "egg"]
    assert np.array_equal(MinHasher(64, 1).signature(tokens), MinHasher(64, 1).signature(tokens))
    assert not np.array_equal(MinHasher(64, 1).signature(tokens), MinHasher(64, 2).signature(tokens))


def test_signature_ignores_order_and_duplicates():
    hasher = MinHasher(64, 0)
    assert np.array_equal(hasher.signature(["a", "b", "c"]), hasher.signature(["c", "a", "b", "a"]))


def test_identical_sets_estimate_one_and_disjoint_sets_estimate_near_zero():
    hasher = MinHasher(256, 0)
    a = hasher.signature([f"x{i}" for i in range(20)])
    b = hasher.signature([f"y{i}" for i in range(20)])
    assert estimate_jaccard(a, a) == 1.0
    assert estimate_jaccard(a, b) < 0.05


def test_empty_set_gets_sentinel_and_estimates_zero():
    hasher = MinHasher(32, 0)
    empty = hasher.signature([])
    assert (empty == EMPTY).all()
    assert estimate_jaccard(empty, empty) == 0.0
    assert estimate_jaccard(empty, hasher.signature(["a"])) == 0.0


def test_batch_signatures_equal_single_signatures():
    sets = [["a", "b", "c"], [], ["c"], ["d", "a"], []]
    matrix, vocab = to_csr(sets)
    hasher = MinHasher(48, 3)
    batch = hasher.signatures(matrix, vocab)
    for row, tokens in enumerate(sets):
        assert np.array_equal(batch[row], hasher.signature(tokens))


def test_estimate_is_unbiased_within_sampling_error():
    # 크기 10 인 집합 쌍을 겹침 0~10 으로 만들어 정확 Jaccard 와 비교한다.
    # 추정값의 표준오차는 sqrt(J(1-J)/k). 200쌍 평균의 오차는 그 1/sqrt(200) 이다.
    k, pairs_per_level = 256, 200
    hasher = MinHasher(k, 7)
    counter = 0
    for overlap in range(0, 11):
        estimates = []
        for _ in range(pairs_per_level):
            shared = [f"s{counter}_{i}" for i in range(overlap)]
            a = shared + [f"a{counter}_{i}" for i in range(10 - overlap)]
            b = shared + [f"b{counter}_{i}" for i in range(10 - overlap)]
            counter += 1
            estimates.append(estimate_jaccard(hasher.signature(a), hasher.signature(b)))
        exact = jaccard(set(a), set(b))
        standard_error = np.sqrt(exact * (1 - exact) / k / pairs_per_level)
        assert abs(np.mean(estimates) - exact) <= 4 * standard_error + 1e-3


def test_error_matches_datasketch():
    """검증용 라이브러리와 추정 오차 수준을 비교한다. datasketch 가 없으면 건너뛴다."""
    datasketch = pytest.importorskip("datasketch")
    rng = np.random.default_rng(0)
    k = 128
    hasher = MinHasher(k, 0)
    ours, theirs = [], []
    for _ in range(300):
        a = {f"t{v}" for v in rng.choice(60, size=12, replace=False)}
        b = {f"t{v}" for v in rng.choice(60, size=12, replace=False)}
        exact = jaccard(a, b)
        ours.append(estimate_jaccard(hasher.signature(a), hasher.signature(b)) - exact)
        ma, mb = datasketch.MinHash(num_perm=k, seed=1), datasketch.MinHash(num_perm=k, seed=1)
        for t in a:
            ma.update(t.encode("utf-8"))
        for t in b:
            mb.update(t.encode("utf-8"))
        theirs.append(ma.jaccard(mb) - exact)
    ours_rmse, theirs_rmse = np.sqrt(np.mean(np.square(ours))), np.sqrt(np.mean(np.square(theirs)))
    assert abs(np.mean(ours)) < 0.01
    assert ours_rmse < 1.3 * theirs_rmse
