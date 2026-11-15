import numpy as np
import pytest

from dmrec.lsh import LSHIndex, approximate_threshold, band_keys, collision_probability
from dmrec.minhash import EMPTY, MinHasher, estimate_jaccard


def brute_force_candidates(signatures, bands, rows, row):
    """정의 그대로: 어떤 band 에서든 r 개 값이 모두 같은 다른 행."""
    banded = signatures.reshape(len(signatures), bands, rows)
    same = (banded == banded[row]).all(axis=2).any(axis=1)
    same[row] = False
    return np.flatnonzero(same & (signatures[:, 0] != EMPTY))


def random_signatures(n, length, seed, values=3):
    # 값의 종류를 적게 해서 band 가 우연히 일치하는 경우가 충분히 생기게 한다
    return np.random.default_rng(seed).integers(0, values, size=(n, length)).astype(np.uint32)


def test_collision_probability_and_threshold():
    assert collision_probability(0.0, 20, 5) == 0.0
    assert collision_probability(1.0, 20, 5) == 1.0
    assert np.isclose(collision_probability(0.5, 1, 1), 0.5)
    assert np.isclose(collision_probability(0.8, 20, 5), 1 - (1 - 0.8**5) ** 20)
    grid = np.linspace(0, 1, 50)
    assert (np.diff(collision_probability(grid, 32, 4)) >= 0).all()
    assert np.isclose(approximate_threshold(32, 4), (1 / 32) ** 0.25)


def test_band_keys_equal_exactly_when_band_values_equal():
    sig = random_signatures(200, 12, 0)
    keys = band_keys(sig, 4, 3)
    banded = sig.reshape(200, 4, 3)
    for band in range(4):
        same_values = (banded[:, None, band] == banded[None, :, band]).all(axis=2)
        assert np.array_equal(keys[:, band][:, None] == keys[:, band][None, :], same_values)
    # 같은 값이라도 band 가 다르면 키가 다르다
    constant = np.zeros((1, 12), dtype=np.uint32)
    assert len(set(band_keys(constant, 4, 3)[0].tolist())) == 4


def test_signature_must_be_long_enough_and_extra_columns_only_affect_ranking():
    with pytest.raises(ValueError):
        band_keys(np.zeros((2, 10), dtype=np.uint32), 4, 3)
    sig = random_signatures(200, 16, 9)
    longer = LSHIndex(4, 3).build(sig)                      # 앞 12 개만 banding 에 쓴다
    exact = LSHIndex(4, 3).build(sig[:, :12])
    assert all(np.array_equal(longer.candidates(r), exact.candidates(r)) for r in range(200))
    rows, estimates = longer.query(0)
    assert np.allclose(estimates, estimate_jaccard(sig[rows], sig[0][None, :]))   # 순위는 16 개 전체로
    with pytest.raises(ValueError):
        LSHIndex(0, 3)


@pytest.mark.parametrize("bands,rows", [(4, 3), (6, 2), (12, 1), (1, 12)])
def test_candidates_match_definition(bands, rows):
    sig = random_signatures(300, 12, 1)
    index = LSHIndex(bands, rows).build(sig)
    for row in range(len(sig)):
        assert np.array_equal(index.candidates(row), brute_force_candidates(sig, bands, rows, row))


def test_identical_signatures_are_always_candidates_and_self_is_excluded():
    sig = random_signatures(50, 16, 2, values=1000)
    sig[7] = sig[3]
    index = LSHIndex(4, 4).build(sig)
    assert 7 in index.candidates(3) and 3 in index.candidates(7)
    assert 3 not in index.candidates(3)


def test_empty_sets_are_not_indexed():
    sig = random_signatures(20, 8, 3)
    sig[5] = EMPTY
    sig[9] = EMPTY
    index = LSHIndex(4, 2).build(sig)
    assert len(index.candidates(5)) == 0
    assert all(5 not in index.candidates(r) and 9 not in index.candidates(r) for r in range(20))
    assert len(index.candidates_for_signature(np.full(8, EMPTY, dtype=np.uint32))) == 0


def test_query_ranks_by_estimated_jaccard_then_row():
    sig = random_signatures(300, 12, 4)
    index = LSHIndex(4, 3).build(sig)
    rows, estimates = index.query(0)
    assert np.array_equal(np.sort(rows), index.candidates(0))
    assert np.allclose(estimates, estimate_jaccard(sig[rows], sig[0][None, :]))
    assert list(zip(-estimates, rows)) == sorted(zip(-estimates, rows))
    top_rows, top_estimates = index.query(0, top_n=5)
    assert np.array_equal(top_rows, rows[:5]) and np.array_equal(top_estimates, estimates[:5])
    filtered_rows, filtered = index.query(0, min_similarity=0.5)
    assert (filtered >= 0.5).all() and len(filtered_rows) == (estimates >= 0.5).sum()


def test_external_signature_query_matches_indexed_query():
    sig = random_signatures(300, 12, 5)
    index = LSHIndex(4, 3).build(sig)
    external = index.candidates_for_signature(sig[10])
    assert 10 in external                                   # 인덱스 안의 같은 시그니처는 자기 자신도 나온다
    assert np.array_equal(external[external != 10], index.candidates(10))
    unseen = np.full(12, 999, dtype=np.uint32)
    assert len(index.candidates_for_signature(unseen)) == 0
    assert len(index.query_signature(unseen)[0]) == 0


def test_build_is_deterministic_and_stats_are_consistent():
    sig = random_signatures(200, 12, 6)
    a, b = LSHIndex(4, 3).build(sig), LSHIndex(4, 3).build(sig)
    assert all(np.array_equal(a.candidates(r), b.candidates(r)) for r in range(200))
    assert a.bucket_sizes().sum() == 200 * 4                # 모든 행이 band 마다 버킷 하나에 들어간다
    pairs = sum(len(brute_force_candidates(sig, 4, 3, r)) for r in range(200)) // 2
    assert a.candidate_pair_count() == pairs


def test_candidate_rate_follows_s_curve():
    # Jaccard 가 정해진 집합 쌍을 많이 만들어 후보가 되는 비율을 이론값과 비교한다.
    bands, rows, pairs = 16, 4, 400
    hasher = MinHasher(bands * rows, 11)
    counter = 0
    for overlap, total in ((2, 18), (5, 15), (8, 12)):       # Jaccard = 2/18, 5/15, 8/12
        signatures = []
        for _ in range(pairs):
            shared = [f"s{counter}_{i}" for i in range(overlap)]
            rest = total - overlap
            a = shared + [f"a{counter}_{i}" for i in range(rest // 2)]
            b = shared + [f"b{counter}_{i}" for i in range(rest - rest // 2)]
            counter += 1
            signatures += [hasher.signature(a), hasher.signature(b)]
        index = LSHIndex(bands, rows).build(np.array(signatures))
        hit = np.mean([2 * i + 1 in index.candidates(2 * i) for i in range(pairs)])
        expected = collision_probability(overlap / total, bands, rows)
        tolerance = 4 * np.sqrt(expected * (1 - expected) / pairs) + 0.01
        assert abs(hit - expected) <= tolerance
