import numpy as np

from dmrec.ingredients import to_csr
from dmrec.jaccard import jaccard, pairwise_jaccard, top_k_neighbors

SETS = [["a", "b", "c"], ["a", "b"], ["c", "d"], ["x"], []]


def test_jaccard_of_sets():
    assert jaccard({"a", "b", "c"}, {"a", "b"}) == 2 / 3
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, {"a"}) == 1.0


def test_pairwise_matches_set_definition():
    matrix, _ = to_csr(SETS)
    left = np.array([0, 0, 0, 3, 4])
    right = np.array([1, 2, 3, 4, 4])
    expected = [jaccard(set(SETS[i]), set(SETS[j])) for i, j in zip(left, right)]
    assert np.allclose(pairwise_jaccard(matrix, left, right), expected)


def test_top_k_neighbors_matches_brute_force():
    rng = np.random.default_rng(0)
    sets = [list(rng.choice(30, size=rng.integers(1, 8), replace=False).astype(str)) for _ in range(60)]
    matrix, _ = to_csr(sets)
    neighbors, scores = top_k_neighbors(matrix, 5, chunk=7)
    for i in range(len(sets)):
        sims = [(-jaccard(set(sets[i]), set(sets[j])), j) for j in range(len(sets)) if j != i]
        expected = sorted(sims)[:5]
        assert [j for _, j in expected] == neighbors[i].tolist()
        assert np.allclose([-s for s, _ in expected], scores[i])
