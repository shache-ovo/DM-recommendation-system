import numpy as np
import pytest

from dmrec.ingredients import to_csr
from dmrec.jaccard import jaccard
from dmrec.minhash import MinHasher
from dmrec.similar import SimilarRecipes

SETS = {
    101: ["egg", "flour", "milk", "sugar", "butter"],
    102: ["egg", "flour", "milk", "sugar", "butter"],          # 101 과 같다
    103: ["egg", "flour", "milk", "sugar", "vanilla"],         # 101 과 Jaccard 4/6
    104: ["beef", "onion", "garlic", "tomato"],
    105: ["beef", "onion", "garlic", "tomato", "pepper"],      # 104 와 Jaccard 4/5
    106: ["tofu", "soy sauce"],
}


IDS = np.array(list(SETS))


def make(exact: bool, bands: int = 32, rows: int = 2) -> SimilarRecipes:
    hasher = MinHasher(64, 0)
    signatures = np.array([hasher.signature(SETS[i]) for i in IDS])
    matrix = to_csr([SETS[i] for i in IDS])[0] if exact else None
    return SimilarRecipes(IDS, signatures, bands, rows, matrix)


@pytest.fixture(params=[False, True], ids=["estimate", "exact"])
def finder(request):
    return make(request.param)


def test_similar_returns_recipe_ids_sorted_by_similarity(finder):
    result = finder.similar(101)
    assert result[0] == (102, 1.0)
    assert [rid for rid, _ in result][:2] == [102, 103]
    assert all(a[1] >= b[1] for a, b in zip(result, result[1:]))
    assert 101 not in [rid for rid, _ in result]
    assert finder.similar(101, top_n=1) == [(102, 1.0)]
    assert all(score >= 0.9 for _, score in finder.similar(101, min_jaccard=0.9))


def test_unknown_recipe_raises_and_membership_works(finder):
    assert 101 in finder and 999 not in finder and len(finder) == 6
    with pytest.raises(KeyError):
        finder.similar(999)


def test_candidates_for_merges_sources_and_excludes(finder):
    merged = finder.candidates_for([101, 104])
    assert 101 not in merged and 104 not in merged          # 입력 레시피는 빠진다
    assert merged[102] == 1.0 and 105 in merged
    assert 102 not in finder.candidates_for([101, 104], exclude=[102])
    assert finder.candidates_for([999]) == {}                # 인덱스에 없는 입력은 건너뛴다
    # 여러 입력과 닿는 후보는 가장 큰 유사도를 갖는다
    both = finder.candidates_for([101, 102])
    assert both[103] == max(dict(finder.similar(101))[103], dict(finder.similar(102))[103])


def test_neighbor_table_has_expected_columns_and_ranks(finder):
    table = finder.neighbor_table(top_n=2)
    assert list(table.columns) == ["recipe_id", "neighbor_recipe_id", "rank", "jaccard", "est_jaccard"]
    assert table.groupby("recipe_id")["rank"].max().max() <= 2
    first = table[table.recipe_id == 101].iloc[0]
    assert (first.neighbor_recipe_id, first["rank"], first.jaccard, first.est_jaccard) == (102, 1, 1.0, 1.0)
    assert not (table.recipe_id == table.neighbor_recipe_id).any()


def test_exact_ranking_returns_true_jaccard_of_candidates():
    finder = make(exact=True)
    for rid in IDS:
        result = finder.similar(int(rid))
        for other, score in result:
            assert np.isclose(score, jaccard(set(SETS[int(rid)]), set(SETS[other])))
        assert [s for _, s in result] == sorted((s for _, s in result), reverse=True)
    assert dict(finder.similar(101))[103] == 4 / 6
    assert dict(finder.similar(104))[105] == 4 / 5
    # 정확값으로 걸러낸다: 0.7 이상은 105 뿐이다
    assert [rid for rid, _ in finder.similar(104, min_jaccard=0.7)] == [105]


def test_exact_and_estimate_share_the_same_candidates():
    exact, estimate = make(True), make(False)
    for rid in IDS:
        assert {r for r, _ in exact.similar(int(rid))} == {r for r, _ in estimate.similar(int(rid))}
