import numpy as np

from dmrec.ingredients import IngredientNormalizer, build_plural_map, to_csr, token_hash


def test_plural_map_only_folds_when_singular_exists():
    mapping = build_plural_map(["egg", "eggs", "tomato", "tomatoes", "berry", "berries", "molasses", "asparagus", "peas"])
    assert mapping == {"eggs": "egg", "tomatoes": "tomato", "berries": "berry"}


def test_normalizer_cleans_folds_and_applies_aliases():
    recipes = [["eggs", "kosher  salt "], ["egg", "salt and pepper"], ["garlic cloves", "garlic clove", "garlic"]]
    norm = IngredientNormalizer(aliases={"kosher salt": "salt", "garlic clove": "garlic", "salt and pepper": ["salt", "pepper"]}).fit(recipes)
    assert norm.transform(recipes[0]) == ["egg", "salt"]
    assert norm.transform(recipes[1]) == ["egg", "pepper", "salt"]
    assert norm.transform(recipes[2]) == ["garlic"]          # 복수형을 접은 뒤 별칭을 적용한다


def test_high_frequency_ingredients_are_removed():
    recipes = [["salt", "a"], ["salt", "b"], ["salt", "c"], ["salt"]]
    norm = IngredientNormalizer(max_doc_freq=0.5).fit(recipes)
    assert norm.stop_ingredients == ["salt"]
    assert norm.transform(recipes[0]) == ["a"]
    assert norm.transform(recipes[3]) == []                   # 고빈도 재료만 있던 레시피


def test_no_rules_keeps_names():
    norm = IngredientNormalizer(fold_plurals=False).fit([["eggs"], ["egg"]])
    assert norm.transform(["eggs", "egg"]) == ["egg", "eggs"]


def test_token_hash_is_stable_and_in_range():
    assert token_hash("salt") == token_hash("salt")
    assert token_hash("salt") != token_hash("sugar")
    assert all(0 <= token_hash(t) < 2**31 - 1 for t in ["salt", "olive oil", ""])


def test_to_csr_builds_binary_rows():
    matrix, vocab = to_csr([["b", "a"], [], ["a"]])
    assert vocab == ["a", "b"]
    assert matrix.toarray().tolist() == [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
    assert np.diff(matrix.indptr).tolist() == [2, 0, 1]
