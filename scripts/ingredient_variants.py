"""재료 집합을 만드는 방법(정규화, 고빈도 재료 제거)을 비교한다. docs/minhash.md 에서 인용한다.

방법마다 추천 대상 레시피의 재료 집합을 만들고 전수 비교로 정확한 Jaccard 이웃을 구한 뒤 다음을 잰다.
  - 어휘 크기, 집합 크기, 빈 집합 수
  - 무작위 쌍의 평균 Jaccard (낮을수록 무관한 레시피가 덜 닮아 보인다)
  - 상위 10개 이웃과의 태그 Jaccard (재료가 비슷한 레시피가 실제로 비슷한 종류인가)
  - validation 적중률: 사용자가 train 에서 좋아한 레시피들의 상위 20개 이웃 안에
    그 사용자의 validation 정답이 들어 있는 비율 (test 는 쓰지 않는다)

사용법:
    python3 scripts/ingredient_variants.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.ingredients import IngredientNormalizer, to_csr  # noqa: E402
from dmrec.jaccard import pairwise_jaccard, top_k_neighbors  # noqa: E402

TOP_TAG = 10
TOP_HIT = 20


def main() -> None:
    config = load_config()
    seed = config["seed"]
    aliases = config["ingredients"]["aliases"]
    processed = resolve_path(config, "processed_dir")
    recipes = pd.read_csv(processed / "recipes.csv", usecols=["recipe_id", "recipe_idx", "ingredients", "tags"])
    all_ingredients = recipes["ingredients"].map(json.loads).tolist()
    core = recipes[recipes["recipe_idx"] >= 0].sort_values("recipe_idx").reset_index(drop=True)
    core_ingredients = core["ingredients"].map(json.loads).tolist()
    tag_matrix, _ = to_csr(core["tags"].map(json.loads).tolist())
    n = len(core)

    train = pd.read_csv(processed / "interactions_train.csv")
    valid = pd.read_csv(processed / "interactions_valid.csv")
    idx = dict(zip(core["recipe_id"], core["recipe_idx"]))
    train_items = train.assign(i=train["recipe_id"].map(idx)).groupby("user_id")["i"].agg(list)
    valid_item = valid.assign(i=valid["recipe_id"].map(idx)).set_index("user_id")["i"]

    rng = np.random.default_rng(seed)
    rand_left, rand_right = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    keep = rand_left != rand_right
    rand_left, rand_right = rand_left[keep], rand_right[keep]

    variants = [
        ("raw", dict(fold_plurals=False, aliases=None, max_doc_freq=None)),
        ("plural", dict(fold_plurals=True, aliases=None, max_doc_freq=None)),
        ("plural+alias", dict(fold_plurals=True, aliases=aliases, max_doc_freq=None)),
        ("plural+alias, drop>=20%", dict(fold_plurals=True, aliases=aliases, max_doc_freq=0.20)),
        ("plural+alias, drop>=10%", dict(fold_plurals=True, aliases=aliases, max_doc_freq=0.10)),
        ("plural+alias, drop>=5%", dict(fold_plurals=True, aliases=aliases, max_doc_freq=0.05)),
    ]
    rows = []
    for name, params in variants:
        started = time.time()
        # 규칙(복수형 사전, 고빈도 재료)은 전체 레시피에서 만든다. 상호작용은 쓰지 않는다.
        normalizer = IngredientNormalizer(**params).fit(all_ingredients)
        sets = [normalizer.transform(names) for names in core_ingredients]
        matrix, vocabulary = to_csr(sets)
        sizes = np.diff(matrix.indptr)
        neighbors, scores = top_k_neighbors(matrix, TOP_HIT)

        left = np.repeat(np.arange(n), TOP_TAG)
        tag_agreement = pairwise_jaccard(tag_matrix, left, neighbors[:, :TOP_TAG].ravel()).mean()

        hits = 0
        candidate_sizes = []
        for user, target in valid_item.items():
            candidates = set(neighbors[train_items[user]].ravel().tolist())
            candidate_sizes.append(len(candidates))
            hits += target in candidates

        rows.append({
            "variant": name,
            "vocabulary": len(vocabulary),
            "removed_ingredients": len(normalizer.stop_ingredients),
            "mean_set_size": round(float(sizes.mean()), 2),
            "empty_sets": int((sizes == 0).sum()),
            "sets_with_lt_3": int((sizes < 3).sum()),
            "random_pair_jaccard": round(float(pairwise_jaccard(matrix, rand_left, rand_right).mean()), 4),
            "top1_jaccard_mean": round(float(scores[:, 0].mean()), 4),
            "top10_jaccard_mean": round(float(scores[:, :TOP_TAG].mean()), 4),
            "top10_tag_jaccard": round(float(tag_agreement), 4),
            "valid_hit_rate_top20": round(hits / len(valid_item), 4),
            "median_candidates_per_user": float(np.median(candidate_sizes)),
            "removed": ", ".join(normalizer.stop_ingredients),
            "seconds": round(time.time() - started, 1),
        })
        print(rows[-1], flush=True)

    # 비교 기준: 인기순 상위 N 개를 후보로 했을 때의 적중률 (N 은 위 후보 수의 중앙값과 같게)
    popular = core.merge(train["recipe_id"].value_counts().rename("n"), left_on="recipe_id", right_index=True).sort_values("n", ascending=False)["recipe_idx"].to_numpy()
    baseline = []
    for size in sorted({int(r["median_candidates_per_user"]) for r in rows}):
        top = set(popular[:size].tolist())
        seen = train_items
        hit = sum((t in top) and (t not in set(seen[u])) for u, t in valid_item.items()) / len(valid_item)
        baseline.append({"candidates": size, "popularity_hit_rate": round(hit, 4)})
    table = pd.DataFrame(rows)
    out = REPO_ROOT / "reports" / "tables"
    table.to_csv(out / "ingredient_variants.csv", index=False)
    pd.DataFrame(baseline).to_csv(out / "ingredient_variants_popularity_baseline.csv", index=False)
    print(table.drop(columns=["removed"]).T.to_string())
    print(pd.DataFrame(baseline).to_string(index=False))


if __name__ == "__main__":
    main()
