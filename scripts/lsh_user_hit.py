"""재료 유사 레시피를 추천 후보로 썼을 때 validation 정답을 얼마나 포함하는지 잰다.

사용자가 train 에서 좋아한 레시피마다 상위 N 개 이웃을 모아 후보로 삼는다. 이웃을 찾는 방법 세 가지를 비교한다.
  - lsh:        설정된 (b, r) 의 LSH 후보를 정확한 Jaccard 로 정렬한 상위 N 개
  - exact:      전수 비교로 구한 정확한 상위 N 개 (LSH 가 놓치는 것이 없는 경우)
  - popularity: 같은 수의 후보를 train 인기순으로 뽑은 경우 (재료 정보를 쓰지 않는 기준선)
후보에서 사용자가 이미 본 레시피(train 양성, 비양성 기록)는 뺀다. test 는 쓰지 않는다.

사용법:
    python3 scripts/lsh_user_hit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.jaccard import top_k_neighbors  # noqa: E402
from dmrec.similar import SimilarRecipes  # noqa: E402

N_VALUES = [5, 10, 20, 50, 100]


def main() -> None:
    config = load_config()
    processed = resolve_path(config, "processed_dir")
    finder = SimilarRecipes.from_processed(processed, config, scope="core", rank_by="exact")
    n = len(finder)
    max_n = max(N_VALUES)

    # 이웃 행렬 (행 번호 == recipe_idx). 이웃이 모자라면 -1 로 채운다.
    lsh = np.full((n, max_n), -1, dtype=np.int64)
    for row in range(n):
        found, _, _ = finder._query_row(row, max_n, 0.0)
        lsh[row, :len(found)] = found
    exact, _ = top_k_neighbors(finder.matrix, max_n)

    idx = {int(rid): i for i, rid in enumerate(finder.recipe_ids)}
    train = pd.read_csv(processed / "interactions_train.csv")
    valid = pd.read_csv(processed / "interactions_valid.csv")
    seen = pd.read_csv(processed / "interactions_seen_non_positive.csv")
    train_items = train.assign(i=train["recipe_id"].map(idx)).groupby("user_id")["i"].agg(list)
    seen_items = pd.concat([train, seen]).assign(i=lambda d: d["recipe_id"].map(idx)).groupby("user_id")["i"].agg(set)
    target = valid.assign(i=valid["recipe_id"].map(idx)).set_index("user_id")["i"]
    popular = np.argsort(-np.bincount(train["recipe_id"].map(idx), minlength=n), kind="stable")

    rows = []
    for top in N_VALUES:
        result = {"neighbors_per_liked_recipe": top}
        sizes = {}
        for name, table in (("lsh", lsh), ("exact", exact)):
            hits, counts = 0, []
            for user, t in target.items():
                cand = set(table[train_items[user], :top].ravel().tolist()) - seen_items[user]
                cand.discard(-1)
                counts.append(len(cand))
                hits += t in cand
            sizes[name] = counts
            result[f"{name}_hit_rate"] = round(hits / len(target), 4)
            result[f"{name}_candidates_median"] = float(np.median(counts))
        # 인기순: 사용자마다 LSH 후보와 같은 수만큼, 이미 본 것을 빼고 뽑는다
        hits = 0
        for (user, t), size in zip(target.items(), sizes["lsh"]):
            if size == 0:
                continue
            skip = seen_items[user]
            picked = [p for p in popular[:size + len(skip)].tolist() if p not in skip][:size]
            hits += t in picked
        result["popularity_hit_rate_same_size"] = round(hits / len(target), 4)
        rows.append(result)

    table = pd.DataFrame(rows)
    table.insert(0, "bands", config["lsh"]["bands"])
    table.insert(1, "rows", config["lsh"]["rows"])
    table.to_csv(REPO_ROOT / "reports" / "tables" / "lsh_user_hit.csv", index=False)
    print(table.T.to_string())
    print("users evaluated:", len(target), "| core recipes:", n)


if __name__ == "__main__":
    main()
