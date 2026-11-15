"""LSH 인덱스를 만들고 레시피마다 재료가 비슷한 레시피 목록을 data/processed/lsh_neighbors.csv 에 저장한다.

build_minhash.py 를 먼저 실행해야 한다.

사용법:
    python3 scripts/build_lsh.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.lsh import approximate_threshold  # noqa: E402
from dmrec.similar import SimilarRecipes  # noqa: E402


def main() -> None:
    config = load_config()
    cfg = config["lsh"]
    processed = resolve_path(config, "processed_dir")

    started = time.perf_counter()
    finder = SimilarRecipes.from_processed(processed, config)
    build_seconds = time.perf_counter() - started

    started = time.perf_counter()
    table = finder.neighbor_table(cfg["neighbors_top_n"], cfg["neighbors_min_jaccard"])
    query_seconds = time.perf_counter() - started

    # recipe_idx 를 함께 넣는다. core 범위에서는 행 번호가 곧 recipe_idx 다.
    data = np.load(processed / "minhash_signatures.npz")
    idx_of = dict(zip(data["recipe_id"].tolist(), data["recipe_idx"].tolist()))
    table.insert(1, "recipe_idx", table["recipe_id"].map(idx_of))
    table.insert(3, "neighbor_recipe_idx", table["neighbor_recipe_id"].map(idx_of))
    table.to_csv(processed / "lsh_neighbors.csv", index=False)

    per_recipe = table.groupby("recipe_id").size().reindex(finder.recipe_ids, fill_value=0)
    sizes = finder.index.bucket_sizes()
    n = len(finder)
    candidate_counts = np.array([len(finder.index.candidates(row)) for row in range(n)])
    summary = {
        "index_scope": cfg["index_scope"],
        "recipes_indexed": len(finder),
        "bands": cfg["bands"],
        "rows": cfg["rows"],
        "approximate_threshold": round(approximate_threshold(cfg["bands"], cfg["rows"]), 4),
        "neighbors_top_n": cfg["neighbors_top_n"],
        "neighbor_rows": len(table),
        "recipes_with_no_neighbor": int((per_recipe == 0).sum()),
        "recipes_with_full_top_n": int((per_recipe == cfg["neighbors_top_n"]).sum()),
        "neighbors_per_recipe_median": float(per_recipe.median()),
        "est_jaccard_of_rank1_median": round(float(table[table["rank"] == 1]["est_jaccard"].median()), 4),
        "candidates_per_recipe_median": float(np.median(candidate_counts)),
        "candidates_per_recipe_mean": round(float(candidate_counts.mean()), 1),
        "candidates_per_recipe_max": int(candidate_counts.max()),
        "candidate_pairs": int(candidate_counts.sum() // 2),
        "candidate_pairs_pct_of_all_pairs": round(float(candidate_counts.sum() / 2 / (n * (n - 1) / 2) * 100), 4),
        "buckets": int(len(sizes)),
        "bucket_size_max": int(sizes.max()),
        "bucket_size_mean": round(float(sizes.mean()), 3),
        "build_seconds": round(build_seconds, 2),
        "all_queries_seconds": round(query_seconds, 2),
        "ms_per_query": round(query_seconds / len(finder) * 1000, 3),
    }
    (REPO_ROOT / "reports" / "tables" / "lsh_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
