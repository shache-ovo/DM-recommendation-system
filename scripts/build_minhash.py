"""정규화한 재료 집합과 MinHash 시그니처를 만들어 data/processed 에 저장한다.

산출물:
  ingredient_sets.csv        recipe_id, recipe_idx, n_ingredients_norm, ingredients_norm (JSON 리스트)
  minhash_signatures.npz     recipe_id, recipe_idx, signatures (n x num_perm, uint32), num_perm, seed
행 순서는 recipes.csv 와 같다.

사용법:
    python3 scripts/build_minhash.py
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
from dmrec.ingredients import normalizer_from_config, to_csr  # noqa: E402
from dmrec.minhash import MinHasher  # noqa: E402


def main() -> None:
    config = load_config()
    processed = resolve_path(config, "processed_dir")
    recipes = pd.read_csv(processed / "recipes.csv", usecols=["recipe_id", "recipe_idx", "ingredients"])
    raw = recipes["ingredients"].map(json.loads).tolist()

    started = time.time()
    normalizer = normalizer_from_config(config).fit(raw)
    sets = [normalizer.transform(names) for names in raw]
    matrix, vocabulary = to_csr(sets)
    normalize_seconds = time.time() - started

    started = time.time()
    hasher = MinHasher(config["minhash"]["num_perm"], config["seed"])
    signatures = hasher.signatures(matrix, vocabulary)
    signature_seconds = time.time() - started

    out = recipes[["recipe_id", "recipe_idx"]].copy()
    out["n_ingredients_norm"] = [len(s) for s in sets]
    out["ingredients_norm"] = [json.dumps(s, ensure_ascii=False) for s in sets]
    out.to_csv(processed / "ingredient_sets.csv", index=False)
    np.savez(processed / "minhash_signatures.npz", recipe_id=recipes["recipe_id"].to_numpy(), recipe_idx=recipes["recipe_idx"].to_numpy(),
             signatures=signatures, num_perm=hasher.num_perm, seed=hasher.seed)

    summary = {
        "recipes": len(sets),
        "vocabulary": len(vocabulary),
        "plural_folds": len(normalizer.plural_map),
        "aliases": len(normalizer.aliases),
        "removed_high_frequency": normalizer.stop_ingredients,
        "empty_sets": int(sum(1 for s in sets if not s)),
        "mean_set_size": round(float(np.mean([len(s) for s in sets])), 3),
        "num_perm": hasher.num_perm,
        "seed": hasher.seed,
        "normalize_seconds": round(normalize_seconds, 2),
        "signature_seconds": round(signature_seconds, 2),
    }
    (REPO_ROOT / "reports" / "tables" / "minhash_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
