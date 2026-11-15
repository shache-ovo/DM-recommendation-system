"""MinHash 추정값을 정확한 Jaccard 와 비교한다. 표는 reports/tables, 그림은 reports/figures 에 쓴다.

추천 대상 레시피에서 두 종류의 쌍을 뽑는다.
  - 무작위 쌍: 대부분 Jaccard 가 0 근처다
  - 이웃 쌍: 무작위 레시피의 정확한 상위 20개 이웃. Jaccard 가 높은 구간을 채운다
시그니처 길이 k 를 바꿔 가며 추정 오차를 재고, 이론값 sqrt(J(1-J)/k) 와 비교한다.

사용법:
    python3 scripts/minhash_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.ingredients import normalizer_from_config, to_csr  # noqa: E402
from dmrec.jaccard import pairwise_jaccard, top_k_neighbors  # noqa: E402
from dmrec.minhash import MinHasher, estimate_jaccard  # noqa: E402

BLUE, AQUA, INK, MUTED, GRID = "#2a78d6", "#1baa9c", "#0b0b0b", "#52514e", "#e4e3df"
K_VALUES = [16, 32, 64, 128, 256]
N_RANDOM, N_QUERIES, TOP = 100_000, 2_000, 20
N_SEEDS = 20
BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def main() -> None:
    config = load_config()
    seed = config["seed"]
    processed = resolve_path(config, "processed_dir")
    tables = REPO_ROOT / "reports" / "tables"
    figures = resolve_path(config, "figures_dir")

    recipes = pd.read_csv(processed / "recipes.csv", usecols=["recipe_id", "recipe_idx", "ingredients"])
    raw = recipes["ingredients"].map(json.loads).tolist()
    normalizer = normalizer_from_config(config).fit(raw)
    core_rows = recipes[recipes["recipe_idx"] >= 0].sort_values("recipe_idx").index.to_numpy()
    sets = [normalizer.transform(raw[i]) for i in core_rows]
    matrix, vocabulary = to_csr(sets)
    n = len(sets)

    # 시그니처는 가장 긴 길이로 한 번 만들고, 앞의 k 개만 써서 길이 k 의 MinHash 로 본다
    hasher = MinHasher(max(K_VALUES), seed)
    signatures = hasher.signatures(matrix, vocabulary)

    rng = np.random.default_rng(seed)
    left, right = rng.integers(0, n, N_RANDOM), rng.integers(0, n, N_RANDOM)
    keep = left != right
    random_pairs = (left[keep], right[keep])
    queries = rng.choice(n, N_QUERIES, replace=False)
    neighbors, _ = top_k_neighbors(matrix, TOP, rows=queries)
    neighbor_pairs = (np.repeat(queries, TOP), neighbors.ravel())

    rows, bin_rows = [], []
    scatter = None
    for pair_name, (a, b) in (("random", random_pairs), ("neighbor", neighbor_pairs)):
        exact = pairwise_jaccard(matrix, a, b)
        for k in K_VALUES:
            estimate = estimate_jaccard(signatures[a, :k], signatures[b, :k])
            error = estimate - exact
            rows.append({
                "pairs": pair_name, "num_perm": k, "n_pairs": len(exact),
                "exact_mean": round(float(exact.mean()), 4),
                "bias": round(float(error.mean()), 5),
                "mae": round(float(np.abs(error).mean()), 5),
                "rmse": round(float(np.sqrt((error**2).mean())), 5),
                "theory_rmse": round(float(np.sqrt((exact * (1 - exact) / k).mean())), 5),
                "max_abs_error": round(float(np.abs(error).max()), 4),
            })
            if pair_name == "neighbor":
                which = np.digitize(exact, BINS) - 1
                for i in range(len(BINS) - 1):
                    mask = which == i
                    if mask.sum() == 0:
                        continue
                    bin_rows.append({
                        "num_perm": k, "exact_from": BINS[i], "exact_to": min(BINS[i + 1], 1.0), "n_pairs": int(mask.sum()),
                        "bias": round(float(error[mask].mean()), 5),
                        "rmse": round(float(np.sqrt((error[mask]**2).mean())), 5),
                        "theory_rmse": round(float(np.sqrt((exact[mask] * (1 - exact[mask]) / k).mean())), 5),
                    })
                if k == config["minhash"]["num_perm"]:
                    scatter = (exact, estimate)

    table = pd.DataFrame(rows)
    table.to_csv(tables / "minhash_error.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(tables / "minhash_error_by_jaccard.csv", index=False)

    # 해시 함수를 바꿔 가며 평균 오차를 본다. 모든 쌍이 같은 해시 함수를 쓰고 흔한 재료를 공유하므로
    # 한 번의 실행 안에서는 쌍들의 오차가 서로 상관된다. 추정이 치우쳤는지는 seed 를 바꿔 봐야 알 수 있다.
    a, b = neighbor_pairs
    exact = pairwise_jaccard(matrix, a, b)
    k = config["minhash"]["num_perm"]
    seed_rows = []
    for s in range(N_SEEDS):
        sig = MinHasher(k, s).signatures(matrix, vocabulary)
        error = estimate_jaccard(sig[a], sig[b]) - exact
        seed_rows.append({"seed": s, "num_perm": k, "bias": round(float(error.mean()), 5), "rmse": round(float(np.sqrt((error**2).mean())), 5)})
    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(tables / "minhash_bias_by_seed.csv", index=False)

    # 시그니처 생성 시간: 추천 대상 레시피와 전체 레시피
    timing = []
    all_sets = [normalizer.transform(names) for names in raw]
    all_matrix, all_vocabulary = to_csr(all_sets)
    for label, m, v in (("core", matrix, vocabulary), ("all", all_matrix, all_vocabulary)):
        for k in (64, 128, 256):
            started = time.perf_counter()
            MinHasher(k, seed).signatures(m, v)
            timing.append({"recipes": label, "n": m.shape[0], "nnz": int(m.nnz), "num_perm": k, "seconds": round(time.perf_counter() - started, 3)})
    pd.DataFrame(timing).to_csv(tables / "minhash_timing.csv", index=False)

    # 그림 1: 시그니처 길이에 따른 RMSE (관측 vs 이론)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for pair_name, color in (("neighbor", BLUE), ("random", AQUA)):
        part = table[table["pairs"] == pair_name]
        ax.plot(part["num_perm"], part["rmse"], color=color, linewidth=2, marker="o", markersize=5, label=f"{pair_name} pairs, observed")
        ax.plot(part["num_perm"], part["theory_rmse"], color=color, linewidth=1.2, linestyle="--", label=f"{pair_name} pairs, theory")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Signature length k")
    ax.set_ylabel("RMSE of estimated Jaccard")
    ax.set_title("MinHash estimation error", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(figures / "minhash_error_vs_k.png", dpi=200)
    plt.close(fig)

    # 그림 2: 정확값 대 추정값 (설정된 시그니처 길이)
    exact, estimate = scatter
    k = config["minhash"]["num_perm"]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.scatter(exact, estimate, s=3, color=BLUE, alpha=0.12, linewidths=0)
    grid = np.linspace(0, 1, 101)
    band = 2 * np.sqrt(grid * (1 - grid) / k)
    ax.plot(grid, grid, color=INK, linewidth=1)
    ax.plot(grid, np.clip(grid + band, 0, 1), color=MUTED, linewidth=0.8, linestyle="--")
    ax.plot(grid, np.clip(grid - band, 0, 1), color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Exact Jaccard")
    ax.set_ylabel(f"MinHash estimate (k = {k})")
    ax.set_title("Estimate vs exact (dashed: ±2 SE)", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(figures / "minhash_estimate_vs_exact.png", dpi=200)
    plt.close(fig)

    inside = np.abs(estimate - exact) <= 2 * np.sqrt(exact * (1 - exact) / k) + 1e-12
    print(table.to_string(index=False))
    print(pd.DataFrame(bin_rows).query("num_perm == @k").to_string(index=False))
    print(pd.DataFrame(timing).to_string(index=False))
    print(f"neighbor pairs within ±2 SE at k={k}: {inside.mean():.4f}")
    bias = seed_table["bias"]
    summary = {
        "num_perm": k,
        "neighbor_pairs_within_2se": round(float(inside.mean()), 4),
        "seeds": N_SEEDS,
        "bias_mean_over_seeds": round(float(bias.mean()), 5),
        "bias_sd_across_seeds": round(float(bias.std(ddof=1)), 5),
        "bias_se_of_mean": round(float(bias.std(ddof=1) / np.sqrt(N_SEEDS)), 5),
        "rmse_mean_over_seeds": round(float(seed_table["rmse"].mean()), 5),
    }
    print(json.dumps(summary, indent=2))
    (tables / "minhash_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
