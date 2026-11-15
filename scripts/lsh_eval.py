"""LSH 를 정확한 Jaccard 와 비교한다: (b, r) 별 재현율, 후보 수, 조회 시간, S-curve.

추천 대상 레시피 전체를 인덱스에 넣고, 무작위 질의 레시피에 대해 전수 비교로 구한 정확한 Jaccard 를 기준값으로 쓴다.
표는 reports/tables, 그림은 reports/figures 에 쓴다.

사용법:
    python3 scripts/lsh_eval.py
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
from dmrec.jaccard import jaccard_to_all  # noqa: E402
from dmrec.ingredients import to_csr  # noqa: E402
from dmrec.lsh import LSHIndex, approximate_threshold, collision_probability  # noqa: E402
from dmrec.minhash import MinHasher, estimate_jaccard  # noqa: E402

BLUE, AQUA, YELLOW, INK, MUTED, GRID = "#2a78d6", "#1baa9c", "#eaa300", "#0b0b0b", "#52514e", "#e4e3df"
# 시그니처 128 개 안에서 만들 수 있는 조합. r 이 커질수록 문턱이 높아진다.
CONFIGS = [(64, 2), (42, 3), (32, 4), (25, 5), (21, 6), (16, 8)]
N_QUERIES, CHUNK, TOP_K = 2000, 500, 10
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
BIN_EDGES = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_TIMING = 500
SCALE_CONFIGS = [(42, 3), (32, 4), (25, 5)]
N_SCALE_QUERIES = 300
N_SEEDS, N_SEED_QUERIES, SEED_RANGE = 12, 600, (0.2, 0.4)


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


def top_k_exact(sim: np.ndarray, k: int) -> np.ndarray:
    """각 행에서 값이 큰 k 개의 열 번호. 동률은 열 번호가 작은 쪽이 먼저다."""
    kth = np.partition(sim, sim.shape[1] - k, axis=1)[:, sim.shape[1] - k]
    out = np.empty((sim.shape[0], k), dtype=np.int64)
    for r in range(sim.shape[0]):
        cand = np.flatnonzero(sim[r] >= kth[r])
        out[r] = cand[np.lexsort((cand, -sim[r, cand]))][:k]
    return out


def exact_query(index: LSHIndex, row: int, matrix, sizes: np.ndarray, k: int) -> np.ndarray:
    """LSH 후보를 정확한 Jaccard 로 다시 정렬해 상위 k 개의 행 번호를 돌려준다."""
    found = index.candidates(row)
    if len(found) == 0:
        return found
    inter = np.asarray((matrix[found] @ matrix[row].T).todense()).ravel().astype(np.float64)
    scores = inter / (sizes[found] + sizes[row] - inter)
    return found[np.lexsort((found, -scores))][:k]


def scale_experiment(data, all_sets, seed: int) -> pd.DataFrame:
    """레시피 수를 늘렸을 때의 조회 시간과 재현율: 추천 대상(37,818) 대 전체(231,352)."""
    rows_out = []
    core = np.flatnonzero(data["recipe_idx"] >= 0)
    for label, keep in (("core", core), ("all", np.arange(len(data["recipe_id"])))):
        signatures = data["signatures"][keep]
        matrix, _ = to_csr([json.loads(s) for s in all_sets[keep]])
        sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
        n = matrix.shape[0]
        queries = np.sort(np.random.default_rng(seed).choice(n, N_SCALE_QUERIES, replace=False))

        started = time.perf_counter()
        truth = []
        for row in queries:
            sim = jaccard_to_all(matrix, np.array([row]))
            sim[0, row] = -1.0
            truth.append(top_k_exact(sim, TOP_K)[0])
        exact_ms = (time.perf_counter() - started) / len(queries) * 1000

        for bands, rows in SCALE_CONFIGS:
            started = time.perf_counter()
            index = LSHIndex(bands, rows).build(signatures)
            build = time.perf_counter() - started
            started = time.perf_counter()
            results = [exact_query(index, row, matrix, sizes, TOP_K) for row in queries]
            query_ms = (time.perf_counter() - started) / len(queries) * 1000
            recall = np.mean([len(np.intersect1d(t, r)) / TOP_K for t, r in zip(truth, results)])
            counts = [len(index.candidates(row)) for row in queries]
            rows_out.append({
                "recipes": label, "n": n, "bands": bands, "rows": rows,
                "candidates_median": float(np.median(counts)), "candidates_mean": round(float(np.mean(counts)), 1),
                f"top{TOP_K}_recall": round(float(recall), 4),
                "build_seconds": round(build, 2), "query_ms": round(query_ms, 3),
                "exact_brute_force_ms": round(exact_ms, 3), "speedup_vs_exact": round(exact_ms / query_ms, 1),
            })
    return pd.DataFrame(rows_out)


def seed_experiment(matrix, vocabulary, bands: int, rows: int, seed: int) -> pd.DataFrame:
    """해시 함수를 바꿔 가며 문턱 근처 쌍이 후보가 되는 비율을 잰다.

    모든 쌍이 같은 해시 함수를 쓰고 흔한 재료를 공유하므로, 한 번의 실행에서 관측한 비율은 이론값에서 벗어날 수 있다.
    """
    n = matrix.shape[0]
    queries = np.sort(np.random.default_rng(seed + 1).choice(n, N_SEED_QUERIES, replace=False))
    sim = jaccard_to_all(matrix, queries)
    sim[np.arange(len(queries)), queries] = -1.0
    selected = (sim >= SEED_RANGE[0]) & (sim < SEED_RANGE[1])
    theory = float(collision_probability(sim[selected], bands, rows).mean())
    out = []
    for s in range(N_SEEDS):
        signatures = MinHasher(bands * rows, s).signatures(matrix, vocabulary)
        index = LSHIndex(bands, rows).build(signatures)
        mask = np.zeros(sim.shape, dtype=bool)
        for i, row in enumerate(queries):
            mask[i, index.candidates(row)] = True
        out.append({"seed": s, "bands": bands, "rows": rows, "pairs": int(selected.sum()),
                    "observed": round(float(mask[selected].mean()), 5), "theory": round(theory, 5)})
    return pd.DataFrame(out)


def main() -> None:
    config = load_config()
    seed = config["seed"]
    processed = resolve_path(config, "processed_dir")
    tables = REPO_ROOT / "reports" / "tables"
    figures = resolve_path(config, "figures_dir")

    data = np.load(processed / "minhash_signatures.npz")
    keep = np.flatnonzero(data["recipe_idx"] >= 0)
    keep = keep[np.argsort(data["recipe_idx"][keep])]
    signatures = data["signatures"][keep]
    all_sets = pd.read_csv(processed / "ingredient_sets.csv", usecols=["ingredients_norm"])["ingredients_norm"].to_numpy()
    matrix, vocabulary = to_csr([json.loads(s) for s in all_sets[keep]])
    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    n = matrix.shape[0]

    rng = np.random.default_rng(seed)
    queries = np.sort(rng.choice(n, N_QUERIES, replace=False))
    timing_queries = queries[:N_TIMING]

    indexes = {}
    build_seconds = {}
    for bands, rows in CONFIGS:
        started = time.perf_counter()
        indexes[(bands, rows)] = LSHIndex(bands, rows).build(signatures)
        build_seconds[(bands, rows)] = time.perf_counter() - started

    n_bins = len(BIN_EDGES) - 1
    pairs_in_bin = np.zeros(n_bins, dtype=np.int64)
    candidates_in_bin = {c: np.zeros(n_bins, dtype=np.int64) for c in CONFIGS}
    expected_in_bin = {c: np.zeros(n_bins, dtype=np.float64) for c in CONFIGS}
    pairs_at = {t: 0 for t in THRESHOLDS}
    found_at = {c: {t: 0 for t in THRESHOLDS} for c in CONFIGS}
    stats = {c: {"candidates": [], "in_candidates": 0, "in_ranked": 0, "no_candidates": 0} for c in CONFIGS}
    exact_top_sum = 0.0

    for start in range(0, N_QUERIES, CHUNK):
        part = queries[start:start + CHUNK]
        sim = jaccard_to_all(matrix, part)
        sim[np.arange(len(part)), part] = -1.0                # 자기 자신 제외
        exact_top = top_k_exact(sim, TOP_K)
        exact_top_sum += np.take_along_axis(sim, exact_top, axis=1).sum()
        which = np.clip(np.digitize(sim, BIN_EDGES) - 1, 0, n_bins - 1)
        valid = sim >= 0
        pairs_in_bin += np.bincount(which[valid], minlength=n_bins)
        for t in THRESHOLDS:
            pairs_at[t] += int((sim >= t).sum())

        for c in CONFIGS:
            index = indexes[c]
            mask = np.zeros(sim.shape, dtype=bool)
            for i, row in enumerate(part):
                cand = index.candidates(row)
                mask[i, cand] = True
                stats[c]["candidates"].append(len(cand))
                stats[c]["no_candidates"] += len(cand) == 0
                # 정확한 상위 k 개가 후보 안에 있는가, LSH 가 돌려준 상위 k 개 안에 있는가
                truth = exact_top[i]
                stats[c]["in_candidates"] += int(mask[i, truth].sum())
                ranked, _ = index.query(row, top_n=TOP_K)
                stats[c]["in_ranked"] += len(np.intersect1d(truth, ranked))
            candidates_in_bin[c] += np.bincount(which[mask & valid], minlength=n_bins)
            # 이론값: 구간 안의 각 쌍의 정확한 Jaccard 에 S-curve 를 적용해 더한다
            expected_in_bin[c] += np.bincount(which[valid], weights=collision_probability(sim[valid], *c), minlength=n_bins)
            for t in THRESHOLDS:
                found_at[c][t] += int((mask & (sim >= t)).sum())

    # 조회 시간: 후보 조회 + 추정 Jaccard 로 상위 k 개 고르기
    query_ms = {}
    for c in CONFIGS:
        index = indexes[c]
        started = time.perf_counter()
        for row in timing_queries:
            index.query(row, top_n=TOP_K)
        query_ms[c] = (time.perf_counter() - started) / len(timing_queries) * 1000

    # 조회 시간 (정확한 Jaccard 로 다시 정렬하는 경우)
    rerank_ms = {}
    for c in CONFIGS:
        started = time.perf_counter()
        for row in timing_queries:
            exact_query(indexes[c], row, matrix, sizes, TOP_K)
        rerank_ms[c] = (time.perf_counter() - started) / len(timing_queries) * 1000

    # 비교 기준 1: 정확한 Jaccard 를 모든 레시피와 계산
    started = time.perf_counter()
    for row in timing_queries:
        sim = jaccard_to_all(matrix, np.array([row]))
        sim[0, row] = -1.0
        top_k_exact(sim, TOP_K)
    exact_ms = (time.perf_counter() - started) / len(timing_queries) * 1000
    # 비교 기준 2: MinHash 시그니처를 모든 레시피와 비교 (LSH 없이)
    started = time.perf_counter()
    for row in timing_queries:
        est = estimate_jaccard(signatures, signatures[row][None, :]).astype(np.float32)
        est[row] = -1.0
        top_k_exact(est[None, :], TOP_K)
    scan_ms = (time.perf_counter() - started) / len(timing_queries) * 1000

    total_pairs = N_QUERIES * (n - 1)
    rows_out = []
    for c in CONFIGS:
        bands, rows = c
        counts = np.array(stats[c]["candidates"])
        row = {
            "bands": bands, "rows": rows, "signature_values_used": bands * rows,
            "threshold": round(approximate_threshold(bands, rows), 3),
            "candidates_median": float(np.median(counts)),
            "candidates_mean": round(float(counts.mean()), 1),
            "pairs_compared_pct": round(float(counts.sum() / total_pairs * 100), 3),
            "queries_without_candidates_pct": round(stats[c]["no_candidates"] / N_QUERIES * 100, 2),
            # 후보를 정확한 Jaccard 로 다시 정렬하면 정확한 상위 k 개 중 후보에 든 것을 모두 찾는다
            f"top{TOP_K}_recall_exact_rerank": round(stats[c]["in_candidates"] / (N_QUERIES * TOP_K), 4),
            f"top{TOP_K}_recall_minhash_rank": round(stats[c]["in_ranked"] / (N_QUERIES * TOP_K), 4),
            "build_seconds": round(build_seconds[c], 2),
            "query_ms_exact_rerank": round(rerank_ms[c], 3),
            "query_ms_minhash_rank": round(query_ms[c], 3),
            "speedup_vs_exact": round(exact_ms / rerank_ms[c], 1),
        }
        for t in THRESHOLDS:
            row[f"recall_j>={t}"] = round(found_at[c][t] / pairs_at[t], 4) if pairs_at[t] else None
        rows_out.append(row)
    table = pd.DataFrame(rows_out)
    table.to_csv(tables / "lsh_configs.csv", index=False)

    curve_rows = []
    for c in CONFIGS:
        for i in range(n_bins):
            if pairs_in_bin[i] == 0:
                continue
            curve_rows.append({
                "bands": c[0], "rows": c[1], "jaccard_from": BIN_EDGES[i], "jaccard_to": BIN_EDGES[i + 1], "pairs": int(pairs_in_bin[i]),
                "observed": round(float(candidates_in_bin[c][i] / pairs_in_bin[i]), 5),
                "theory": round(float(expected_in_bin[c][i] / pairs_in_bin[i]), 5),
            })
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(tables / "lsh_s_curve.csv", index=False)

    summary = {
        "recipes_indexed": int(n), "queries": N_QUERIES, "top_k": TOP_K,
        "pairs_with_jaccard_at_least": {str(t): pairs_at[t] for t in THRESHOLDS},
        "exact_top_k_mean_jaccard": round(float(exact_top_sum / (N_QUERIES * TOP_K)), 4),
        "exact_brute_force_ms_per_query": round(exact_ms, 3),
        "minhash_scan_ms_per_query": round(scan_ms, 3),
        "timing_queries": int(len(timing_queries)),
    }

    # 그림 1: S-curve (이론: 선, 관측: 점)
    shown = [(64, 2), (42, 3), (16, 8)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    grid = np.linspace(0, 1, 201)
    for c, color in zip(shown, (AQUA, BLUE, YELLOW)):
        part = curve[(curve["bands"] == c[0]) & (curve["rows"] == c[1]) & (curve["pairs"] >= 30)]
        ax.plot(grid, collision_probability(grid, *c), color=color, linewidth=2, label=f"b={c[0]}, r={c[1]}")
        ax.scatter((part["jaccard_from"] + part["jaccard_to"]) / 2, part["observed"], s=22, color=color, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlabel("Exact Jaccard")
    ax.set_ylabel("Probability of becoming a candidate")
    ax.set_title("S-curve: theory (lines) vs observed (dots)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(figures / "lsh_s_curve.png", dpi=200)
    plt.close(fig)

    # 그림 2: 비교한 쌍의 비율 대 상위 k 재현율
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(table["pairs_compared_pct"], table[f"top{TOP_K}_recall_exact_rerank"], color=BLUE, linewidth=2, marker="o", markersize=6, label="candidates re-ranked by exact Jaccard")
    ax.plot(table["pairs_compared_pct"], table[f"top{TOP_K}_recall_minhash_rank"], color=AQUA, linewidth=2, marker="s", markersize=5, label="ranked by MinHash estimate")
    for _, r in table.iterrows():
        ax.annotate(f"b={int(r['bands'])}, r={int(r['rows'])}", (r["pairs_compared_pct"], r[f"top{TOP_K}_recall_exact_rerank"]),
                    textcoords="offset points", xytext=(-4, 7), fontsize=7, color=MUTED, ha="right")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel("Pairs compared (% of all pairs, log scale)")
    ax.set_ylabel(f"Recall of exact top-{TOP_K} neighbors")
    ax.set_ylim(0, 1.02)
    ax.set_title("Cost vs recall by (b, r)", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(figures / "lsh_cost_vs_recall.png", dpi=200)
    plt.close(fig)

    by_seed = seed_experiment(matrix, vocabulary, config["lsh"]["bands"], config["lsh"]["rows"], seed)
    by_seed.to_csv(tables / "lsh_s_curve_by_seed.csv", index=False)
    summary["seed_check"] = {
        "bands": config["lsh"]["bands"], "rows": config["lsh"]["rows"], "jaccard_range": list(SEED_RANGE), "seeds": N_SEEDS,
        "theory": float(by_seed["theory"].iloc[0]),
        "observed_mean": round(float(by_seed["observed"].mean()), 5),
        "observed_sd_across_seeds": round(float(by_seed["observed"].std(ddof=1)), 5),
        "observed_se_of_mean": round(float(by_seed["observed"].std(ddof=1) / np.sqrt(N_SEEDS)), 5),
        "observed_min": float(by_seed["observed"].min()), "observed_max": float(by_seed["observed"].max()),
    }
    scale = scale_experiment(data, all_sets, seed)
    (tables / "lsh_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    scale.to_csv(tables / "lsh_scale.csv", index=False)

    print(table.T.to_string())
    print(scale.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
