"""원본 데이터의 탐색적 분석. 표는 reports/tables, 그림은 reports/figures 에 쓴다.

사용법:
    python3 scripts/eda.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmrec.config import load_config, resolve_path  # noqa: E402
from dmrec.filtering import k_core  # noqa: E402

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
K_VALUES = [1, 2, 3, 5, 10, 20]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def describe(series: pd.Series) -> dict:
    out = {"min": float(series.min()), "mean": float(series.mean()), "max": float(series.max())}
    for q in QUANTILES:
        out[f"p{q * 100:g}"] = float(series.quantile(q))
    return out


def main() -> None:
    config = load_config()
    tables = REPO_ROOT / "reports" / "tables"
    figures = resolve_path(config, "figures_dir")
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    positive_min = config["preprocess"]["positive_min_rating"]
    nutrition_fields = config["preprocess"]["nutrition_fields"]
    summary: dict = {}

    recipes = pd.read_csv(resolve_path(config, "recipes_csv"))
    inter = pd.read_csv(resolve_path(config, "interactions_csv"), parse_dates=["date"])

    # ---------- 결측 ----------
    summary["missing_recipes"] = {c: int(n) for c, n in recipes.isna().sum().items() if n}
    summary["missing_interactions"] = {c: int(n) for c, n in inter.isna().sum().items() if n}

    # ---------- 평점 ----------
    rating = inter["rating"].value_counts().sort_index()
    rating_table = pd.DataFrame({"rating": rating.index, "count": rating.values, "percent": (rating.values / len(inter) * 100).round(2)})
    rating_table.to_csv(tables / "rating_distribution.csv", index=False)
    summary["n_interactions"] = len(inter)
    summary["n_positive"] = int((inter["rating"] >= positive_min).sum())
    summary["positive_min_rating"] = positive_min

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    bars = ax.bar(rating_table["rating"].astype(str), rating_table["count"], color=BLUE, width=0.6)
    for bar, pct in zip(bars, rating_table["percent"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_xlabel("Rating (0 = review without a rating)")
    ax.set_ylabel("Interactions")
    ax.set_title("Rating distribution", fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    style_axes(ax)
    save(fig, figures / "rating_distribution.png")

    # ---------- 중복, 날짜 ----------
    summary["duplicate_user_recipe_pairs"] = int(inter.duplicated(["user_id", "recipe_id"]).sum())
    summary["date_min"] = str(inter["date"].min().date())
    summary["date_max"] = str(inter["date"].max().date())
    by_year = inter["date"].dt.year.value_counts().sort_index()
    pd.DataFrame({"year": by_year.index, "interactions": by_year.values}).to_csv(tables / "interactions_by_year.csv", index=False)

    # ---------- 사용자/레시피별 상호작용 수 ----------
    per_user = inter["user_id"].value_counts()
    per_recipe = inter["recipe_id"].value_counts()
    summary["per_user"] = describe(per_user) | {"n": len(per_user), "share_with_1": float((per_user == 1).mean()), "share_lt_5": float((per_user < 5).mean())}
    summary["per_recipe"] = describe(per_recipe) | {"n": len(per_recipe), "share_with_1": float((per_recipe == 1).mean()), "share_lt_5": float((per_recipe < 5).mean())}
    # 상위 사용자에게 상호작용이 얼마나 몰려 있는가
    sorted_user = np.sort(per_user.values)[::-1]
    summary["top_1pct_users_share_of_interactions"] = float(sorted_user[: max(1, len(sorted_user) // 100)].sum() / len(inter))

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
    for ax, counts, label in ((axes[0], per_user, "user"), (axes[1], per_recipe, "recipe")):
        freq = counts.value_counts().sort_index()
        ax.scatter(freq.index, freq.values, s=8, color=BLUE, linewidths=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Interactions per {label}")
        ax.set_title(f"Per {label}", fontsize=10, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("Number of users / recipes")
    save(fig, figures / "interaction_count_distribution.png")

    # ---------- k-core ----------
    rows = []
    for name, frame in (("all", inter), (f"rating>={positive_min}", inter[inter["rating"] >= positive_min])):
        pairs = frame[["user_id", "recipe_id"]]
        for k in K_VALUES:
            core = k_core(pairs, k)
            n_u, n_r = core["user_id"].nunique(), core["recipe_id"].nunique()
            rows.append({
                "interactions_used": name, "k": k, "interactions": len(core), "users": n_u, "recipes": n_r,
                "interactions_kept_pct": round(len(core) / len(inter) * 100, 2),
                "density_pct": round(len(core) / (n_u * n_r) * 100, 4) if n_u and n_r else 0.0,
            })
    k_table = pd.DataFrame(rows)
    k_table.to_csv(tables / "k_core.csv", index=False)

    # 분할에서 같은 날짜 처리: 사용자의 마지막 날짜에 양성이 여러 개인 사용자 비율 (5-core, 양성 기준)
    core5 = k_core(inter.loc[inter["rating"] >= positive_min, ["user_id", "recipe_id", "date"]], 5)
    last_date = core5.groupby("user_id")["date"].transform("max")
    n_on_last = core5[core5["date"] == last_date].groupby("user_id").size()
    summary["core5_users_with_tie_on_last_date_share"] = float((n_on_last > 1).mean())

    # ---------- 영양 ----------
    nutrition = pd.DataFrame(recipes["nutrition"].map(json.loads).tolist(), columns=nutrition_fields)
    nut_table = nutrition.describe(percentiles=QUANTILES).T.drop(columns=["count", "std"]).round(1)
    nut_table["zero_share_pct"] = ((nutrition == 0).mean() * 100).round(2)
    nut_table.to_csv(tables / "nutrition_summary.csv", index_label="field")
    calories = nutrition["calories"]
    thresholds = [1000, 2000, 3000, 5000, 10000]
    cal_rows = [{"rule": f"calories > {t}", "recipes": int((calories > t).sum()), "percent": round(float((calories > t).mean() * 100), 3)} for t in thresholds]
    for q in (0.99, 0.995, 0.999):
        cut = float(calories.quantile(q))
        cal_rows.append({"rule": f"calories > p{q * 100:g} ({cut:.1f})", "recipes": int((calories > cut).sum()), "percent": round(float((calories > cut).mean() * 100), 3)})
    cal_rows.append({"rule": "calories == 0", "recipes": int((calories == 0).sum()), "percent": round(float((calories == 0).mean() * 100), 3)})
    pd.DataFrame(cal_rows).to_csv(tables / "calorie_outlier_candidates.csv", index=False)
    summary["nutrition_all_zero_recipes"] = int((nutrition == 0).all(axis=1).sum())
    corr = nutrition.corr(method="spearman").round(2)
    corr.to_csv(tables / "nutrition_spearman_corr.csv")

    fig, axes = plt.subplots(2, 4, figsize=(9, 4))
    for ax, field in zip(axes.flat, nutrition_fields):
        values = nutrition[field]
        ax.hist(values.clip(upper=values.quantile(0.99)), bins=50, color=BLUE)
        unit = "" if field == "calories" else " (% daily value)"
        ax.set_title(f"{field.replace('_', ' ')}{unit}", fontsize=8, loc="left")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        style_axes(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, "Values above the 99th percentile\nare clipped into the last bin.\ny-axis: number of recipes", fontsize=8, color=MUTED, va="center")
    save(fig, figures / "nutrition_distributions.png")

    # ---------- 조리 시간, 단계 수 ----------
    summary["minutes"] = describe(recipes["minutes"]) | {"zero": int((recipes["minutes"] == 0).sum()), "over_1_day": int((recipes["minutes"] > 1440).sum())}
    summary["n_steps"] = describe(recipes["n_steps"]) | {"zero": int((recipes["n_steps"] == 0).sum())}

    # ---------- 재료 ----------
    ingredients = recipes["ingredients"].map(ast.literal_eval)
    set_sizes = ingredients.map(lambda xs: len(set(xs)))
    summary["ingredients"] = {
        "set_size": describe(set_sizes),
        "n_ingredients_column_mismatch": int((ingredients.map(len) != recipes["n_ingredients"]).sum()),
        "recipes_with_repeated_ingredient": int((ingredients.map(len) != set_sizes).sum()),
        "recipes_with_lt_3_ingredients": int((set_sizes < 3).sum()),
        "recipes_with_1_ingredient": int((set_sizes == 1).sum()),
    }
    doc_freq = Counter(x for xs in ingredients for x in set(xs))
    freq = pd.Series(doc_freq).sort_values(ascending=False)
    summary["ingredients"]["vocabulary"] = len(freq)
    summary["ingredients"]["in_1_recipe"] = int((freq == 1).sum())
    summary["ingredients"]["in_lt_5_recipes"] = int((freq < 5).sum())
    summary["ingredients"]["in_ge_1pct_recipes"] = int((freq >= len(recipes) * 0.01).sum())
    summary["ingredients"]["in_ge_10pct_recipes"] = int((freq >= len(recipes) * 0.10).sum())
    top = pd.DataFrame({"ingredient": freq.index[:30], "recipes": freq.values[:30], "percent": (freq.values[:30] / len(recipes) * 100).round(2)})
    top.to_csv(tables / "top_ingredients.csv", index=False)
    # 표기만 다른 재료의 예: 공백/복수형으로 묶이는 이름
    names = set(freq.index)
    summary["ingredients"]["plural_pairs"] = int(sum(1 for n in names if n.endswith("s") and n[:-1] in names))
    # 재료 집합이 완전히 같은 레시피 (LSH 에서 Jaccard = 1)
    frozen = ingredients.map(lambda xs: tuple(sorted(set(xs))))
    dup_sizes = frozen.value_counts()
    summary["ingredients"]["recipes_sharing_identical_set"] = int(dup_sizes[dup_sizes > 1].sum())
    summary["ingredients"]["identical_set_groups"] = int((dup_sizes > 1).sum())

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    top20 = top.head(20).iloc[::-1]
    ax.barh(top20["ingredient"], top20["percent"], color=BLUE, height=0.6)
    ax.set_xlabel("Share of recipes containing the ingredient (%)")
    ax.set_title("Top 20 ingredients", fontsize=10, loc="left")
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(axis="y", colors=INK)
    save(fig, figures / "top_ingredients.png")

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    size_counts = set_sizes.clip(upper=30).value_counts().sort_index()
    ax.bar(size_counts.index, size_counts.values, color=BLUE, width=0.7)
    ax.set_xlabel("Distinct ingredients per recipe (30 = 30 or more)")
    ax.set_ylabel("Recipes")
    ax.set_title("Ingredient set size", fontsize=10, loc="left")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    style_axes(ax)
    save(fig, figures / "ingredient_set_size.png")

    # ---------- 태그 ----------
    tags = recipes["tags"].map(ast.literal_eval)
    tag_freq = pd.Series(Counter(t for ts in tags for t in set(ts))).sort_values(ascending=False)
    summary["tags"] = {"vocabulary": len(tag_freq), "per_recipe": describe(tags.map(len)), "recipes_without_tags": int((tags.map(len) == 0).sum())}
    pd.DataFrame({"tag": tag_freq.index[:30], "recipes": tag_freq.values[:30], "percent": (tag_freq.values[:30] / len(recipes) * 100).round(2)}).to_csv(tables / "top_tags.csv", index=False)

    (tables / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for name in ("rating_distribution", "k_core", "nutrition_summary", "calorie_outlier_candidates", "top_ingredients"):
        print(f"\n== {name}")
        print(pd.read_csv(tables / f"{name}.csv").to_string(index=False))


if __name__ == "__main__":
    main()
