"""Static charts for the evaluation report (PNG, embedded in README / final report).

* model_comparison.png - small multiples, one panel per metric, one bar per model. Model identity is
  carried by the axis labels, so a single hue is used and every bar is directly labelled.
* ndcg_by_segment.png  - NDCG@K per user-activity segment; models use fixed categorical slots,
  with a legend and value labels (the palette's lighter slots are below 3:1 contrast).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_ORDER = ["popularity", "item_knn", "content", "mf", "hybrid"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _ordered(results: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return results.assign(_o=results["model"].map(order).fillna(99)).sort_values("_o").drop(columns="_o")


def plot_results(results: pd.DataFrame, segments: pd.DataFrame, k: int, out_dir: Path) -> None:
    res = _ordered(results)
    metrics = [(f"precision_at_{k}", f"Precision@{k}"), (f"recall_at_{k}", f"Recall@{k}"),
               (f"ndcg_at_{k}", f"NDCG@{k}"), (f"coverage_at_{k}", f"Coverage@{k}"),
               (f"diversity_at_{k}", f"Diversity@{k} (ILD)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.4), sharey=True, facecolor=SURFACE)
    y = np.arange(len(res))[::-1]
    for ax, (col, title) in zip(axes, metrics):
        vals = res[col].values
        ax.barh(y, vals, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        vmax = max(vals.max(), 1e-9)
        fmt = ".4f" if vmax < 0.1 else ".3f"
        for yi, v in zip(y, vals):
            ax.text(v + vmax * 0.03, yi, format(v, fmt), va="center", fontsize=8.5, color=TEXT)
        ax.set_xlim(0, vmax * 1.35)
        ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
        _style(ax)
        ax.set_xticks([])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(res["model"], fontsize=10, color=TEXT)
    fig.suptitle("Model comparison on the time-based test window", x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    if segments is None or segments.empty:
        return
    col = f"ndcg_at_{k}"
    segs = list(dict.fromkeys(segments["segment"]))
    models = [m for m in MODEL_ORDER if m in set(segments["model"])]
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    width = 0.8 / len(models)
    base = np.arange(len(segs))[::-1]
    for j, m in enumerate(models):
        sub = segments[segments["model"] == m].set_index("segment").reindex(segs)
        pos = base + 0.4 - (j + 0.5) * width
        ax.barh(pos, sub[col].values, height=width, color=SERIES[j], edgecolor=SURFACE, linewidth=2, label=m)
        for p, v in zip(pos, sub[col].values):
            if pd.notna(v):
                ax.text(v, p, f" {v:.4f}", va="center", fontsize=7.5, color=TEXT_2)
    users = segments.drop_duplicates("segment").set_index("segment")["users"].reindex(segs)
    ax.set_yticks(base)
    ax.set_yticklabels([f"{s}\n({int(n):,} users)" for s, n in zip(segs, users)], fontsize=9, color=TEXT)
    ax.set_xlim(0, segments[col].max() * 1.25)
    _style(ax)
    ax.set_title(f"NDCG@{k} by user activity in the train period", fontsize=11, color=TEXT, loc="left", pad=26)
    ax.legend(ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5,
              labelcolor=TEXT, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "ndcg_by_segment.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
