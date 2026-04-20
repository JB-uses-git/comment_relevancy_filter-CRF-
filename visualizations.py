"""
Visualization module: all charts, plots, and visual evaluation outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from adjustText import adjust_text
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    auc,
    roc_curve,
)
from pathlib import Path
from typing import Dict, List, Optional

from config import OUTPUT_DIR


# ─── Style Setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "grid.color": "#333",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 10,
})

COLORS = {
    "relevant": "#00d4aa",
    "irrelevant": "#e94560",
    "accent": "#0f3460",
    "highlight": "#ffd700",
    "line": "#53a8b6",
    "bg_dark": "#1a1a2e",
    "bg_card": "#16213e",
}


def plot_score_distribution(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    save: bool = True,
) -> None:
    """Plot score distribution histogram for relevant vs irrelevant comments."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.hist(
        df[df["true_label"] == 1]["relevance_score"], bins=30, alpha=0.75,
        color=COLORS["relevant"], label="Relevant (ground truth)", edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        df[df["true_label"] == 0]["relevance_score"], bins=30, alpha=0.75,
        color=COLORS["irrelevant"], label="Irrelevant (ground truth)", edgecolor="white", linewidth=0.5,
    )

    if threshold:
        ax.axvline(x=threshold, color=COLORS["highlight"], linestyle="--", linewidth=2,
                    label=f"Threshold = {threshold}")

    ax.set_xlabel("Relevance Score (Cosine Similarity)", fontsize=12)
    ax.set_ylabel("Number of Comments", fontsize=12)
    ax.set_title("Score Distribution: Relevant vs Irrelevant Comments", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_f1_vs_threshold(
    thresholds: np.ndarray,
    f1_scores: List[float],
    optimal_threshold: float,
    best_f1: float,
    split_name: str = "validation",
    save: bool = True,
) -> None:
    """Plot F1 score vs threshold sweep."""
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(thresholds, f1_scores, color=COLORS["line"], linewidth=2.5)
    ax.fill_between(thresholds, f1_scores, alpha=0.1, color=COLORS["line"])
    ax.axvline(
        x=optimal_threshold, color=COLORS["highlight"], linestyle="--", linewidth=2,
        label=f"Optimal = {optimal_threshold} (F1={best_f1:.3f})"
    )
    ax.scatter([optimal_threshold], [best_f1], color=COLORS["highlight"], s=100, zorder=5, edgecolors="white")

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"F1 Score vs Threshold — Optimized on {split_name.upper()} Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "f1_vs_threshold.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    split_name: str = "test",
    save: bool = True,
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Custom colormap
    cmap = sns.light_palette(COLORS["line"], as_cmap=True)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=["Pred Irrelevant", "Pred Relevant"],
        yticklabels=["Actual Irrelevant", "Actual Relevant"],
        linewidths=2, linecolor=COLORS["bg_dark"], cbar=False, ax=ax,
        annot_kws={"size": 20, "weight": "bold", "color": "#1a1a2e"},
    )

    ax.set_title(
        f"Confusion Matrix — {split_name.upper()} Set\n(Threshold: {threshold})",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("Model Prediction", fontsize=11)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"confusion_matrix_{split_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    split_name: str = "test",
    save: bool = True,
) -> None:
    """Plot Precision-Recall curve with AUC."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color=COLORS["line"], linewidth=2.5,
            label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.fill_between(recall_vals, precision_vals, alpha=0.1, color=COLORS["line"])

    # Mark optimal threshold operating point
    y_pred = (scores >= threshold).astype(int)
    opt_p = precision_score(y_true, y_pred, zero_division=0)
    opt_r = recall_score(y_true, y_pred, zero_division=0)
    ax.scatter([opt_r], [opt_p], color=COLORS["highlight"], s=150, zorder=5,
               edgecolors="white", linewidth=2,
               label=f"Operating point (T={threshold}) → P={opt_p:.2f}, R={opt_r:.2f}")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        f"Precision-Recall Curve — {split_name.upper()} Set",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"precision_recall_{split_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    split_name: str = "test",
    save: bool = True,
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS["line"], linewidth=2.5,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=COLORS["irrelevant"], linestyle="--", linewidth=2, label="Random Guess")
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["line"])

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {split_name.upper()} Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, facecolor=COLORS["bg_card"], edgecolor="#555", loc="lower right")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / f"roc_curve_{split_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()



def plot_top_comments(
    top_df: pd.DataFrame,
    question: str,
    threshold: float,
    save: bool = True,
) -> None:
    """Plot the main 2-panel visualization: bar chart + scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f'Comment Relevancy Filter — Top 20 Upvoted Comments\nQ: "{question[:72]}..."',
        fontsize=12, fontweight="bold", y=1.02, color="white"
    )

    # ── Panel 1: Horizontal bars ──
    ax1 = axes[0]
    colors = [COLORS["relevant"] if s >= threshold else COLORS["irrelevant"]
              for s in top_df["relevance_score"]]
    ax1.barh(range(len(top_df)), top_df["relevance_score"], color=colors,
             edgecolor="white", linewidth=0.3, alpha=0.9)
    ax1.axvline(x=threshold, color=COLORS["highlight"], linestyle="--", linewidth=2,
                label=f"Threshold = {threshold}")
    ax1.set_yticks(range(len(top_df)))
    ax1.set_yticklabels(
        [f"#{i+1} [{row['upvotes']:,}↑]" for i, row in top_df.iterrows()],
        fontsize=7
    )
    ax1.set_xlabel("Relevance Score", fontsize=10)
    ax1.set_title("Relevance Score per Comment\n(ranked by upvotes)", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()

    green_patch = mpatches.Patch(color=COLORS["relevant"], label="Relevant")
    red_patch = mpatches.Patch(color=COLORS["irrelevant"], label="Irrelevant")
    line_handle = plt.Line2D([0], [0], color=COLORS["highlight"], linestyle="--", label=f"T={threshold}")
    ax1.legend(handles=[green_patch, red_patch, line_handle], fontsize=8, loc="lower right",
               facecolor=COLORS["bg_card"], edgecolor="#555")
    ax1.set_xlim(min(0, top_df["relevance_score"].min()) - 0.05, max(top_df["relevance_score"]) + 0.1)
    ax1.grid(axis="x", alpha=0.2)

    # ── Panel 2: Scatter plot ──
    ax2 = axes[1]
    rel_mask = top_df["relevance_score"] >= threshold
    ax2.scatter(top_df[rel_mask]["relevance_score"], top_df[rel_mask]["upvotes"],
                color=COLORS["relevant"], s=140, zorder=5, label="Relevant",
                edgecolors="white", linewidth=0.8, alpha=0.9)
    ax2.scatter(top_df[~rel_mask]["relevance_score"], top_df[~rel_mask]["upvotes"],
                color=COLORS["irrelevant"], s=140, zorder=5, label="Irrelevant",
                edgecolors="white", linewidth=0.8, alpha=0.9)

    texts = []
    for i, row in top_df.iterrows():
        texts.append(ax2.text(row["relevance_score"], row["upvotes"], f"#{i+1}", fontsize=7))
    try:
        adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except Exception:
        pass

    ax2.axvline(x=threshold, color=COLORS["highlight"], linestyle="--", linewidth=2,
                label=f"Threshold = {threshold}")
    ax2.set_xlabel("Relevance Score", fontsize=10)
    ax2.set_ylabel("Upvotes", fontsize=10)
    ax2.set_title("Upvotes vs Relevance Score\n(top-left = viral but irrelevant)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax2.grid(alpha=0.2)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "top_comments_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot model comparison bar chart."""
    metrics = ["precision", "recall", "f1", "ndcg@10"]
    x = np.arange(len(comparison_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    palette = [COLORS["relevant"], COLORS["line"], COLORS["highlight"], COLORS["irrelevant"]]
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric.upper(),
               color=palette[i], edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Bi-Encoder Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["model"], fontsize=8, rotation=15)
    ax.legend(fontsize=10, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "model_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_cross_encoder_improvement(
    bi_scores: np.ndarray,
    ce_scores: np.ndarray,
    y_true: np.ndarray,
    save: bool = True,
) -> None:
    """
    Show how cross-encoder reranking improves over bi-encoder scores.
    Side-by-side scatter: bi-encoder score vs cross-encoder score, colored by true label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, scores, title in [
        (axes[0], bi_scores, "Bi-Encoder Scores"),
        (axes[1], ce_scores, "Cross-Encoder (Reranked) Scores"),
    ]:
        rel_mask = y_true == 1
        ax.hist(scores[rel_mask], bins=25, alpha=0.75, color=COLORS["relevant"],
                label="Relevant", edgecolor="white", linewidth=0.5)
        ax.hist(scores[~rel_mask], bins=25, alpha=0.75, color=COLORS["irrelevant"],
                label="Irrelevant", edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, facecolor=COLORS["bg_card"], edgecolor="#555")
        ax.grid(alpha=0.2)

    fig.suptitle("Bi-Encoder vs Cross-Encoder Score Separation", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "cross_encoder_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()


def plot_multi_question_results(
    results_df: pd.DataFrame,
    save: bool = True,
) -> None:
    """Plot multi-question evaluation results as grouped bars."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(results_df))
    width = 0.25
    metrics = ["precision", "recall", "f1"]
    palette = [COLORS["relevant"], COLORS["line"], COLORS["highlight"]]

    for i, (metric, color) in enumerate(zip(metrics, palette)):
        ax.bar(x + i * width, results_df[metric], width, label=metric.upper(),
               color=color, edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Question", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Threshold Generalization Across Multiple Questions", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["question_id"], fontsize=10)
    ax.legend(fontsize=10, facecolor=COLORS["bg_card"], edgecolor="#555")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "multi_question_results.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {path}")
    plt.show()
    plt.close()
