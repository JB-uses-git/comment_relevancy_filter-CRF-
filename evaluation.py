"""
Evaluation module: train/val/test split, threshold optimization, and comprehensive metrics.
Implements Precision, Recall, F1, NDCG, PR-AUC, confusion matrix, and multi-question testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    auc,
    ndcg_score,
)

from config import (
    TEST_SPLIT_RATIO, VAL_SPLIT_RATIO, RANDOM_SEED,
    THRESHOLD_SWEEP_RANGE
)


def create_splits(
    df: pd.DataFrame,
    test_ratio: float = TEST_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    Stratified by true_label to maintain class balance across splits.

    Returns (train_df, val_df, test_df)
    """
    if "true_label" not in df.columns:
        raise ValueError("DataFrame must have 'true_label' column for splitting")

    label_counts = df["true_label"].value_counts(dropna=False)
    if len(label_counts) < 2 or label_counts.min() < 2:
        print("⚠️  Dataset too small/imbalanced for stratified split; using deterministic non-stratified split.")
        test_size = max(1, int(round(len(df) * test_ratio)))
        val_size = max(1, int(round(len(df) * val_ratio)))
        if test_size + val_size >= len(df):
            val_size = max(1, len(df) // 10)
            test_size = max(1, len(df) // 5)
            if test_size + val_size >= len(df):
                test_size = 1
                val_size = 1

        shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        test = shuffled.iloc[:test_size].copy()
        val = shuffled.iloc[test_size:test_size + val_size].copy()
        train = shuffled.iloc[test_size + val_size:].copy()
        if len(train) == 0:
            train = shuffled.iloc[:-2].copy()
            val = shuffled.iloc[-2:-1].copy()
            test = shuffled.iloc[-1:].copy()

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)

        print(f"Data splits:")
        print(f"   Train: {len(train)} ({train['true_label'].mean():.1%} relevant)")
        print(f"   Val  : {len(val)} ({val['true_label'].mean():.1%} relevant)")
        print(f"   Test : {len(test)} ({test['true_label'].mean():.1%} relevant)")
        return train, val, test

    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_ratio, random_state=seed,
        stratify=df["true_label"]
    )

    # Second split: separate validation from training
    if isinstance(test_ratio, int) and isinstance(val_ratio, int):
        adjusted_val_ratio = val_ratio
    else:
        adjusted_val_ratio = val_ratio / (1 - test_ratio)
        
    train, val = train_test_split(
        train_val, test_size=adjusted_val_ratio, random_state=seed,
        stratify=train_val["true_label"]
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Data splits:")
    print(f"   Train: {len(train)} ({train['true_label'].mean():.1%} relevant)")
    print(f"   Val  : {len(val)} ({val['true_label'].mean():.1%} relevant)")
    print(f"   Test : {len(test)} ({test['true_label'].mean():.1%} relevant)")

    return train, val, test


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    sweep_range: Tuple[float, float, float] = THRESHOLD_SWEEP_RANGE,
) -> Tuple[float, float, List[float], np.ndarray]:
    """
    Sweep thresholds on VALIDATION set to find optimal F1.
    This should ONLY be run on train/val data, never on test.

    Returns: (optimal_threshold, best_f1, f1_scores_list, thresholds_array)
    """
    thresholds = np.arange(*sweep_range)
    f1_scores_list = []

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        f1_scores_list.append(f1)

    best_idx = np.argmax(f1_scores_list)
    optimal = round(thresholds[best_idx], 3)
    best_f1 = f1_scores_list[best_idx]

    print(f"Optimal threshold: {optimal} (F1={best_f1:.3f})")
    return optimal, best_f1, f1_scores_list, thresholds


def evaluate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    split_name: str = "test",
    queries: Optional[np.ndarray] = None,
) -> Dict:
    """
    Evaluate a fixed threshold on a dataset. Use on the held-out TEST set
    for honest evaluation (threshold was optimized on val set).
    """
    y_pred = (scores >= threshold).astype(int)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # PR-AUC
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
    pr_auc_val = auc(recall_vals, precision_vals)

    # NDCG — treats relevance scores as ranking
    ndcg = compute_ndcg(y_true, scores, queries=queries)

    results = {
        "split": split_name,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc_val,
        "ndcg": ndcg,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm,
        "n_samples": len(y_true),
    }

    _print_results(results)
    return results


def compute_ndcg(
    y_true: np.ndarray,
    scores: np.ndarray,
    k_values: List[int] = [5, 10, 20, None],
    queries: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute NDCG (Normalized Discounted Cumulative Gain) at various k values.

    NDCG rewards ranking relevant documents higher — more meaningful than
    binary classification for information retrieval.
    Computes mean NDCG across all queries if queries array is provided.
    """
    ndcg_results = {}
    
    if queries is not None:
        unique_queries = np.unique(queries)
        ndcgs = {k: [] for k in k_values}
        for q in unique_queries:
            mask = (queries == q)
            q_true = y_true[mask].reshape(1, -1)
            q_scores = scores[mask].reshape(1, -1)
            
            # Skip if there's no ground truth variance or no items
            if q_true.shape[1] < 2 or np.sum(q_true) == 0:
                continue
                
            for k in k_values:
                try:
                    val = ndcg_score(q_true, q_scores, k=k)
                    ndcgs[k].append(val)
                except Exception:
                    pass
                    
        for k in k_values:
            label = f"@{k}" if k else "@all"
            ndcg_results[label] = round(np.mean(ndcgs[k]), 3) if len(ndcgs[k]) > 0 else 0.0
            
    else:
        y_true_2d = y_true.reshape(1, -1)
        scores_2d = scores.reshape(1, -1)

        for k in k_values:
            label = f"@{k}" if k else "@all"
            try:
                ndcg_val = ndcg_score(y_true_2d, scores_2d, k=k)
                ndcg_results[label] = round(ndcg_val, 3)
            except Exception:
                ndcg_results[label] = 0.0

    return ndcg_results


def evaluate_multiple_questions(
    df: pd.DataFrame,
    encode_fn,
    threshold: float,
    questions: List[Dict] = None,
) -> pd.DataFrame:
    # Deprecated: The pipeline now computes performance against a unified dataframe of (query, comment) pairs.
    pass


def compare_models(
    df: pd.DataFrame,
    question: str,
    context: str,
    model_engines: dict,
    threshold_per_model: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compare multiple bi-encoder models on the same dataset.
    Returns a comparison DataFrame with metrics for each model.
    """
    results = []
    comments = df["comment"].tolist()
    y_true = df["true_label"].values

    for model_name, engine in model_engines.items():
        print(f"\n🔍 Evaluating: {model_name}")

        # Encode
        query_emb = engine.encode_query(question, context)
        comment_embs = engine.encode(comments)
        scores = engine.compute_scores(query_emb, comment_embs)

        # Find optimal threshold if not provided
        if threshold_per_model and model_name in threshold_per_model:
            threshold = threshold_per_model[model_name]
        else:
            threshold, _, _, _ = find_optimal_threshold(y_true, scores)

        y_pred = (scores >= threshold).astype(int)

        results.append({
            "model": model_name,
            "threshold": threshold,
            "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "ndcg@10": round(compute_ndcg(y_true, scores).get("@10", 0), 3),
        })

    return pd.DataFrame(results)


def _print_results(results: Dict) -> None:
    """Pretty-print evaluation results."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS — {results['split'].upper()} SET")
    print(f"  ({results['n_samples']} samples, threshold={results['threshold']})")
    print(f"{sep}")
    print(f"  Accuracy  : {results['accuracy']:.3f}")
    print(f"  Precision : {results['precision']:.3f}")
    print(f"  Recall    : {results['recall']:.3f}")
    print(f"  F1 Score  : {results['f1']:.3f}")
    print(f"  PR-AUC    : {results['pr_auc']:.3f}")
    print(f"  NDCG      : {results['ndcg']}")
    print(f"  ---------------------------------------")
    print(f"  TP={results['true_positives']}  FP={results['false_positives']}  "
          f"TN={results['true_negatives']}  FN={results['false_negatives']}")
    print(f"{sep}")
