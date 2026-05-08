"""Temporary script to capture evaluation results for the report."""
import pandas as pd
import numpy as np
from config import DEFAULT_BI_ENCODER, DATASET_PATH
from embeddings import EmbeddingEngine
from evaluation import create_splits, find_optimal_threshold, evaluate_threshold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 1. Load dataset
df = pd.read_csv(DATASET_PATH)
total = len(df)
rel = (df["true_label"] == 1).sum()
irr = (df["true_label"] == 0).sum()
print(f"Total: {total}, Relevant: {rel}, Irrelevant: {irr}")

# 2. Encode
engine = EmbeddingEngine(DEFAULT_BI_ENCODER)
queries = df["query"].tolist()
comments = df["comment"].tolist()
query_embs = engine.encode(queries, show_progress=False)
comment_embs = engine.encode(comments, show_progress=False)
scores = np.sum(query_embs * comment_embs, axis=1)
df["relevance_score"] = scores

# 3. Splits
train_df, val_df, test_df = create_splits(df)

# 4. Threshold sweep on validation
optimal_threshold, best_f1, f1_list, thresholds = find_optimal_threshold(
    val_df["true_label"].values, val_df["relevance_score"].values
)

# 5. Test evaluation
print("\n=== TEST SET EVALUATION ===")
test_results = evaluate_threshold(
    test_df["true_label"].values, test_df["relevance_score"].values, optimal_threshold, "test",
    queries=test_df["query"].values
)

# 6. Val evaluation
print("\n=== VALIDATION SET EVALUATION ===")
val_results = evaluate_threshold(
    val_df["true_label"].values, val_df["relevance_score"].values, optimal_threshold, "validation",
    queries=val_df["query"].values
)

# 7. Multi-threshold table
print("\n--- THRESHOLD SWEEP TABLE ---")
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    y_true = test_df["true_label"].values
    y_pred = (test_df["relevance_score"].values >= t).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    print(f"T={t:.2f} | P={p:.3f} | R={r:.3f} | F1={f:.3f} | Acc={a:.3f}")

# 8. Score stats
print(f"\nScore stats:")
print(f"  Min: {scores.min():.4f}")
print(f"  Max: {scores.max():.4f}")
print(f"  Mean: {scores.mean():.4f}")
print(f"  Std: {scores.std():.4f}")
rel_mean = df[df["true_label"] == 1]["relevance_score"].mean()
irr_mean = df[df["true_label"] == 0]["relevance_score"].mean()
print(f"  Relevant mean: {rel_mean:.4f}")
print(f"  Irrelevant mean: {irr_mean:.4f}")

# 9. Dataset info
print(f"\nDataset: {DATASET_PATH}")
print(f"Unique queries: {df['query'].nunique()}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Model: {DEFAULT_BI_ENCODER}")
print(f"Embedding dim: {query_embs.shape[1]}")
