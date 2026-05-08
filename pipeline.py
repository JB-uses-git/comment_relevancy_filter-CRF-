"""
Main Pipeline — Semantic Comment Relevancy (Generic IR Standard)
=================================================================

Standard steps:
1. Load balanced (query, comment, true_label) dataset.
2. Split dataset into train/val/test.
3. Optimize threshold on validation.
4. Honest evaluation on test (NDCG + F1).
5. Interactive CLI prompt for custom testing!
"""

import sys
import time
import pandas as pd
import numpy as np

from config import DEFAULT_BI_ENCODER, DATASET_PATH
from embeddings import EmbeddingEngine, CrossEncoderReranker
from evaluation import create_splits, find_optimal_threshold, evaluate_threshold

def main():
    print("="*60)
    print("  GAMING RELEVANCY FILTER (Standard IR Pipeline)")
    print("="*60)

    # 1. Load Dataset
    print("\n[1] Loading Generic Balanced Dataset...")
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Please run dataset_generator.py first.")
        return
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Total Comments: {len(df)}")
    print(f"Relevant: {(df['true_label']==1).sum()} | Irrelevant: {(df['true_label']==0).sum()}")

    # 2. Extract pairs
    queries = df["query"].tolist()
    comments = df["comment"].tolist()
    labels = df["true_label"].values

    # 3. Apply Bi-Encoder
    print(f"\n[2] Generating Embeddings using {DEFAULT_BI_ENCODER}...")
    engine = EmbeddingEngine(DEFAULT_BI_ENCODER)
    
    print("Encoding Queries...")
    query_embs = engine.encode(queries, show_progress=False)
    
    print("Encoding Comments...")
    comment_embs = engine.encode(comments, show_progress=False)
    
    # 4. Compute Pairwise Scores
    print("\n[3] Computing Cosine Similarity Scores...")
    # Because there are thousands of unique pairs, dot product corresponding indices
    scores = np.sum(query_embs * comment_embs, axis=1)
    df["relevance_score"] = scores

    # 5. Splits and Evaluation
    print("\n[4] Creating Val/Test Splits & Tuning Threshold...")
    train_df, val_df, test_df = create_splits(df)

    optimal_threshold, best_f1, f1_scores_list, thresholds_array = find_optimal_threshold(
        val_df["true_label"].values, 
        val_df["relevance_score"].values
    )

    print("\n[5] Honest Test Set Evaluation")
    test_results = evaluate_threshold(
        test_df["true_label"].values, 
        test_df["relevance_score"].values, 
        optimal_threshold, 
        "test",
        queries=test_df["query"].values
    )

    print("\n[6] Generating ML Output Graphs...")
    from visualizations import (
        plot_score_distribution, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, plot_f1_vs_threshold
    )
    import matplotlib
    matplotlib.use("Agg")  # Run headless without opening windows
    plot_score_distribution(df, optimal_threshold, save=True)
    
    test_preds = (test_df["relevance_score"].values >= optimal_threshold).astype(int)
    plot_confusion_matrix(test_df["true_label"].values, test_preds, optimal_threshold, "test", save=True)
    plot_precision_recall_curve(test_df["true_label"].values, test_df["relevance_score"].values, optimal_threshold, "test", save=True)
    plot_roc_curve(test_df["true_label"].values, test_df["relevance_score"].values, "test", save=True)
    plot_f1_vs_threshold(thresholds_array, f1_scores_list, optimal_threshold, best_f1, save=True)
    print("Graphs saved to /output/ folder!")

    # 7. Weak Baseline Model Comparison
    print("\n[7] Running Weak Baseline Model for Comparison...")
    from config import WEAK_BI_ENCODER
    from visualizations import (
        plot_comparison_f1_curves, plot_comparison_pr_curves,
        plot_comparison_roc_curves, plot_comparison_metrics_bar
    )

    weak_engine = EmbeddingEngine(WEAK_BI_ENCODER)

    print(f"Encoding Queries with baseline: {WEAK_BI_ENCODER}...")
    weak_query_embs = weak_engine.encode(queries, show_progress=False)

    print(f"Encoding Comments with baseline: {WEAK_BI_ENCODER}...")
    weak_comment_embs = weak_engine.encode(comments, show_progress=False)

    weak_scores = np.sum(weak_query_embs * weak_comment_embs, axis=1)
    df["weak_score"] = weak_scores

    # Re-split (same seed = identical split, but now has weak_score column)
    _, weak_val_df, weak_test_df = create_splits(df)

    weak_opt_threshold, weak_best_f1, weak_f1_list, weak_thresholds = find_optimal_threshold(
        weak_val_df["true_label"].values,
        weak_val_df["weak_score"].values
    )

    print("\n[8] Weak Baseline — Test Set Evaluation")
    weak_test_results = evaluate_threshold(
        weak_test_df["true_label"].values,
        weak_test_df["weak_score"].values,
        weak_opt_threshold,
        "test (baseline)",
        queries=weak_test_df["query"].values
    )

    # 8. Generate Comparison Graphs
    print("\n[9] Generating Model Comparison Graphs...")
    main_label = "Multi-QA MiniLM (Ours)"
    weak_label = "ALBERT-Small (Baseline)"

    plot_comparison_f1_curves(
        thresholds_array, f1_scores_list, optimal_threshold, best_f1, main_label,
        weak_thresholds, weak_f1_list, weak_opt_threshold, weak_best_f1, weak_label
    )
    plot_comparison_pr_curves(
        test_df["true_label"].values,
        test_df["relevance_score"].values, main_label,
        weak_test_df["weak_score"].values, weak_label
    )
    plot_comparison_roc_curves(
        test_df["true_label"].values,
        test_df["relevance_score"].values, main_label,
        weak_test_df["weak_score"].values, weak_label
    )
    plot_comparison_metrics_bar(test_results, main_label, weak_test_results, weak_label)
    print("Comparison graphs saved to /output/ folder!")

    # 9. Interactive Search Mode
    print("\n" + "="*60)
    print("   [GAME] INTRODUCING DYNAMIC CUSTOM SEARCH")
    print("="*60)
    print("We have built a massive generic FAISS index of all comments.")
    print("Type any question below, and the AI will pull the most relevant answers!")
    
    # Build FAISS index of ALL comments to let user search
    unique_comments = list(set(comments))
    engine.build_faiss_index(unique_comments)
    
    reranker = CrossEncoderReranker()

    while True:
        try:
            user_query = input("\nEnter a gaming question (or 'q' to quit): ")
            if user_query.strip().lower() in ['q', 'quit', 'exit']:
                break
            
            # Step 1: Bi-Encoder Retrieval
            start_t = time.time()
            q_emb = engine.encode([user_query], show_progress=False)
            faiss_scores, faiss_indices = engine.search_faiss(q_emb, top_k=10)
            
            retrieved_comments = [unique_comments[idx] for idx in faiss_indices]
            
            # Step 2: Cross-Encoder Reranking
            print("\n  Reranking results with Cross-Encoder...")
            rerank_results = reranker.rerank(user_query, retrieved_comments, top_k=5)
            
            elapsed = time.time() - start_t
            
            print(f"\n  Top 5 Relevant Results (Found in {elapsed:.2f}s):")
            for rank, (local_idx, score) in enumerate(rerank_results):
                comment = retrieved_comments[local_idx]
                badge = "[RELEVANT]" if score > 0 else "[IRRELEVANT]"
                print(f"  #{rank+1} [{badge}]: {comment}")
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
