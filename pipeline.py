"""
Main Pipeline — Comment Relevancy Filter v3
============================================

Complete end-to-end pipeline:
1. Load data (Reddit API or synthetic fallback)
2. Train/Val/Test split
3. Multi-model comparison (3 bi-encoders)
4. Threshold optimization on VALIDATION set
5. Honest evaluation on held-out TEST set
6. Cross-encoder reranking
7. NDCG evaluation
8. Multi-question generalization testing
9. FAISS index building
10. Full visualization suite

Run with: python pipeline.py
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_BI_ENCODER, BI_ENCODER_MODELS, CROSS_ENCODER_TOP_K,
    EVALUATION_QUESTIONS, OUTPUT_DIR
)
from data_scraper import load_or_scrape_data
from embeddings import EmbeddingEngine, CrossEncoderReranker
from evaluation import (
    create_splits, find_optimal_threshold, evaluate_threshold,
    evaluate_multiple_questions, compare_models, compute_ndcg
)
from visualizations import (
    plot_score_distribution, plot_f1_vs_threshold, plot_confusion_matrix,
    plot_precision_recall_curve, plot_top_comments, plot_model_comparison,
    plot_cross_encoder_improvement, plot_multi_question_results
)


def main():
    start_time = time.time()
    print("=" * 70)
    print("  🎮 COMMENT RELEVANCY FILTER — v3 PIPELINE")
    print("  Full evaluation with train/val/test splits, multi-model,")
    print("  cross-encoder reranking, NDCG, and multi-question testing")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 1: Loading Data")
    print("─" * 70)

    df = load_or_scrape_data()
    print(f"\n  Total: {len(df)} comments")
    print(f"  Relevant: {df['true_label'].sum()} | Irrelevant: {(df['true_label']==0).sum()}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Train/Val/Test Split
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 2: Creating Stratified Splits")
    print("─" * 70)

    train_df, val_df, test_df = create_splits(df)

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Load Default Model & Score Everything
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print(f"  STEP 3: Encoding with Default Model ({DEFAULT_BI_ENCODER})")
    print("─" * 70)

    question_config = EVALUATION_QUESTIONS[0]  # Elden Beast
    QUESTION = question_config["question"]
    CONTEXT = question_config["context"]

    engine = EmbeddingEngine(DEFAULT_BI_ENCODER)

    # Encode query
    query_emb = engine.encode_query(QUESTION, CONTEXT)
    print(f"  Query embedding shape: {query_emb.shape}")

    # Encode all comments
    all_comments = df["comment"].tolist()
    all_embs = engine.encode(all_comments, show_progress=True)
    print(f"  Comment embeddings shape: {all_embs.shape}")

    # Compute scores for all splits
    all_scores = engine.compute_scores(query_emb, all_embs)
    df["relevance_score"] = all_scores

    # Map scores to splits
    train_scores = engine.compute_scores(
        query_emb, engine.encode(train_df["comment"].tolist(), show_progress=False)
    )
    val_scores = engine.compute_scores(
        query_emb, engine.encode(val_df["comment"].tolist(), show_progress=False)
    )
    test_scores = engine.compute_scores(
        query_emb, engine.encode(test_df["comment"].tolist(), show_progress=False)
    )

    train_df["relevance_score"] = train_scores
    val_df["relevance_score"] = val_scores
    test_df["relevance_score"] = test_scores

    print(f"\n  Score ranges:")
    print(f"    Train: {train_scores.min():.3f} — {train_scores.max():.3f}")
    print(f"    Val  : {val_scores.min():.3f} — {val_scores.max():.3f}")
    print(f"    Test : {test_scores.min():.3f} — {test_scores.max():.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Optimize Threshold on VALIDATION Set
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 4: Threshold Optimization (VALIDATION set only)")
    print("─" * 70)

    optimal_threshold, best_f1, f1_scores, thresholds = find_optimal_threshold(
        val_df["true_label"].values, val_scores
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate on HELD-OUT TEST Set (Honest Evaluation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 5: Evaluation on HELD-OUT TEST Set")
    print("  (Threshold was optimized on val set — this is honest evaluation)")
    print("─" * 70)

    test_results = evaluate_threshold(
        test_df["true_label"].values, test_scores, optimal_threshold, "test"
    )

    # Also evaluate on validation set for comparison
    val_results = evaluate_threshold(
        val_df["true_label"].values, val_scores, optimal_threshold, "validation"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Multi-Model Comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 6: Comparing Multiple Bi-Encoder Models")
    print("─" * 70)

    model_engines = {}
    for model_name in BI_ENCODER_MODELS:
        model_engines[model_name] = EmbeddingEngine(model_name)

    comparison_df = compare_models(
        val_df, QUESTION, CONTEXT, model_engines
    )
    print(f"\n{comparison_df.to_string(index=False)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: Cross-Encoder Reranking
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 7: Cross-Encoder Reranking (top-k improvement)")
    print("─" * 70)

    reranker = CrossEncoderReranker()

    # Get bi-encoder top candidates from test set
    test_comments = test_df["comment"].tolist()
    test_true = test_df["true_label"].values

    # Use bi-encoder to get top-k candidates
    top_k_indices = np.argsort(test_scores)[::-1][:CROSS_ENCODER_TOP_K]
    top_k_comments = [test_comments[i] for i in top_k_indices]
    top_k_true = test_true[top_k_indices]
    top_k_bi_scores = test_scores[top_k_indices]

    # Rerank with cross-encoder
    ce_results = reranker.rerank(QUESTION, top_k_comments, top_k=len(top_k_comments))
    ce_scores_arr = np.zeros(len(top_k_comments))
    for local_idx, ce_score in ce_results:
        ce_scores_arr[local_idx] = ce_score

    # Compare NDCG improvement
    bi_ndcg = compute_ndcg(top_k_true, top_k_bi_scores)
    ce_ndcg = compute_ndcg(top_k_true, ce_scores_arr)

    print(f"\n  Bi-Encoder  NDCG@10: {bi_ndcg.get('@10', 0):.3f} | NDCG@all: {bi_ndcg.get('@all', 0):.3f}")
    print(f"  Cross-Enc.  NDCG@10: {ce_ndcg.get('@10', 0):.3f} | NDCG@all: {ce_ndcg.get('@all', 0):.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Multi-Question Generalization Test
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 8: Multi-Question Generalization Test")
    print("  (Does the threshold work for other questions?)")
    print("─" * 70)

    def encode_fn(question, context, comments):
        q_emb = engine.encode_query(question, context)
        c_embs = engine.encode(comments, show_progress=False)
        return engine.compute_scores(q_emb, c_embs)

    multi_q_results = evaluate_multiple_questions(df, encode_fn, optimal_threshold)
    print(f"\n{multi_q_results.to_string(index=False)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: Build FAISS Index
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 9: Building FAISS Vector Index")
    print("─" * 70)

    engine.build_faiss_index(all_comments, all_embs)
    engine.save_faiss_index("comments")

    # Quick search demo
    demo_scores, demo_indices = engine.search_faiss(query_emb, top_k=5)
    print("\n  FAISS Search Demo (top 5 for Elden Beast question):")
    for rank, (score, idx) in enumerate(zip(demo_scores, demo_indices)):
        print(f"    #{rank+1} [score={score:.3f}] {all_comments[idx][:80]}...")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: Generate All Visualizations
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 10: Generating Visualizations")
    print("─" * 70)

    # Add predictions to full dataset
    df["predicted_label"] = (df["relevance_score"] >= optimal_threshold).astype(int)

    # 1. Score distribution
    plot_score_distribution(df, optimal_threshold)

    # 2. F1 vs threshold
    plot_f1_vs_threshold(thresholds, f1_scores, optimal_threshold, best_f1, "validation")

    # 3. Confusion matrix on test set
    test_preds = (test_scores >= optimal_threshold).astype(int)
    plot_confusion_matrix(test_df["true_label"].values, test_preds, optimal_threshold, "test")

    # 4. Precision-recall curve on test set
    plot_precision_recall_curve(test_df["true_label"].values, test_scores, optimal_threshold, "test")

    # 5. Top 20 analysis
    top20 = df.nlargest(20, "upvotes").reset_index(drop=True)
    plot_top_comments(top20, QUESTION, optimal_threshold)

    # 6. Model comparison
    plot_model_comparison(comparison_df)

    # 7. Cross-encoder improvement
    plot_cross_encoder_improvement(top_k_bi_scores, ce_scores_arr, top_k_true)

    # 8. Multi-question results
    if len(multi_q_results) > 0:
        plot_multi_question_results(multi_q_results)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print("\n" + "═" * 70)
    print("  📊 FINAL PROJECT REPORT")
    print("═" * 70)
    print(f"  Dataset          : {len(df)} comments ({df['true_label'].sum()} relevant, {(df['true_label']==0).sum()} irrelevant)")
    print(f"  Splits           : Train={len(train_df)} / Val={len(val_df)} / Test={len(test_df)}")
    print(f"  Default Model    : {DEFAULT_BI_ENCODER}")
    print(f"  Cross-Encoder    : cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"  FAISS Index      : {engine._faiss_index.ntotal} vectors")
    print(f"  Optimal Threshold: {optimal_threshold} (optimized on VAL set)")
    print()
    print(f"  ─── Validation Set Results ───")
    print(f"  Accuracy  : {val_results['accuracy']:.3f}")
    print(f"  Precision : {val_results['precision']:.3f}")
    print(f"  Recall    : {val_results['recall']:.3f}")
    print(f"  F1 Score  : {val_results['f1']:.3f}")
    print(f"  PR-AUC    : {val_results['pr_auc']:.3f}")
    print()
    print(f"  ─── TEST Set Results (Honest Evaluation) ───")
    print(f"  Accuracy  : {test_results['accuracy']:.3f}")
    print(f"  Precision : {test_results['precision']:.3f}")
    print(f"  Recall    : {test_results['recall']:.3f}")
    print(f"  F1 Score  : {test_results['f1']:.3f}")
    print(f"  PR-AUC    : {test_results['pr_auc']:.3f}")
    print(f"  NDCG      : {test_results['ndcg']}")
    print()
    print(f"  ─── Cross-Encoder Improvement ───")
    print(f"  Bi-Enc  NDCG@10 : {bi_ndcg.get('@10', 0):.3f}")
    print(f"  Cross   NDCG@10 : {ce_ndcg.get('@10', 0):.3f}")
    print()
    print(f"  ─── Models Compared ───")
    for _, row in comparison_df.iterrows():
        print(f"  {row['model']:40s} F1={row['f1']:.3f} NDCG@10={row['ndcg@10']:.3f}")
    print()
    print(f"  ─── Multi-Question Generalization ───")
    for _, row in multi_q_results.iterrows():
        print(f"  {row['question_id']:20s} F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f}")
    print()
    print(f"  ─── Output Files ───")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  📊 {f.name}")
    print()
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  API: Run `python api.py` then visit http://localhost:8000/docs")
    print("═" * 70)

    return {
        "optimal_threshold": optimal_threshold,
        "test_results": test_results,
        "val_results": val_results,
        "comparison_df": comparison_df,
        "multi_q_results": multi_q_results,
    }


if __name__ == "__main__":
    results = main()
