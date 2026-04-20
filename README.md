# 🎮 Gaming Semantic Search Filter (Standard IR Pipeline)

**A local Machine Learning Information Retrieval system that matches user queries to the most semantically relevant gaming forum comments.**

This project acts as an offline, dynamic search engine. It solves the problem of simple keyword-matching by using a two-stage Dense Retriever (Bi-Encoder) and Reranker (Cross-Encoder) to understand the *intent* of queries and comments, accurately separating genuinely helpful strategy tips from viral jokes/memes.

## 🚀 How to Run

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate the Dataset**
*(You must do this first! This locally constructs a balanced 1,000-comment ground-truth dataset across 5 games).*
```bash
python dataset_generator.py
```

**3. Run the Evaluation & Search Engine (CLI)**
```bash
python pipeline.py
```
*Wait ~10 seconds for the evaluation script to calculate thresholds and metric scores. At the very end, it will open an interactive CLI where you can type your own gaming questions and get live semantic search results!*

**4. Run the Web Interface (Optional)**
If you prefer a visual experience, a Streamlit frontend is included!
```bash
streamlit run app.py
```

---

## 🧠 Architecture Overview

Instead of hardcoded rules, this uses a standard production NLP pattern:

```text
  [Massive Offline Dataset] 
          │
          ▼
   ┌───────────────┐     ┌───────────────┐
   │ Bi-Encoder    │────▶│ FAISS Index   │  (ANN search, builds vector db)
   │ MiniLM-L6     │     └───────────────┘
   └──────┬────────┘
          │ 
     Fast Retrieval (Top-10 Candidates)
          │
          ▼
   ┌─────────────────┐
   │ Cross-Encoder   │  (Heavy context reranking)
   │ ms-marco        │
   └──────┬──────────┘
          │
          ▼
    Sorted Relevant Results!
```

## ✨ Key Features

- **True Ground Truth Data**: Utilizes a pre-built, perfectly balanced 1,000 comment dataset (`dataset_generator.py`). No more weak heuristic labeling!
- **Dynamic Inference**: There are no hardcoded evaluation arrays. You can literally swap the CSV file to a medical dataset, and it instantly becomes a medical search engine.
- **Two-Stage IR Pipeline**: Uses `multi-qa-MiniLM-L6-cos-v1` for blazing-fast FAISS retrieval, and `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise semantic reranking.
- **Strict ML Metrics**: Evaluates using robust Information Retrieval metrics like **NDCG** (Normalized Discounted Cumulative Gain) and **PR-AUC**.
- **Interactive CLI**: Lets you query the vector database in real-time right in your terminal.

## 📂 Project Structure

```text
comment_relevancy_filter/
├── dataset_generator.py  # Builds the balanced 'gaming_queries_dataset.csv' target dataset
├── config.py             # All settings: ML models, metric thresholds, file paths
├── embeddings.py         # Handles Bi-encoder mapping, FAISS, and Cross-Encoder indexing
├── evaluation.py         # Standard ML Train/Val/Test splits, NDCG, PR-AUC metrics
├── visualizations.py     # Matplotlib charts for analysis (confusion matrix, distributions)
├── pipeline.py           # Main execution loop and CLI Interactive Engine
├── requirements.txt      # Python Dependencies
├── data/                 # Raw and processed datasets
├── cache/                # Embedding cache + local FAISS indices
└── output/               # Generated png charts and reports
```

## 📊 Metrics Explained

| Metric | What it measures in this pipeline |
|--------|-----------------|
| **Precision** | Out of all comments the AI said were helpful, how many actually were? |
| **Recall** | Out of all the truly helpful comments sitting in the database, how many did the AI successfully find? |
| **F1** | The harmonic average of Precision and Recall. |
| **NDCG** | Extremely important ranking metric! It rewards the AI for putting the absolute *best* answer at #1, rather than #5. |
| **ROC** | Standard evaluation curve tracking true positive versus false positive rating across all thresholds! |
