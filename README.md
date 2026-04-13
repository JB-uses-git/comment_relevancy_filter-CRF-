# 🎮 Comment Relevancy Filter v3

**NLP-powered system that filters forum comments by semantic relevance to a question.**

Separates genuinely helpful strategy tips from viral jokes/memes — even when irrelevant comments have 10x more upvotes.

## Architecture

```
Question + Context
       │
       ▼
  ┌─────────────┐     ┌───────────────┐
  │ Bi-Encoder   │────▶│ FAISS Index   │  (ANN search, cached embeddings)
  │ (3 models)   │     └───────────────┘
  └──────┬───────┘
         │ top-k candidates
         ▼
  ┌─────────────────┐
  │ Cross-Encoder    │  (reranking for better accuracy)
  │ ms-marco-MiniLM  │
  └──────┬──────────┘
         │ reranked scores
         ▼
  ┌─────────────┐
  │ Threshold    │  (optimized on validation set)
  │ Filter       │
  └──────┬───────┘
         │
    Relevant / Irrelevant
```

## Features

- **Real Data**: Reddit API scraping via PRAW (r/Eldenring, r/gaming) — falls back to 300-comment synthetic dataset
- **Proper Evaluation**: Train/Val/Test splits — threshold optimized on val, evaluated on held-out test
- **Multi-Model Comparison**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`
- **Cross-Encoder Reranking**: Bi-encoder retrieval → cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranking
- **FAISS Vector Store**: Pre-computed embeddings indexed for ANN search
- **Embedding Cache**: Disk-cached embeddings — no re-encoding on repeat runs
- **NDCG Metric**: Ranking-aware evaluation, not just binary classification
- **Multi-Question Testing**: Tests threshold generalization across 3 different questions
- **FastAPI Endpoint**: Deployable API with `/rank` and `/search` endpoints
- **Full Visualizations**: Score distributions, PR curves, confusion matrices, model comparisons

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python pipeline.py

# Start the API server
python api.py
# Then visit http://localhost:8000/docs
```

## Using Real Reddit Data

Set environment variables before running:

```bash
set REDDIT_CLIENT_ID=your_client_id
set REDDIT_CLIENT_SECRET=your_client_secret
python pipeline.py
```

Get credentials at https://www.reddit.com/prefs/apps (create a "script" app).

### Labeling real data (recommended)

For honest model evaluation, store human-labeled Reddit comments in:

`data/reddit_comments_labeled.csv`

Required columns:

- `comment`
- `upvotes`
- `true_label` (1 relevant, 0 irrelevant)
- `topic` (`elden_beast`, `malenia_strategy`, `best_build`, or `irrelevant`)

If this file is missing, the pipeline will still run by applying weak heuristic labels to raw scraped data. That path is useful for smoke testing but should not be treated as final quality evaluation.

## API Usage

```bash
# Start server
python api.py

# Rank comments
curl -X POST http://localhost:8000/rank \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How to beat Elden Beast?",
    "comments": ["Use bleed build", "Git gud lol", "Black Flame works great"],
    "threshold": 0.3,
    "use_cross_encoder": true
  }'
```

## Project Structure

```
comment_relevancy_filter/
├── config.py           # All settings: models, paths, thresholds
├── data_scraper.py     # Reddit PRAW scraper + synthetic fallback
├── embeddings.py       # Bi-encoder, cross-encoder, FAISS, caching
├── evaluation.py       # Splits, metrics, NDCG, multi-question
├── visualizations.py   # All charts and plots
├── api.py              # FastAPI server
├── pipeline.py         # Main end-to-end pipeline
├── requirements.txt    # Dependencies
├── data/               # Raw and processed data
├── cache/              # Embedding cache + FAISS indices
└── output/             # Generated charts and reports
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| Precision | Of comments predicted relevant, how many actually were |
| Recall | Of truly relevant comments, how many were caught |
| F1 | Harmonic mean of precision and recall |
| PR-AUC | Area under precision-recall curve |
| NDCG@k | Ranking quality — rewards placing relevant items higher |
