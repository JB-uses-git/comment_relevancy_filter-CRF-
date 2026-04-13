"""
FastAPI application for the Comment Relevancy Filter.

Endpoints:
  POST /rank       — Input: question + comments → Output: ranked relevant comments with scores
  POST /search     — Input: question → Output: search pre-indexed FAISS database
  GET  /health     — Health check
  GET  /models     — List available models
"""

import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from embeddings import EmbeddingEngine, CrossEncoderReranker
from config import DEFAULT_BI_ENCODER, CROSS_ENCODER_TOP_K, BI_ENCODER_MODELS

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Comment Relevancy Filter API",
    description=(
        "NLP-powered API that ranks forum comments by semantic relevance to a question. "
        "Uses bi-encoder retrieval + optional cross-encoder reranking."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State (lazy-loaded) ─────────────────────────────────────────────

_engines: dict = {}
_reranker: Optional[CrossEncoderReranker] = None


def _get_engine(model_name: str = DEFAULT_BI_ENCODER) -> EmbeddingEngine:
    if model_name not in _engines:
        _engines[model_name] = EmbeddingEngine(model_name)
    return _engines[model_name]


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ─── Request / Response Models ───────────────────────────────────────────────

class RankRequest(BaseModel):
    question: str = Field(..., description="The question to rank comments against")
    context: str = Field("", description="Optional reference best-answer context")
    comments: List[str] = Field(..., description="List of comment texts to rank")
    threshold: float = Field(0.3, description="Relevance score threshold (0-1)")
    top_k: int = Field(20, description="Number of top results to return")
    model: str = Field(DEFAULT_BI_ENCODER, description="Bi-encoder model name")
    use_cross_encoder: bool = Field(False, description="Use cross-encoder reranking")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the best strategy to defeat the Elden Beast?",
                "context": "Use bleed builds, avoid Holy damage, dodge constellation attack sideways.",
                "comments": [
                    "Use Rivers of Blood katana for bleed damage.",
                    "Git gud lmao",
                    "Black Flame incantations deal percentage damage.",
                    "This game is trash compared to Dark Souls 3."
                ],
                "threshold": 0.3,
                "top_k": 10,
                "use_cross_encoder": False,
            }
        }


class RankedComment(BaseModel):
    rank: int
    comment: str
    score: float
    is_relevant: bool
    cross_encoder_score: Optional[float] = None


class RankResponse(BaseModel):
    question: str
    model: str
    threshold: float
    total_comments: int
    relevant_count: int
    irrelevant_count: int
    processing_time_ms: float
    results: List[RankedComment]


class SearchRequest(BaseModel):
    question: str
    context: str = ""
    top_k: int = Field(20, description="Number of nearest neighbors")
    model: str = Field(DEFAULT_BI_ENCODER)


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    faiss_indices: dict


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    """Serve the frontend UI."""
    # Ensure index.html exists; if not, fallback to a basic message
    import os
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "UI not found. Please create 'index.html' in the root directory."}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with loaded model info."""
    faiss_info = {}
    for name, eng in _engines.items():
        if eng._faiss_index:
            faiss_info[name] = eng._faiss_index.ntotal
    return HealthResponse(
        status="ok",
        loaded_models=list(_engines.keys()),
        faiss_indices=faiss_info,
    )


@app.get("/models")
async def list_models():
    """List available bi-encoder models."""
    return {
        "available_models": BI_ENCODER_MODELS,
        "default_model": DEFAULT_BI_ENCODER,
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


@app.post("/rank", response_model=RankResponse)
async def rank_comments(req: RankRequest):
    """
    Rank comments by semantic relevance to a question.

    Pipeline:
    1. Bi-encoder: encode question + comments → cosine similarity scores
    2. (Optional) Cross-encoder: rerank top-k candidates for better accuracy
    3. Apply threshold → separate relevant vs irrelevant
    4. Return ranked results
    """
    if not req.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    if req.model not in BI_ENCODER_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model. Available: {BI_ENCODER_MODELS}"
        )

    start = time.time()

    # Step 1: Bi-encoder scoring
    engine = _get_engine(req.model)
    query_emb = engine.encode_query(req.question, req.context)
    comment_embs = engine.encode(req.comments, show_progress=False)
    bi_scores = engine.compute_scores(query_emb, comment_embs)

    # Step 2: Optional cross-encoder reranking
    ce_scores_map = {}
    if req.use_cross_encoder:
        # Get top candidates from bi-encoder
        top_indices = np.argsort(bi_scores)[::-1][:CROSS_ENCODER_TOP_K]
        top_comments = [req.comments[i] for i in top_indices]

        reranker = _get_reranker()
        reranked = reranker.rerank(req.question, top_comments)

        # Map back to original indices
        for local_idx, ce_score in reranked:
            original_idx = top_indices[local_idx]
            ce_scores_map[original_idx] = ce_score

    # Step 3: Build results
    results = []
    for i in range(len(req.comments)):
        # Use cross-encoder score for ranking if available, otherwise bi-encoder
        final_score = ce_scores_map.get(i, float(bi_scores[i]))
        results.append({
            "idx": i,
            "comment": req.comments[i],
            "bi_score": float(bi_scores[i]),
            "ce_score": ce_scores_map.get(i),
            "final_score": final_score,
            "is_relevant": float(bi_scores[i]) >= req.threshold,
        })

    # Sort by final score descending
    if req.use_cross_encoder:
        # Sort: cross-encoded items first (by CE score), then the rest by bi-encoder
        results.sort(key=lambda x: (x["ce_score"] is not None, x["final_score"]), reverse=True)
    else:
        results.sort(key=lambda x: x["final_score"], reverse=True)

    # Limit to top_k
    results = results[:req.top_k]

    elapsed_ms = (time.time() - start) * 1000

    ranked_comments = [
        RankedComment(
            rank=rank + 1,
            comment=r["comment"],
            score=round(r["bi_score"], 4),
            is_relevant=r["is_relevant"],
            cross_encoder_score=round(r["ce_score"], 4) if r["ce_score"] is not None else None,
        )
        for rank, r in enumerate(results)
    ]

    relevant_count = sum(1 for r in ranked_comments if r.is_relevant)

    return RankResponse(
        question=req.question,
        model=req.model,
        threshold=req.threshold,
        total_comments=len(req.comments),
        relevant_count=relevant_count,
        irrelevant_count=len(ranked_comments) - relevant_count,
        processing_time_ms=round(elapsed_ms, 1),
        results=ranked_comments,
    )


@app.post("/search")
async def search_indexed(req: SearchRequest):
    """
    Search pre-built FAISS index. Must call /index first or
    run the pipeline to build the index.
    """
    engine = _get_engine(req.model)

    if engine._faiss_index is None:
        # Try loading from disk
        if not engine.load_faiss_index("comments"):
            raise HTTPException(
                status_code=404,
                detail="No FAISS index found. Run the main pipeline first to build it."
            )

    query_emb = engine.encode_query(req.question, req.context)
    scores, indices = engine.search_faiss(query_emb, top_k=req.top_k)

    results = []
    for score, idx in zip(scores, indices):
        if idx < len(engine._indexed_texts):
            results.append({
                "rank": len(results) + 1,
                "comment": engine._indexed_texts[int(idx)],
                "score": round(float(score), 4),
            })

    return {
        "question": req.question,
        "model": req.model,
        "results": results,
    }


# ─── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    print(f"🚀 Starting API at http://{API_HOST}:{API_PORT}")
    print(f"📖 Docs at http://localhost:{API_PORT}/docs")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
