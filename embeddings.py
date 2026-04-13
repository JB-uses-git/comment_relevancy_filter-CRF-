"""
Embedding engine with caching, multiple bi-encoder support, and FAISS vector indexing.
Handles model loading, encoding, caching, and approximate nearest neighbor search.
"""

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

from config import (
    BI_ENCODER_MODELS, DEFAULT_BI_ENCODER, CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_K, EMBEDDING_CACHE_DIR, FAISS_INDEX_DIR,
    FAISS_N_NEIGHBORS
)


class EmbeddingEngine:
    """
    Manages bi-encoder embeddings with disk caching and FAISS indexing.
    Supports multiple models for comparison.
    """

    def __init__(self, model_name: str = DEFAULT_BI_ENCODER):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._indexed_texts: List[str] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"🔄 Loading bi-encoder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded: {self.model_name}")
        return self._model

    def _cache_key(self, texts: List[str]) -> str:
        """Generate a deterministic cache key from model name + text content."""
        content = f"{self.model_name}::{';'.join(texts)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return EMBEDDING_CACHE_DIR / f"{self.model_name.replace('/', '_')}_{cache_key}.npy"

    def encode(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Encode texts to embeddings with disk caching.
        Returns normalized embeddings for cosine similarity via dot product.
        """
        if not texts:
            return np.array([])

        cache_key = self._cache_key(texts)
        cache_path = self._cache_path(cache_key)

        if use_cache and cache_path.exists():
            embeddings = np.load(cache_path)
            if len(embeddings) == len(texts):
                return embeddings

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalize for cosine sim via dot product
        )

        if use_cache:
            np.save(cache_path, embeddings)

        return embeddings

    def encode_query(self, question: str, context: str = "") -> np.ndarray:
        """
        Encode a query (question + optional reference context).
        Averages question and context embeddings for richer representation.
        """
        q_emb = self.encode([question], use_cache=True, show_progress=False)
        if context:
            c_emb = self.encode([context], use_cache=True, show_progress=False)
            combined = (q_emb + c_emb) / 2.0
            # Re-normalize after averaging
            norm = np.linalg.norm(combined, axis=1, keepdims=True)
            combined = combined / norm
            return combined
        return q_emb

    def compute_scores(
        self,
        query_emb: np.ndarray,
        comment_embs: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity scores between query and all comments."""
        # Since embeddings are normalized, dot product = cosine similarity
        scores = np.dot(comment_embs, query_emb.T).flatten()
        return scores

    # ─── FAISS Vector Index ──────────────────────────────────────────────

    def build_faiss_index(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Build a FAISS index from texts/embeddings for fast ANN search.
        Uses Inner Product (equivalent to cosine sim for normalized vectors).
        """
        if embeddings is None:
            embeddings = self.encode(texts)

        self._indexed_texts = texts
        dim = embeddings.shape[1]

        # For small datasets (<10k), flat index is fine
        # For larger datasets, switch to IVF
        if len(texts) < 10000:
            self._faiss_index = faiss.IndexFlatIP(dim)
        else:
            nlist = min(100, len(texts) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.train(embeddings.astype(np.float32))

        self._faiss_index.add(embeddings.astype(np.float32))
        print(f"✅ FAISS index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def search_faiss(
        self,
        query_emb: np.ndarray,
        top_k: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the FAISS index for nearest neighbors.
        Returns (scores, indices) arrays.
        """
        if self._faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")

        top_k = top_k or FAISS_N_NEIGHBORS
        top_k = min(top_k, self._faiss_index.ntotal)

        scores, indices = self._faiss_index.search(
            query_emb.astype(np.float32), top_k
        )
        return scores[0], indices[0]

    def save_faiss_index(self, name: str = "default") -> Path:
        """Save FAISS index and metadata to disk."""
        if self._faiss_index is None:
            raise ValueError("No FAISS index to save.")

        index_path = FAISS_INDEX_DIR / f"{name}_{self.model_name.replace('/', '_')}.index"
        meta_path = FAISS_INDEX_DIR / f"{name}_{self.model_name.replace('/', '_')}.meta"

        faiss.write_index(self._faiss_index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self._indexed_texts, f)

        print(f"💾 FAISS index saved: {index_path}")
        return index_path

    def load_faiss_index(self, name: str = "default") -> bool:
        """Load FAISS index and metadata from disk."""
        index_path = FAISS_INDEX_DIR / f"{name}_{self.model_name.replace('/', '_')}.index"
        meta_path = FAISS_INDEX_DIR / f"{name}_{self.model_name.replace('/', '_')}.meta"

        if not index_path.exists() or not meta_path.exists():
            return False

        self._faiss_index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self._indexed_texts = pickle.load(f)

        print(f"📂 FAISS index loaded: {self._faiss_index.ntotal} vectors")
        return True


class CrossEncoderReranker:
    """
    Cross-encoder for reranking top-k bi-encoder candidates.
    Cross-encoders are more accurate but slower (O(n) per query vs O(1) with embeddings).
    Standard production pattern: bi-encoder retrieval → cross-encoder reranking.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"🔄 Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print(f"✅ Cross-encoder loaded: {self.model_name}")
        return self._model

    def rerank(
        self,
        question: str,
        comments: List[str],
        top_k: int = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank comments by cross-encoder relevance to question.
        Returns list of (original_index, cross_encoder_score) sorted by score desc.
        """
        top_k = top_k or CROSS_ENCODER_TOP_K

        # Build query-document pairs
        pairs = [(question, comment) for comment in comments]

        # Score all pairs
        scores = self.model.predict(pairs, show_progress_bar=len(pairs) > 50)

        # Create (index, score) pairs and sort
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]


def load_all_models() -> dict:
    """Load all configured bi-encoder models for comparison."""
    engines = {}
    for model_name in BI_ENCODER_MODELS:
        print(f"\n{'─' * 50}")
        engines[model_name] = EmbeddingEngine(model_name)
        # Force model load
        _ = engines[model_name].model
    return engines


if __name__ == "__main__":
    # Quick test
    engine = EmbeddingEngine()
    texts = ["Hello world", "How to beat the final boss", "Best pizza in New York"]
    embs = engine.encode(texts)
    print(f"Encoded {len(texts)} texts → shape {embs.shape}")

    # Build and search FAISS
    engine.build_faiss_index(texts, embs)
    query = engine.encode(["final boss strategy"])
    scores, indices = engine.search_faiss(query, top_k=3)
    print(f"Top results: {[(texts[i], f'{s:.3f}') for s, i in zip(scores, indices)]}")
