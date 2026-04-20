"""
Configuration for the Comment Relevancy Filter project.
Centralizes all settings safely.
"""

from pathlib import Path
import os

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
FAISS_INDEX_DIR = CACHE_DIR / "faiss_indices"

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_CACHE_DIR = CACHE_DIR / "models"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"

for d in [OUTPUT_DIR, MODEL_CACHE_DIR, EMBEDDING_CACHE_DIR, FAISS_INDEX_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cache env paths
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))

# ─── Data Source ─────────────────────────────────────────────────────────────
DATASET_PATH = DATA_DIR / "gaming_queries_dataset.csv"

# ─── Bi-Encoder Models (for comparison) ─────────────────────────────────────
BI_ENCODER_MODELS = [
    "all-MiniLM-L6-v2",           
    "all-mpnet-base-v2",          
    "multi-qa-MiniLM-L6-cos-v1",  
]

DEFAULT_BI_ENCODER = "multi-qa-MiniLM-L6-cos-v1"

# ─── Cross-Encoder (for reranking) ──────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = 20

# ─── Threshold & Evaluation ─────────────────────────────────────────────────
THRESHOLD_SWEEP_RANGE = (0.05, 0.75, 0.01)  
TEST_SPLIT_RATIO = 250
VAL_SPLIT_RATIO = 250
RANDOM_SEED = 42

# ─── FAISS ───────────────────────────────────────────────────────────────────
FAISS_N_NEIGHBORS = 20
