"""
Configuration for the Comment Relevancy Filter project.
Centralizes all settings: Reddit API, models, thresholds, paths, etc.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
IS_VERCEL = bool(os.getenv("VERCEL"))

# Keep repository data/cache paths for reading bundled assets (e.g., FAISS index).
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
FAISS_INDEX_DIR = CACHE_DIR / "faiss_indices"

# Use a writable runtime directory on Vercel (/tmp is writable there).
RUNTIME_ROOT = (
    Path(os.getenv("CRF_RUNTIME_DIR", "/tmp/comment_relevancy_filter"))
    if IS_VERCEL
    else PROJECT_ROOT
)
OUTPUT_DIR = RUNTIME_ROOT / "output"
MODEL_CACHE_DIR = RUNTIME_ROOT / "cache" / "models"
EMBEDDING_CACHE_DIR = RUNTIME_ROOT / "cache" / "embeddings"

# Create only writable runtime dirs on import.
for d in [OUTPUT_DIR, MODEL_CACHE_DIR, EMBEDDING_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Ensure transformer/model downloads use writable cache paths in serverless runtime.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))

# ─── Reddit API (PRAW) ──────────────────────────────────────────────────────
# Set these as environment variables or fill in directly
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "CommentRelevancyFilter/1.0")

# Subreddits and search queries for scraping
SCRAPE_CONFIG = {
    "subreddits": ["Eldenring", "gaming", "fromsoftware"],
    "search_queries": [
        "best strategy Elden Beast",
        "how to beat Elden Ring final boss",
        "Elden Beast tips",
        "hardest boss Elden Ring strategy",
        "Elden Ring build guide final boss",
    ],
    "max_posts_per_query": 10,
    "max_comments_per_post": 50,
    "min_comment_length": 15,  # Skip very short comments
}

# ─── Evaluation Questions ────────────────────────────────────────────────────
# Multiple questions for testing generalization (not just Elden Beast)
EVALUATION_QUESTIONS = [
    {
        "id": "elden_beast",
        "question": "What is the best strategy to defeat the Elden Beast final boss in Elden Ring?",
        "context": (
            "Use a bleed build or Black Flame incantations. Avoid Holy weapons as Elden Beast resists them. "
            "Dodge the constellation attack sideways, sprint from the holy beam, and heal freely when it swims away. "
            "Summon Mimic Tear or Black Knife Tiche. Upgrade your flasks to +12 and have at least 8 charges. "
            "Stay at medium load for fast rolls. Learn the golden sword slam dodge and punish after big attacks."
        ),
        "keywords": ["bleed", "flask", "dodge", "summon", "build", "weapon", "incantation",
                      "talisman", "ash", "spell", "vigor", "stamina", "holy", "fire", "roll",
                      "damage", "poise", "attack", "combo", "heal", "rune", "level"],
    },
    {
        "id": "malenia_strategy",
        "question": "How do you beat Malenia Blade of Miquella in Elden Ring?",
        "context": (
            "Malenia heals on every hit so blocking is bad. Use Bloodhound Step to dodge Waterfowl Dance. "
            "Bleed and frost builds work well. Summon Mimic Tear to split aggro. In phase 2 she gains "
            "Scarlet Rot attacks — bring Preserving Boluses. Stagger her with jump attacks."
        ),
        "keywords": ["malenia", "waterfowl", "bleed", "frost", "dodge", "heal", "summon",
                      "rot", "scarlet", "bolus", "stagger", "phase"],
    },
    {
        "id": "best_build",
        "question": "What is the best build for a first playthrough of Elden Ring?",
        "context": (
            "Strength builds with a big weapon are beginner friendly. Vigor to 40 first. "
            "Quality builds let you try everything. Bleed builds are overpowered. "
            "Intelligence builds are strong late game but weak early. Get a good shield."
        ),
        "keywords": ["build", "strength", "vigor", "quality", "bleed", "intelligence",
                      "weapon", "shield", "stat", "level", "dex", "faith", "arcane"],
    },
]

# ─── Bi-Encoder Models (for comparison) ─────────────────────────────────────
BI_ENCODER_MODELS = [
    "all-MiniLM-L6-v2",           # Original baseline — fast, decent
    "all-mpnet-base-v2",           # Larger, better quality
    "multi-qa-MiniLM-L6-cos-v1",  # Trained on Q&A pairs — best for this use case
]

DEFAULT_BI_ENCODER = "multi-qa-MiniLM-L6-cos-v1"

# ─── Cross-Encoder (for reranking) ──────────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = 50  # Number of bi-encoder candidates to rerank

# ─── Threshold & Evaluation ─────────────────────────────────────────────────
THRESHOLD_SWEEP_RANGE = (0.05, 0.75, 0.01)  # start, stop, step
TEST_SPLIT_RATIO = 0.2  # 20% held-out test set
VAL_SPLIT_RATIO = 0.1   # 10% validation set (from remaining 80%)
RANDOM_SEED = 42

# ─── FAISS ───────────────────────────────────────────────────────────────────
FAISS_N_NEIGHBORS = 50  # Number of nearest neighbors to retrieve

# ─── FastAPI ─────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
