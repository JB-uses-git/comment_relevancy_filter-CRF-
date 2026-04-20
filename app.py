import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

from embeddings import EmbeddingEngine, CrossEncoderReranker
from config import DEFAULT_BI_ENCODER, DATASET_PATH, OUTPUT_DIR

st.set_page_config(page_title="Semantic Review Classifier", page_icon="🎮", layout="wide")

@st.cache_resource
def load_models():
    engine = EmbeddingEngine(DEFAULT_BI_ENCODER)
    reranker = CrossEncoderReranker()
    
    # Load and index data so search works
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        comments = list(set(df["comment"].tolist()))
        engine.build_faiss_index(comments)
    else:
        comments = []
        st.error("Dataset not generated! Please run `python dataset_generator.py` first.")
        
    return engine, reranker, comments

st.title("🎮 Gaming Semantic Search Engine")
st.markdown("A two-stage IR Retrieval pipeline using Bi-Encoders and Cross-Encoders.")

engine, reranker, comments = load_models()

# --- SIDEBAR FOR GRAPHICS ---
with st.sidebar:
    st.header("📊 ML Metrics Analysis")
    st.markdown("These plots visually demonstrate the models competence over the Test set.")
    
    # If the user has ran the pipeline
    if os.path.exists(OUTPUT_DIR / "score_distribution.png"):
        img = Image.open(OUTPUT_DIR / "score_distribution.png")
        st.image(img, caption="Vector Similarity Scores")
    
    if os.path.exists(OUTPUT_DIR / "confusion_matrix_test.png"):
        img = Image.open(OUTPUT_DIR / "confusion_matrix_test.png")
        st.image(img, caption="Confusion Matrix")

    if os.path.exists(OUTPUT_DIR / "precision_recall_test.png"):
        img = Image.open(OUTPUT_DIR / "precision_recall_test.png")
        st.image(img, caption="Precision-Recall Curve")
        
    if os.path.exists(OUTPUT_DIR / "roc_curve_test.png"):
        img = Image.open(OUTPUT_DIR / "roc_curve_test.png")
        st.image(img, caption="ROC Curve")
        
    if not os.path.exists(OUTPUT_DIR / "score_distribution.png"):
        st.warning("Run `python pipeline.py` first to generate the ML evaluation graphs here.")

# --- SEARCH FRONTEND ---
st.subheader("Try the Model")
query = st.text_input("Ask a gaming question (e.g. 'How to dodge Malenias attacks?', 'Best agent for Ascent?'):")

if query:
    if not comments:
        st.error("Vector database is empty. Generate the dataset first.")
    else:
        with st.spinner("1️⃣ Bi-Encoder fetching Top-10 vectors (Fast Retrieval)..."):
            q_emb = engine.encode([query], show_progress=False)
            scores, indices = engine.search_faiss(q_emb, top_k=10)
            retrieved_comments = [comments[idx] for idx in indices]
            
        with st.spinner("2️⃣ Cross-Encoder rigorously scoring candidate pairs (Reranking)..."):
            rerank_results = reranker.rerank(query, retrieved_comments, top_k=5)
            
        st.success("Retrieval Complete!")
        
        for rank, (local_idx, score) in enumerate(rerank_results):
            comment = retrieved_comments[local_idx]
            
            # Using arbitrary threshold > 0 for standard ms-marco logic to visually separate good/bad
            if score > 0:
                st.success(f"**#{rank+1} (Score: {score:.2f})** - {comment}")
            else:
                st.error(f"**#{rank+1} (Score: {score:.2f})** - {comment}")
