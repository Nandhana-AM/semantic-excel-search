"""
semantic_search.py
Embedding-based semantic search using sentence-transformers + FAISS.
Embeddings are generated dynamically from uploaded Excel data per request.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

# Try to import sentence-transformers + FAISS
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    _model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    EMBEDDINGS_AVAILABLE = False
    _model = None

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# Fallback: TF-IDF based similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 10


def semantic_search(df: pd.DataFrame, query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Perform semantic search on the DataFrame using the query.

    Flow:
    1. Convert each row to a text representation
    2. Generate embeddings (sentence-transformers if available, else TF-IDF)
    3. Build FAISS index (if available) or cosine similarity
    4. Find top-k most similar rows
    """
    if df.empty:
        return []

    # Build text representation per row
    row_texts = _rows_to_text(df)

    if EMBEDDINGS_AVAILABLE and _model is not None:
        results = _embedding_search(df, row_texts, query, top_k)
    else:
        results = _tfidf_search(df, row_texts, query, top_k)

    return results


# ─── Text conversion ──────────────────────────────────────────────────────────

def _rows_to_text(df: pd.DataFrame) -> List[str]:
    """Convert each DataFrame row to a searchable text string."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            val = str(row[col]).strip()
            if val and val.lower() not in ("nan", "none", ""):
                parts.append(f"{col}: {val}")
        texts.append(" | ".join(parts))
    return texts


# ─── Embedding-based search (sentence-transformers + FAISS) ──────────────────

def _embedding_search(df: pd.DataFrame, row_texts: List[str], query: str, top_k: int) -> List[Dict]:
    """Use sentence-transformers embeddings + FAISS for similarity search."""
    # Generate embeddings
    corpus_embeddings = _model.encode(row_texts, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = _model.encode([query], convert_to_numpy=True, show_progress_bar=False)

    # Normalize for cosine similarity
    corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10)
    query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)

    if FAISS_AVAILABLE:
        dim = corpus_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(corpus_norm.astype(np.float32))
        distances, indices = index.search(query_norm.astype(np.float32), min(top_k, len(df)))
        top_indices = indices[0].tolist()
        top_scores = distances[0].tolist()
    else:
        # Fallback: numpy dot product
        scores = (corpus_norm @ query_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        top_scores = scores[top_indices].tolist()

    results = []
    for idx, score in zip(top_indices, top_scores):
        if idx < len(df) and score > 0.1:  # threshold
            record = df.iloc[idx].to_dict()
            record["_similarity_score"] = round(float(score), 4)
            results.append(record)

    return results


# ─── TF-IDF fallback search ───────────────────────────────────────────────────

def _tfidf_search(df: pd.DataFrame, row_texts: List[str], query: str, top_k: int) -> List[Dict]:
    """TF-IDF based similarity as fallback when sentence-transformers is unavailable."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    all_texts = row_texts + [query]

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except Exception:
        return []

    corpus_matrix = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    scores = cosine_similarity(query_vector, corpus_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score > 0.01:
            record = df.iloc[idx].to_dict()
            record["_similarity_score"] = round(score, 4)
            results.append(record)

    return results
