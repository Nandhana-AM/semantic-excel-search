"""
semantic_search.py
Embedding-based semantic search using sentence-transformers + FAISS.
Embeddings are generated dynamically from uploaded Excel data per request.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

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


def clean_query_for_semantic(query: str) -> str:
    """Strip conversational wrappers and filler words to focus semantic search on core skills/roles."""
    q = query.lower().strip()
    prefixes = [
        r"\bfind\s+people\s+skilled\s+in\b",
        r"\bfind\s+someone\s+who\s+knows\b",
        r"\bfind\s+anyone\s+who\s+knows\b",
        r"\bfind\s+people\s+who\s+know\b",
        r"\bsomeone\s+who\s+knows\b",
        r"\bsomeone\s+who\s+know\b",
        r"\banyone\s+who\s+knows\b",
        r"\bpeople\s+who\s+know\b",
        r"\bpeople\s+skilled\s+in\b",
        r"\bwho\s+is\s+expert\s+in\b",
        r"\bexpert\s+in\b",
        r"\bskilled\s+in\b",
        r"\bknowledge\s+of\b",
        r"\bexperienced\s+in\b",
        r"\bbackground\s+in\b",
        r"\bfamiliar\s+with\b",
        r"\bworked\s+on\b",
    ]
    for prefix in prefixes:
        q = re.sub(prefix, "", q)
    fillers = ["find", "show", "list", "get", "give", "who", "people", "person", "anyone", "someone"]
    for filler in fillers:
        q = re.sub(rf"^{filler}\b", "", q).strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q if q else query.lower()


SYNONYM_MAP = {
    "road": "road bridges highway highways civil engineering",
    "roads": "roads bridges highway highways civil engineering",
    "highway": "highway road roads bridges civil engineering",
    "highways": "highways road roads bridges civil engineering",
    "bridge": "bridge road roads highway highways civil engineering",
    "bridges": "bridges road roads highway highways civil engineering",
}


def expand_query_for_semantic(q: str) -> str:
    """Expand specific domain concepts with highly relevant related keywords to boost semantic matching."""
    words = re.findall(r"\b\w+\b", q.lower())
    expansions = []
    for word in words:
        if word in SYNONYM_MAP:
            expansions.append(SYNONYM_MAP[word])
    if expansions:
        q = q + " " + " ".join(list(dict.fromkeys(expansions)))
    return q



def semantic_search(df: pd.DataFrame, query: str, top_k: int = TOP_K, parsed: Optional[Dict[str, Any]] = None) -> List[Dict]:
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

    if parsed is None:
        try:
            from app.query_parser import parse_query
            parsed = parse_query(query)
        except Exception:
            parsed = None

    # Build text representation per row
    row_texts = _rows_to_text(df)

    if EMBEDDINGS_AVAILABLE and _model is not None:
        results = _embedding_search(df, row_texts, query, top_k, parsed)
    else:
        results = _tfidf_search(df, row_texts, query, top_k, parsed)

    return results


# ─── Text conversion ──────────────────────────────────────────────────────────

def _rows_to_text(df: pd.DataFrame) -> List[str]:
    """Convert each DataFrame row to a searchable text string without verbose labels and Name field."""
    target_cols = ["Role", "Location", "Experience", "Skills"]
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in target_cols:
            if col in row:
                val = str(row[col]).strip()
                if val and val.lower() not in ("nan", "none", ""):
                    parts.append(val)
        
        # Fallback to all non-Name columns if target columns are absent
        if not parts:
            for col in df.columns:
                if col != "Name":
                    val = str(row[col]).strip()
                    if val and val.lower() not in ("nan", "none", ""):
                        parts.append(val)
                        
        texts.append(" | ".join(parts))
    return texts


# ─── Embedding-based search (sentence-transformers + FAISS) ──────────────────

def _embedding_search(df: pd.DataFrame, row_texts: List[str], query: str, top_k: int, parsed: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Use sentence-transformers embeddings + FAISS for similarity search."""
    # Generate embeddings
    corpus_embeddings = _model.encode(row_texts, convert_to_numpy=True, show_progress_bar=False)
    cleaned_query = clean_query_for_semantic(query)
    expanded_query = expand_query_for_semantic(cleaned_query)
    query_embedding = _model.encode([expanded_query], convert_to_numpy=True, show_progress_bar=False)

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
    # If dynamic skills are present, pre-encode the dynamic skills text
    dyn_skills_text = None
    dyn_emb_norm = None
    if parsed and parsed.get("dynamic_skills") and EMBEDDINGS_AVAILABLE and _model is not None:
        dyn_skills_text = ", ".join(parsed["dynamic_skills"])
        dyn_emb = _model.encode([dyn_skills_text], convert_to_numpy=True, show_progress_bar=False)
        dyn_emb_norm = dyn_emb / (np.linalg.norm(dyn_emb, axis=1, keepdims=True) + 1e-10)

    # Collect candidate records and their core texts for semantic validation
    candidate_records = []
    cand_core_texts = []
    for idx, score in zip(top_indices, top_scores):
        if idx < len(df):
            record = df.iloc[idx].to_dict()
            boost = 0.0
            if parsed and parsed.get("skills"):
                record_skills = str(record.get("Skills", "")).lower()
                for skill in parsed["skills"]:
                    if skill.lower() in record_skills:
                        boost += 0.12
            
            final_score = min(float(score) + boost, 1.0)
            if final_score > 0.33:  # Increased threshold to filter out poor matches (e.g. 0.33)
                record["_similarity_score"] = round(final_score, 4)
                candidate_records.append(record)
                if dyn_emb_norm is not None:
                    cand_core_texts.append(f"{record.get('Role', '')} | {record.get('Skills', '')}")

    # Batch validate candidate relevance against dynamic skills
    if dyn_emb_norm is not None and cand_core_texts:
        cand_embs = _model.encode(cand_core_texts, convert_to_numpy=True, show_progress_bar=False)
        cand_embs_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)
        
        filtered_records = []
        for record, cand_emb_norm in zip(candidate_records, cand_embs_norm):
            dyn_sim = float(cand_emb_norm @ dyn_emb_norm[0])
            if dyn_sim >= 0.12:
                filtered_records.append(record)
        candidate_records = filtered_records

    # Sort results by final score descending
    results = sorted(candidate_records, key=lambda x: x["_similarity_score"], reverse=True)
    return results


# ─── TF-IDF fallback search ───────────────────────────────────────────────────

def _tfidf_search(df: pd.DataFrame, row_texts: List[str], query: str, top_k: int, parsed: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """TF-IDF based similarity as fallback when sentence-transformers is unavailable."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    cleaned_query = clean_query_for_semantic(query)
    expanded_query = expand_query_for_semantic(cleaned_query)
    all_texts = row_texts + [expanded_query]

    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except Exception:
        return []

    corpus_matrix = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    scores = cosine_similarity(query_vector, corpus_matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    candidate_records = []
    cand_core_texts = []
    for idx in top_indices:
        score = float(scores[idx])
        record = df.iloc[idx].to_dict()
        boost = 0.0
        if parsed and parsed.get("skills"):
            record_skills = str(record.get("Skills", "")).lower()
            for skill in parsed["skills"]:
                if skill.lower() in record_skills:
                    boost += 0.12
        
        final_score = min(score + boost, 1.0)
        if final_score > 0.05: # Increased threshold
            record["_similarity_score"] = round(final_score, 4)
            candidate_records.append(record)
            if parsed and parsed.get("dynamic_skills"):
                cand_core_texts.append(f"{record.get('Role', '')} | {record.get('Skills', '')}")

    if parsed and parsed.get("dynamic_skills") and cand_core_texts:
        dyn_skills_text = ", ".join(parsed["dynamic_skills"])
        try:
            dyn_vector = vectorizer.transform([dyn_skills_text])
            cand_vectors = vectorizer.transform(cand_core_texts)
            sims = cosine_similarity(dyn_vector, cand_vectors).flatten()
            
            filtered_records = []
            for record, sim in zip(candidate_records, sims):
                if sim > 0.0:
                    filtered_records.append(record)
            candidate_records = filtered_records
        except Exception:
            pass

    # Sort results by final score descending
    results = sorted(candidate_records, key=lambda x: x["_similarity_score"], reverse=True)
    return results
