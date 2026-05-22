"""
query_parser.py
Parses natural language queries using spaCy + regex + rule-based logic.
Determines whether the query should go to:
  - structured search (role/location/experience filters)
  - semantic search (skills, fuzzy expertise)
  - hybrid search (combination)
"""

import re
from enum import Enum
from typing import Dict, Any, Optional

# Try to import spaCy; gracefully degrade if not available
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

# ─── Enums ────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    STRUCTURED = "structured"
    SEMANTIC   = "semantic"
    HYBRID     = "hybrid"

# ─── Known keyword banks ──────────────────────────────────────────────────────

ROLE_KEYWORDS = [
    "engineer", "developer", "manager", "analyst", "architect", "designer",
    "consultant", "lead", "director", "officer", "intern", "specialist",
    "tester", "qa", "devops", "data scientist", "ml engineer", "civil",
    "mechanical", "electrical", "software", "backend", "frontend", "fullstack"
]

LOCATION_KEYWORDS = [
    "chennai", "bangalore", "mumbai", "delhi", "hyderabad", "pune",
    "kolkata", "noida", "gurgaon", "remote", "onsite", "hybrid location",
    "new york", "london", "dubai", "singapore"
]

EXPERIENCE_PATTERNS = [
    r"(\d+)\s*\+?\s*years?",
    r"more than\s+(\d+)\s*years?",
    r"at least\s+(\d+)\s*years?",
    r"over\s+(\d+)\s*years?",
    r"(\d+)\s*-\s*(\d+)\s*years?",
    r"fresher|entry.?level|junior|mid.?level|senior|lead|principal"
]

STRUCTURED_TRIGGER_WORDS = [
    "show", "list", "find", "get", "give", "filter", "who", "in", "at",
    "from", "with experience", "having", "located", "based"
]

SEMANTIC_TRIGGER_WORDS = [
    "expert", "skilled", "good at", "knows", "experienced in", "proficient",
    "specializes", "background in", "familiar with", "worked on",
    "knowledge of", "capability", "best", "recommend", "suggest"
]

# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_query(query: str) -> Dict[str, Any]:
    """
    Parse a natural language query and extract:
    - query_type (QueryType enum)
    - role filter
    - location filter
    - experience filter
    - raw skills (for semantic)
    - original query
    """
    q = query.lower().strip()

    result: Dict[str, Any] = {
        "original_query": query,
        "query_type": QueryType.HYBRID,
        "role": None,
        "location": None,
        "experience_min": None,
        "experience_max": None,
        "experience_level": None,
        "skills_text": query,  # full query used for semantic
    }

    # ── Extract role ─────────────────────────────────────────────────────────
    for role in ROLE_KEYWORDS:
        if role in q:
            result["role"] = role.title()
            break

    # spaCy NER for better role extraction
    if SPACY_AVAILABLE and nlp:
        doc = nlp(query)
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                pass  # not useful here
            if ent.label_ == "PERSON":
                pass  # names, skip

    # ── Extract location ─────────────────────────────────────────────────────
    for loc in LOCATION_KEYWORDS:
        if loc in q:
            result["location"] = loc.title()
            break

    # ── Extract experience ───────────────────────────────────────────────────
    # Numeric range: "3-5 years"
    range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*years?", q)
    if range_match:
        result["experience_min"] = int(range_match.group(1))
        result["experience_max"] = int(range_match.group(2))

    # Single number: "5+ years", "at least 3 years"
    elif not range_match:
        num_match = re.search(r"(?:more than|at least|over|minimum)?\s*(\d+)\s*\+?\s*years?", q)
        if num_match:
            result["experience_min"] = int(num_match.group(1))

    # Level keywords
    level_match = re.search(r"\b(fresher|entry.?level|junior|mid.?level|senior|lead|principal)\b", q)
    if level_match:
        result["experience_level"] = level_match.group(1)

    # ── Determine query type ─────────────────────────────────────────────────
    has_structured_signal = (
        result["role"] is not None or
        result["location"] is not None or
        result["experience_min"] is not None or
        result["experience_level"] is not None or
        any(word in q for word in STRUCTURED_TRIGGER_WORDS)
    )

    has_semantic_signal = any(word in q for word in SEMANTIC_TRIGGER_WORDS)

    if has_structured_signal and has_semantic_signal:
        result["query_type"] = QueryType.HYBRID
    elif has_structured_signal:
        result["query_type"] = QueryType.STRUCTURED
    elif has_semantic_signal:
        result["query_type"] = QueryType.SEMANTIC
    else:
        # Default: if query is short and keyword-like → structured
        # If query is longer / descriptive → semantic
        word_count = len(q.split())
        result["query_type"] = QueryType.SEMANTIC if word_count > 5 else QueryType.STRUCTURED

    return result
