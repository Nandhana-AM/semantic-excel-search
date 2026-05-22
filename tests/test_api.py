"""
tests/test_api.py
Test suite for Excel Intelligent Search API.
Run with: pytest tests/ -v
"""

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to path
import sys
sys.path.insert(0, ".")

from app.main import app
from app.validator import validate_schema, REQUIRED_COLUMNS
from app.loader import load_excel
from app.query_parser import parse_query, QueryType
from app.structured_search import structured_search
from app.semantic_search import semantic_search
from app.hybrid_search import hybrid_search

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_sample_df():
    return pd.DataFrame([
        {"Name": "Alice",   "Role": "Civil Engineer",    "Location": "Chennai",   "Experience": "5 years",  "Skills": "AutoCAD, Structural Design, Tunneling"},
        {"Name": "Bob",     "Role": "Software Engineer", "Location": "Bangalore", "Experience": "3 years",  "Skills": "Python, FastAPI, Docker"},
        {"Name": "Carol",   "Role": "Data Scientist",    "Location": "Mumbai",    "Experience": "7 years",  "Skills": "Machine Learning, Python, TensorFlow"},
        {"Name": "Dave",    "Role": "Civil Engineer",    "Location": "Chennai",   "Experience": "10 years", "Skills": "Project Management, Concrete, Bridges"},
        {"Name": "Eve",     "Role": "ML Engineer",       "Location": "Bangalore", "Experience": "4 years",  "Skills": "PyTorch, NLP, Transformers"},
        {"Name": "Frank",   "Role": "DevOps Engineer",   "Location": "Delhi",     "Experience": "6 years",  "Skills": "Kubernetes, Docker, CI/CD"},
    ])


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


# ─── Health Check ─────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─── Schema Validation ────────────────────────────────────────────────────────

def test_validate_schema_valid():
    df = make_sample_df()
    error = validate_schema(df)
    assert error is None


def test_validate_schema_missing_column():
    df = make_sample_df().drop(columns=["Skills"])
    error = validate_schema(df)
    assert error is not None
    assert "Skills" in error


def test_validate_schema_case_insensitive():
    df = make_sample_df()
    df.columns = [c.lower() for c in df.columns]  # lowercase all
    error = validate_schema(df)
    assert error is None  # should pass and normalize


def test_validate_schema_empty_df():
    df = pd.DataFrame()
    error = validate_schema(df)
    assert error is not None


# ─── Loader ───────────────────────────────────────────────────────────────────

def test_load_excel_valid():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    loaded_df, error = load_excel(excel_bytes)
    assert error is None
    assert loaded_df is not None
    assert len(loaded_df) == len(df)


def test_load_excel_invalid_bytes():
    _, error = load_excel(b"not an excel file")
    assert error is not None


# ─── Query Parser ─────────────────────────────────────────────────────────────

def test_parse_query_structured_role_location():
    result = parse_query("show civil engineers in Chennai")
    assert result["role"] is not None
    assert result["location"] is not None
    assert result["query_type"] == QueryType.STRUCTURED


def test_parse_query_experience_number():
    result = parse_query("find engineers with 5+ years experience")
    assert result["experience_min"] == 5


def test_parse_query_experience_range():
    result = parse_query("developers with 3-6 years experience")
    assert result["experience_min"] == 3
    assert result["experience_max"] == 6


def test_parse_query_semantic():
    result = parse_query("who is expert in tunneling and deep excavation")
    assert result["query_type"] in (QueryType.SEMANTIC, QueryType.HYBRID)


def test_parse_query_experience_level():
    result = parse_query("show senior engineers")
    assert result["experience_level"] == "senior"


# ─── Structured Search ────────────────────────────────────────────────────────

def test_structured_search_by_role():
    df = make_sample_df()
    parsed = parse_query("show civil engineers")
    results = structured_search(df, parsed)
    assert all("Civil" in r["Role"] for r in results)
    assert len(results) == 2


def test_structured_search_by_location():
    df = make_sample_df()
    parsed = parse_query("engineers in Bangalore")
    results = structured_search(df, parsed)
    assert all("Bangalore" in r["Location"] for r in results)


def test_structured_search_by_experience():
    df = make_sample_df()
    parsed = {"role": None, "location": None, "experience_min": 6, "experience_max": None, "experience_level": None}
    results = structured_search(df, parsed)
    assert len(results) >= 2  # Carol (7), Dave (10), Frank (6)


def test_structured_search_no_results():
    df = make_sample_df()
    parsed = {"role": "Astronaut", "location": None, "experience_min": None, "experience_max": None, "experience_level": None}
    results = structured_search(df, parsed)
    assert results == []


# ─── Semantic Search ──────────────────────────────────────────────────────────

def test_semantic_search_returns_results():
    df = make_sample_df()
    results = semantic_search(df, "machine learning and deep learning expert")
    assert isinstance(results, list)
    assert len(results) > 0


def test_semantic_search_empty_df():
    df = pd.DataFrame(columns=["Name", "Role", "Location", "Experience", "Skills"])
    results = semantic_search(df, "any query")
    assert results == []


def test_semantic_search_score_present():
    df = make_sample_df()
    results = semantic_search(df, "Python developer")
    for r in results:
        assert "_similarity_score" in r


# ─── Hybrid Search ────────────────────────────────────────────────────────────

def test_hybrid_search_returns_results():
    df = make_sample_df()
    parsed = parse_query("senior software engineers in Bangalore who know Docker")
    results = hybrid_search(df, "senior software engineers in Bangalore who know Docker", parsed)
    assert isinstance(results, list)


def test_hybrid_search_no_duplicates():
    df = make_sample_df()
    parsed = parse_query("engineers in Chennai with tunneling experience")
    results = hybrid_search(df, "engineers in Chennai with tunneling experience", parsed)
    names = [r["Name"] for r in results]
    assert len(names) == len(set(names))  # no duplicates


# ─── API Endpoints ────────────────────────────────────────────────────────────

def test_search_endpoint_valid():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "civil engineers in Chennai", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert "results_count" in body
    assert body["query"] == "civil engineers in Chennai"


def test_search_endpoint_invalid_file_type():
    response = client.post(
        "/search",
        files={"file": ("data.csv", b"Name,Role\nAlice,Engineer", "text/csv")},
        data={"query": "test"}
    )
    assert response.status_code == 400


def test_search_endpoint_missing_columns():
    df = pd.DataFrame([{"Name": "Alice", "Role": "Engineer"}])
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("bad.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "engineers"}
    )
    assert response.status_code == 422


def test_validate_endpoint_valid():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/validate",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_validate_endpoint_invalid_schema():
    df = make_sample_df().drop(columns=["Location", "Skills"])
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/validate",
        files={"file": ("bad.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    )
    assert response.status_code == 422
    assert response.json()["valid"] is False


def test_search_structured_mode():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "engineers", "mode": "structured"}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "structured"


def test_search_semantic_mode():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "someone who knows tunneling and concrete work", "mode": "semantic"}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "semantic"


def test_search_hybrid_mode():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "senior engineers in Bangalore with ML skills", "mode": "hybrid"}
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "hybrid"
