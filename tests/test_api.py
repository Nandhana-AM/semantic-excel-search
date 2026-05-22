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
        {"Name": "Grace",   "Role": "QA Engineer",       "Location": "Bangalore", "Experience": "1 year",   "Skills": "Selenium, Testing"},
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


def test_parse_query_experience_level_plural():
    result = parse_query("find freshers")
    assert result["experience_level"] == "freshers"


def test_parse_query_special_character_skills():
    result = parse_query("someone who knows c++ and c#")
    assert "C++" in result["skills"]
    assert "C#" in result["skills"]


def test_parse_query_no_duplicate_skills():
    result = parse_query("someone who knows neural networks")
    assert "Neural Networks" in result["skills"]
    assert "Neural Network" not in result["skills"]



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


def test_structured_search_by_experience_level():
    df = make_sample_df()
    parsed = {"role": None, "location": None, "experience_min": None, "experience_max": None, "experience_level": "freshers"}
    results = structured_search(df, parsed)
    assert len(results) == 1
    assert results[0]["Name"] == "Grace"



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


def test_query_parser_routing_pure_skills():
    parsed = parse_query("someone who knows machine learning and neural networks")
    assert parsed["query_type"] == QueryType.SEMANTIC


def test_query_parser_routing_hybrid():
    parsed = parse_query("senior software engineers in Bangalore who know Docker")
    assert parsed["query_type"] == QueryType.HYBRID


def test_ml_query_excludes_civil_engineers():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    
    # Query auto mode, which should resolve to semantic search
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "someone who knows machine learning and neural networks", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    
    # Check that civil engineers (Alice and Dave) are not returned
    returned_roles = [r["Role"] for r in body["results"]]
    assert "Civil Engineer" not in returned_roles
    
    # Should return Carol (Data Scientist) and Eve (ML Engineer)
    returned_names = [r["Name"] for r in body["results"]]
    assert "Carol" in returned_names
    assert "Eve" in returned_names


def test_autocad_query_excludes_ui_ux_designer():
    df = pd.DataFrame([
        {"Name": "Alice",   "Role": "Civil Engineer",    "Location": "Chennai",   "Experience": "5 years",  "Skills": "AutoCAD, Structural Design, Tunneling"},
        {"Name": "Nathan",  "Role": "Civil Engineer",    "Location": "Delhi",     "Experience": "3 years",  "Skills": "Site Supervision, AutoCAD, Surveying"},
        {"Name": "Dave",    "Role": "Civil Engineer",    "Location": "Chennai",   "Experience": "10 years", "Skills": "Project Management, Concrete, Bridges"},
        {"Name": "Olivia",  "Role": "UI/UX Designer",    "Location": "Mumbai",    "Experience": "4 years",  "Skills": "Figma, Adobe XD, User Research, Prototyping"},
    ])
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "find people skilled in AutoCAD and structural design", "mode": "semantic"}
    )
    assert response.status_code == 200
    body = response.json()
    
    returned_names = [r["Name"] for r in body["results"]]
    # Olivia (UI/UX Designer) should be excluded
    assert "Olivia" not in returned_names
    
    # Alice (AutoCAD, Structural Design) should be included
    assert "Alice" in returned_names


def test_dynamic_skill_extraction_out_of_dictionary():
    # 1. Test parsing
    parsed = parse_query("fetch people who know rhino")
    assert "Rhino" in parsed["skills"]
    
    # 2. Test search execution via API (should have empty results but show "Rhino" in filters)
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "fetch people who know rhino", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    
    # Assert result is empty because "Rhino" is not in sample data
    assert body["results_count"] == 0
    assert len(body["results"]) == 0
    
    # Assert filter was still extracted and applied
    assert "skills" in body["filters_applied"]
    assert "Rhino" in body["filters_applied"]["skills"]


def test_roads_query_returns_civil_engineers():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "show people who have worked related to roads", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    
    returned_names = [r["Name"] for r in body["results"]]
    # Should include Dave (Bridges skill) and Alice (Tunneling skill)
    assert "Dave" in returned_names
    assert "Alice" in returned_names


def test_huggingface_llm_query_returns_huggingface_candidates():
    df = pd.DataFrame([
        {"Name": "Eve", "Role": "ML Engineer", "Location": "Bangalore", "Experience": "4 years", "Skills": "PyTorch, NLP, Transformers, HuggingFace"},
        {"Name": "Bob", "Role": "Software Engineer", "Location": "Bangalore", "Experience": "3 years", "Skills": "Python, FastAPI, Docker"}
    ])
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "find people with HuggingFace and LLM fine-tuning experience", "mode": "semantic"}
    )
    assert response.status_code == 200
    body = response.json()
    
    returned_names = [r["Name"] for r in body["results"]]
    assert "Eve" in returned_names


def test_hybrid_search_strictly_respects_structured_filters():
    df = pd.DataFrame([
        {"Name": "Olivia George", "Role": "UI/UX Designer", "Location": "Mumbai", "Experience": "4 years", "Skills": "Figma, Adobe XD, User Research, Prototyping"},
        {"Name": "Grace Singh", "Role": "Frontend Developer", "Location": "Hyderabad", "Experience": "2 years", "Skills": "React, TypeScript, CSS, Figma"},
        {"Name": "James Pillai", "Role": "Project Manager", "Location": "Mumbai", "Experience": "12 years", "Skills": "PMP, Agile"},
        {"Name": "Bob Sharma", "Role": "Software Engineer", "Location": "Bangalore", "Experience": "3 years", "Skills": "Python, FastAPI"},
    ])
    
    parsed = parse_query("UI designers in Mumbai with Figma and user research experience")
    results = hybrid_search(df, "UI designers in Mumbai with Figma and user research experience", parsed)
    
    # Should only return Olivia George because she is in Mumbai and is a Designer
    assert len(results) == 1
    assert results[0]["Name"] == "Olivia George"


def test_structural_engineer_role_extraction_and_search():
    df = pd.DataFrame([
        {"Name": "Iris Nair", "Role": "Structural Engineer", "Location": "Chennai", "Experience": "8 years", "Skills": "STAAD Pro, Foundation Design, Steel Structures"},
        {"Name": "Alice Johnson", "Role": "Civil Engineer", "Location": "Chennai", "Experience": "5 years", "Skills": "AutoCAD, Structural Design, Tunneling"},
        {"Name": "Dave Rajan", "Role": "Civil Engineer", "Location": "Chennai", "Experience": "10 years", "Skills": "Project Management, Concrete, Bridges"},
    ])
    
    parsed = parse_query("structural engineers in Chennai skilled in STAAD Pro and steel design")
    assert parsed["role"] == "Structural Engineer"
    
    results = hybrid_search(df, "structural engineers in Chennai skilled in STAAD Pro and steel design", parsed)
    # Should only return Iris Nair because she is the only Structural Engineer (not Civil Engineer)
    assert len(results) == 1
    assert results[0]["Name"] == "Iris Nair"


def test_architect_with_acquaintance_query():
    parsed = parse_query("Need a resource who is an architect with 10 years and have acquantaince to AutoCAD and MS Office")
    assert parsed["role"] == "Architect"
    assert parsed["experience_min"] == 10
    # AutoCAD is in SKILL_KEYWORDS, MS Office is parsed dynamically
    assert "Autocad" in parsed["skills"]
    assert "Ms Office" in parsed["skills"]
    assert "Have Acquantaince To" not in parsed["skills"]
    assert len(parsed["skills"]) == 2


def test_acquaintance_spelling_variations_and_cleanups():
    parsed1 = parse_query("Need a resource who is an architect with 10 years and have acquaintance to AutoCAD and MS Office")
    assert "Autocad" in parsed1["skills"]
    assert "Ms Office" in parsed1["skills"]
    assert len(parsed1["skills"]) == 2

    parsed2 = parse_query("architect having acquaintance with rhino")
    assert "Rhino" in parsed2["skills"]
    assert "Having Acquaintance With" not in parsed2["skills"]
    assert len(parsed2["skills"]) == 1


# ─── spaCy Dynamic NLP Tests ──────────────────────────────────────────────────

def test_spacy_dynamic_location_extraction():
    parsed = parse_query("software engineers in Seattle")
    assert parsed["location"] == "Seattle"
    assert parsed["query_type"] == QueryType.STRUCTURED


def test_spacy_improved_dynamic_skill_extraction():
    parsed = parse_query("developers familiar with cloud orchestration")
    assert "Cloud Orchestration" in parsed["skills"]


def test_spacy_name_extraction():
    parsed = parse_query("is Alice a civil engineer?")
    assert parsed["name"] == "Alice"
    assert parsed["role"] == "Civil Engineer"
    assert parsed["query_type"] == QueryType.STRUCTURED


def test_search_by_name_via_api():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "Who is Bob?", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results_count"] == 1
    assert body["results"][0]["Name"] == "Bob"
    assert body["filters_applied"]["name"] == "Bob"


def test_dynamic_role_from_dataset_routing():
    # "Electrician" is not in ROLE_KEYWORDS, but should be resolved dynamically if passed as dataset_roles
    parsed = parse_query(
        "show electricians with knowledge about ai",
        dataset_roles=["Electrician"],
        dataset_skills=["AI"]
    )
    assert parsed["role"] == "Electrician"
    assert parsed["query_type"] == QueryType.HYBRID
    assert "AI" in parsed["known_skills"]


def test_strict_skills_filtering_bridges():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    
    # Bridges is a known skill. This should strictly return Dave.
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "civil engineer who has built bridges", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results_count"] == 1
    assert body["results"][0]["Name"] == "Dave"


def test_semantic_fallback_mud_work():
    df = make_sample_df()
    excel_bytes = df_to_excel_bytes(df)
    
    # Mud work is not a known skill. It should return all civil engineers sorted semantically.
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "civil engineers who have dealt with mud work", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    # Should only return Civil Engineers (Alice and Dave)
    returned_names = [r["Name"] for r in body["results"]]
    assert len(returned_names) == 2
    assert "Alice" in returned_names
    assert "Dave" in returned_names


def test_semantic_precision_electrician_biotechnology_excluded():
    df = pd.DataFrame([
        {"Name": "Lewis Curry", "Role": "Electrician", "Location": "Chennai", "Experience": "1 year", "Skills": "Electrical, Lighting"}
    ])
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "show electricians with knowledge about about biotechnology", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results_count"] == 0


def test_semantic_precision_electrician_current_included():
    df = pd.DataFrame([
        {"Name": "Lewis Curry", "Role": "Electrician", "Location": "Chennai", "Experience": "1 year", "Skills": "Electrical, Lighting"}
    ])
    excel_bytes = df_to_excel_bytes(df)
    
    response = client.post(
        "/search",
        files={"file": ("employees.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        data={"query": "show electricians with knowledge about about current", "mode": "auto"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results_count"] == 1
    assert body["results"][0]["Name"] == "Lewis Curry"



