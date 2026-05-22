from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.validator import validate_schema
from app.loader import load_excel
from app.query_parser import parse_query, QueryType
from app.structured_search import structured_search
from app.semantic_search import semantic_search
from app.hybrid_search import hybrid_search
from app.response_formatter import format_response

app = FastAPI(
    title="Excel Intelligent Search API",
    description="Upload an Excel file and query it using structured, semantic, or hybrid search.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Excel Intelligent Search API is running.", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(
    file: UploadFile = File(..., description="Excel file (.xlsx) with required columns"),
    query: str = Form(..., description="Search query string"),
    mode: str = Form(default="auto", description="Search mode: auto | structured | semantic | hybrid")
):
    """
    Upload an Excel file and search using natural language.

    Required Excel columns: Name, Role, Location, Experience, Skills

    Modes:
    - auto     : system decides based on query type
    - structured: exact/filter-based search (role, location, experience)
    - semantic  : embedding-based fuzzy search (skills, expertise)
    - hybrid    : structured + semantic combined
    """

    # --- 1. Validate file type ---
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx or .xls files are supported.")

    # --- 2. Load Excel into DataFrame ---
    contents = await file.read()
    df, load_error = load_excel(contents)
    if load_error:
        raise HTTPException(status_code=400, detail=load_error)

    # --- 3. Validate schema ---
    schema_error = validate_schema(df)
    if schema_error:
        raise HTTPException(status_code=422, detail=schema_error)

    # --- 4. Parse query ---
    import re
    dataset_roles = [str(r).strip() for r in df["Role"].dropna().unique() if str(r).strip()] if "Role" in df.columns else []
    dataset_locations = [str(l).strip() for l in df["Location"].dropna().unique() if str(l).strip()] if "Location" in df.columns else []
    dataset_names = [str(n).strip() for n in df["Name"].dropna().unique() if str(n).strip()] if "Name" in df.columns else []
    
    dataset_skills = []
    if "Skills" in df.columns:
        for val in df["Skills"].dropna().unique():
            if isinstance(val, str):
                for part in re.split(r",|;|and", val):
                    part_clean = part.strip()
                    if part_clean and part_clean not in dataset_skills:
                        dataset_skills.append(part_clean)

    parsed = parse_query(
        query,
        dataset_roles=dataset_roles,
        dataset_locations=dataset_locations,
        dataset_names=dataset_names,
        dataset_skills=dataset_skills
    )

    # --- 5. Determine mode ---
    if mode == "auto":
        selected_mode = parsed["query_type"].value
    else:
        selected_mode = mode

    # --- 6. Route to search ---
    if selected_mode == "structured":
        results = structured_search(df, parsed)
    elif selected_mode == "semantic":
        results = semantic_search(df, query, parsed=parsed)
    else:  # hybrid
        results = hybrid_search(df, query, parsed)

    # --- 7. Format and return ---
    response = format_response(
        results=results,
        query=query,
        mode=selected_mode,
        parsed_filters=parsed,
        total_rows=len(df)
    )

    return JSONResponse(content=response)


@app.post("/validate")
async def validate_file(file: UploadFile = File(...)):
    """Validate if an uploaded Excel file has the correct schema."""
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx or .xls files are supported.")

    contents = await file.read()
    df, load_error = load_excel(contents)
    if load_error:
        raise HTTPException(status_code=400, detail=load_error)

    schema_error = validate_schema(df)
    if schema_error:
        return JSONResponse(status_code=422, content={"valid": False, "error": schema_error})

    return {
        "valid": True,
        "rows": len(df),
        "columns": list(df.columns),
        "message": "Schema is valid. Ready for search."
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
