# Excel Intelligent Search API — Week 3

A FastAPI service that accepts a user-uploaded Excel file and searches it using **structured filtering**, **semantic search**, or **hybrid** mode.

---

## Architecture

```
User Uploads Excel + Query
         │
    FastAPI /search
         │
  ┌──────▼──────┐
  │  Validation  │  ← check required columns
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Query       │  ← spaCy + regex + rules
  │  Parser      │    detects role, location,
  └──────┬──────┘    experience, query type
         │
  ┌──────▼──────────────────────────────┐
  │           Decision Router            │
  │  STRUCTURED | SEMANTIC | HYBRID     │
  └──────┬──────────────┬───────────────┘
         │              │
  Pandas filters   Embeddings + FAISS
         │              │
         └──── Merge ───┘
                  │
          JSON Response
```

---

## Required Excel Columns

| Column     | Example              |
|------------|----------------------|
| Name       | Alice Johnson        |
| Role       | Civil Engineer       |
| Location   | Chennai              |
| Experience | 5 years              |
| Skills     | AutoCAD, Tunneling   |

Column names are case-insensitive.

---

## Setup

### Local

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker-compose up --build
```

API will be available at: http://localhost:8000

---

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /validate`
Upload an Excel file to validate its schema.

**Form data:**
- `file`: Excel file (.xlsx)

**Response:**
```json
{
  "valid": true,
  "rows": 15,
  "columns": ["Name", "Role", "Location", "Experience", "Skills"],
  "message": "Schema is valid. Ready for search."
}
```

### `POST /search`
Upload an Excel file and search it.

**Form data:**
- `file`: Excel file (.xlsx)
- `query`: Natural language query (string)
- `mode`: `auto` | `structured` | `semantic` | `hybrid` (default: `auto`)

**Response:**
```json
{
  "query": "show civil engineers in Chennai",
  "mode": "structured",
  "filters_applied": {
    "role": "Civil",
    "location": "Chennai"
  },
  "total_in_file": 15,
  "results_count": 2,
  "results": [
    {
      "Name": "Alice Johnson",
      "Role": "Civil Engineer",
      "Location": "Chennai",
      "Experience": "5 years",
      "Skills": "AutoCAD, Structural Design, Tunneling"
    }
  ]
}
```

---

## Search Modes

| Mode       | When Used                          | How it Works                         |
|------------|-------------------------------------|--------------------------------------|
| structured | Role, location, experience queries  | Pandas `.str.contains()` + filters   |
| semantic   | Skills, expertise, fuzzy queries    | Sentence embeddings + FAISS index    |
| hybrid     | Complex queries (both)              | Structured filter → semantic re-rank |
| auto       | Default                             | Parser decides based on query        |

---

## Query Examples

| Query                                              | Mode       |
|-----------------------------------------------------|------------|
| `show civil engineers in Chennai`                  | structured |
| `find engineers with 5+ years experience`          | structured |
| `developers with 3-6 years`                        | structured |
| `show senior engineers`                            | structured |
| `expert in tunneling and deep foundation work`     | semantic   |
| `someone who knows machine learning and Python`    | semantic   |
| `senior engineers in Bangalore who know Docker`    | hybrid     |

---

## Generate Sample Excel

```bash
python generate_sample_excel.py
# Creates: sample_employees.xlsx (15 rows)
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Interactive Docs

Once running, open:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Postman

Import `postman_collection.json` into Postman and set the `file` field to your Excel.
