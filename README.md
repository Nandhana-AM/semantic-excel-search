# Excel Intelligent Search API

An intelligent, domain-agnostic FastAPI service that accepts any user-uploaded Excel directory/dataset and allows natural language querying against it using **Structured Filtering**, **Semantic Search**, or **Hybrid Search**.

---

## 🌟 Key Features

* **Domain-Agnostic Query Parsing**: Dynamically extracts roles, locations, names, and individual skills from the uploaded Excel sheet to build a vocabulary context. Queries are parsed case-insensitively using regex word-boundary boundaries and singular/plural variations.
* **Exact & Fuzzy Skill Splitting**: Distinguishes between **known skills** (existing in the dataset or skill dictionary) and **dynamic skills** (out-of-dictionary skills like "mud work" or "biotechnology" parsed via NLP triggers).
* **Strict Constraint Matching**: Employs hard filters for explicit role, location, name, and known skill parameters.
* **False-Positive Prevention**: Dynamic skills are validated against the candidate's core profile (`Role | Skills`) using sentence embeddings. Profiles with zero semantic overlap (similarity score `< 0.12`) are automatically filtered out.
* **Hybrid Search Re-Ranking**: Combines the precision of structured filtering with the contextual awareness of semantic search.

---

## 🏗️ Architecture & Component Flow

```
                     ┌──────────────────────────┐
                     │ User Uploads Excel file  │
                     │  + Natural Language Query│
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │   FastAPI /search POST   │
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │  Schema & Column Check   │  ← Name, Role, Location, Experience, Skills
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │ Dataset Vocab Extraction │  ← Extracts unique roles, locations, names,
                     └─────────────┬────────────┘    and individual split skills
                                   │
                     ┌─────────────▼────────────┐
                     │    Query Parser Engine   │  ← Regex, spaCy NER, known/dynamic skills split
                     └─────────────┬────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ STRUCTURED Mode  │     │  SEMANTIC Mode   │     │   HYBRID Mode    │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ - Filters by name│     │ - Clean & expand │     │ - Structured hard│
│ - Exact role match│    │   query          │     │   filters first  │
│ - Location match │     │ - Embeddings via │     │ - Semantic re-rank│
│ - Experience filtering │   SentenceTransformers││   on subset      │
│   (range/seniority)│   │ - Cosine sim     │     │ - Dynamic skill  │
│ - Known skills   │     │   scoring        │     │   check (>= 0.12)│
└─────────┬────────┘     └─────────┬────────┘     └─────────┬────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │  Deduplicate & Format    │
                     └─────────────┬────────────┘
                                   │
                     ┌─────────────▼────────────┐
                     │   JSON Response Payload  │
                     └──────────────────────────┘
```

---

## 🚦 Intent Flows & Routing Logic

When `mode` is set to `auto` (default), the API automatically routes queries to the most appropriate search mode based on extracted components:

### 1. Structured Intent
* **Trigger**: The query contains structural fields (role, location, name, or experience criteria) but **no skills/expertise** triggers.
* **Example**: `"show senior developers in Bangalore with 5+ years experience"`
* **Flow**:
  1. The dataset is filtered using Pandas based on the parsed constraints.
  2. Experience levels (e.g., "senior", "freshers") are mapped to numeric ranges.
  3. No semantic embeddings are generated, optimizing processing speed.

### 2. Semantic Intent
* **Trigger**: The query asks about skills, competencies, or fuzzy topics without specifying structural attributes, or focuses purely on skills.
* **Example**: `"who is expert in tunneling and deep excavation"`
* **Flow**:
  1. conversational wrappers (e.g. "who is expert in") are stripped to focus on core semantic terms.
  2. Text embeddings are generated for both the query and the candidate rows (Role + Location + Experience + Skills).
  3. Similarity is measured using cosine similarity (or FAISS indexing). Profiles scoring above `0.33` are returned, sorted by relevance.

### 3. Hybrid Intent
* **Trigger**: The query mixes structural constraints with specific skills (known or dynamic).
* **Example**: `"civil engineers in Chennai who have built bridges"`
* **Flow**:
  1. **Phase A (Structured Filter)**: Strict constraints (Role = `"Civil Engineer"`, Location = `"Chennai"`, plus any exact matching Known Skills like `"Bridges"`) are applied as hard filters. This reduces the search space.
  2. **Phase B (Semantic Ranking)**: Any dynamic skills in the query are encoded. The semantic model scores the filtered subset of candidates.
  3. **Phase C (Semantic Guard)**: For dynamic/out-of-dictionary skills (e.g. `"mud work"`), we batch-encode the candidate's `"Role | Skills"` and ensure its similarity score is `>= 0.12` to prevent irrelevant matches. Candidates matching the structured criteria but scoring poorly on the skill are excluded.

---

## 🛠️ Step-by-Step Installation & Run Guide

### Prerequisites
* Python 3.10+
* pip

### 1. Setup Virtual Environment & Install Dependencies

```bash
# Clone the repository (or enter project folder)
cd excel_search_api-fin

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download the NLP SpaCy Model
The project uses spaCy's English model for Entity Recognition (NER) to parse names and location filters:
```bash
python -m spacy download en_core_web_sm
```

### 3. Generate Sample Excel Data
You can generate a dummy workforce Excel sheet containing various roles, skills, and experience structures to test immediately:
```bash
python generate_sample_excel.py
# Creates: sample_employees.xlsx
```

### 4. Run the API Locally
Start the FastAPI server using Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, the API is available at `http://localhost:8000`. You can visit Swagger Interactive Docs at `http://localhost:8000/docs`.

### 5. Running the Test Suite
The project contains unit and integration tests. Run them using:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

The application is containerized and can be launched instantly using Docker Compose:

```bash
# Build and run the container
docker-compose up --build
```
The server will bind to port `8000` on the host machine.

---

## 📬 API Specifications

### `GET /health`
Validates that the server is up and responsive.
* **Response**:
  ```json
  {"status": "ok"}
  ```

### `POST /validate`
Uploads and validates an Excel file schema.
* **Form Parameters**:
  * `file`: Excel Binary file (`.xlsx`)
* **Response**:
  ```json
  {
    "valid": true,
    "rows": 16,
    "columns": ["Name", "Role", "Location", "Experience", "Skills"],
    "message": "Schema is valid. Ready for search."
  }
  ```

### `POST /search`
Main endpoint for searching candidates using natural language.
* **Form Parameters**:
  * `file`: Excel file (`.xlsx`)
  * `query`: `"show electricians with knowledge about about current"`
  * `mode`: `"auto"` (options: `"auto"`, `"structured"`, `"semantic"`, `"hybrid"`)
* **Response Sample**:
  ```json
  {
    "query": "show electricians with knowledge about about current",
    "mode": "hybrid",
    "filters_applied": {
      "role": "Electrician",
      "skills": ["Current"]
    },
    "total_in_file": 16,
    "results_count": 1,
    "results": [
      {
        "Name": "Lewis Curry",
        "Role": "Electrician",
        "Location": "Chennai",
        "Experience": "1 year",
        "Skills": "Electrical, Lighting",
        "_similarity_score": 0.5074
      }
    ]
  }
  ```
