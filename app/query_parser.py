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
    "civil engineer", "software engineer", "mechanical engineer", "electrical engineer",
    "structural engineer", "backend developer", "frontend developer", "fullstack developer",
    "data scientist", "ml engineer","ai engineer","artificial intelligence engineer","deep learning engineer", "devops engineer", "qa engineer", "data engineer",
    "ui/ux designer", "ui designer", "ux designer", "project manager", "cloud architect",
    "engineer", "developer", "manager", "analyst", "architect", "designer",
    "consultant", "lead", "director", "officer", "intern", "specialist",
    "tester", "qa", "devops", "civil",
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
    r"freshers?|entry.?levels?|juniors?|mid.?levels?|seniors?|leads?|principals?"
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

SKILL_KEYWORDS = [
    "autocad", "python", "java", "sql", "react", "docker", "kubernetes",
    "aws", "azure", "machine learning", "deep learning", "nlp", "tunneling",
    "artificial intelligence", "ai",
    "structural design", "project management", "concrete", "bridges",
    "site supervision", "surveying", "revit", "c++", "c#", "javascript",
    "typescript", "node.js", "django", "fastapi", "flask", "excel", "word",
    "deep foundation", "foundation design", "staad pro", "steel structures", "rcc",
    "neural networks", "neural network", "keras", "tensorflow", "pytorch",
    "huggingface", "transformers", "spark", "kafka", "hadoop", "airflow",
    "git", "selenium", "pytest", "postman", "jira", "figma", "adobe xd",
    "user research", "prototyping", "html", "css", "rest apis", "mysql",
    "spring boot", "electrical", "lighting"
]


def _extract_dynamic_skills(clean_q: str, existing_skills: list) -> list:
    """Extract skills dynamically from the query when they follow pattern triggers and are not in SKILL_KEYWORDS."""
    triggers = [
        r"\bknows?\b",
        r"\bskilled\s+in\b",
        r"\bexpert\s+in\b",
        r"\bknowledge\s+of\b",
        r"\bfamiliar\s+with\b",
        r"\bworked\s+on\b",
        r"\bexperienced\s+in\b",
        r"\bexperience\s+in\b",
        r"\bexperience\s+with\b",
        r"\bacquantaince\s+to\b",
        r"\bacquaintance\s+to\b",
        r"\bacquantaince\s+with\b",
        r"\bacquaintance\s+with\b",
        r"\bacquainted\s+with\b",
        r"\bacquantaince\b",
        r"\bacquaintance\b",
        r"\bexpert\b",
        r"\bhaving\b",
        r"\bwith\b"
    ]
    
    dynamic_skills = []
    
    # Remove roles and locations to prevent false positives
    for role in sorted(ROLE_KEYWORDS, key=len, reverse=True):
        clean_q = re.sub(rf"\b{re.escape(role)}\b", " ", clean_q)
    for loc in sorted(LOCATION_KEYWORDS, key=len, reverse=True):
        clean_q = re.sub(rf"\b{re.escape(loc)}\b", " ", clean_q)
        
    # Remove experience descriptors
    clean_q = re.sub(r"\b\d+\s*\+?\s*years?\b", " ", clean_q)
    clean_q = re.sub(r"\b\d+\s*-\s*\d+\s*years?\b", " ", clean_q)
    clean_q = re.sub(r"\b(freshers?|entry.?levels?|juniors?|mid.?levels?|seniors?|leads?|principals?)\b", " ", clean_q)
    
    # Remove filler words
    fillers = [
        r"\bfind\b", r"\bfetch\b", r"\bshow\b", r"\blist\b", r"\bget\b", 
        r"\bgive\b", r"\bwho\b", r"\bpeople\b", r"\bperson\b", r"\bsomeone\b", r"\banyone\b"
    ]
    for filler in fillers:
        clean_q = re.sub(filler, " ", clean_q)
        
    clean_q = re.sub(r"\s+", " ", clean_q).strip()

    # Match segments after triggers
    for trigger in triggers:
        match = re.search(rf"{trigger}\s+(.+)", clean_q)
        if match:
            segment = match.group(1).strip()
            parts = re.split(r"\band\b|\bor\b|,|&", segment)
            for part in parts:
                part = part.strip()
                if SPACY_AVAILABLE and nlp:
                    part_doc = nlp(part)
                    valid_tokens = []
                    for token in part_doc:
                        if token.is_space or token.is_punct or token.is_stop:
                            continue
                        if token.lemma_.lower() in (
                            "experience", "skills", "skill", "expert", "expertise", "work", "knowledge",
                            "resource", "candidate", "understanding", "proficiency", "proficient",
                            "strong", "good", "basic", "advanced", "intermediate", "excellent",
                            "acquaintance", "acquantaince", "acquainted", "familiarity", "familiar"
                        ):
                            continue
                        if token.pos_ in ("NOUN", "PROPN", "ADJ", "NUM", "SYM", "X"):
                            valid_tokens.append(token.text)
                    cleaned_part = " ".join(valid_tokens).strip()
                else:
                    cleaned_part = re.sub(r"\b(experience|skills?|expert|expertise|work|knowledge|resource|candidate|who|is|an|a|the|some|any|basic|advanced|intermediate|excellent|acquaintance|acquantaince|acquainted|familiarity|familiar|understanding|proficiency|proficient|strong|good|have|has|had|having|know|knows|knowing|to|with|of|in|on|at|about|for)\b", "", part).strip()
                    cleaned_part = re.sub(r"\s+", " ", cleaned_part).strip()
                
                if cleaned_part and len(cleaned_part) > 1:
                    title_part = cleaned_part.title()
                    if title_part not in existing_skills and title_part not in dynamic_skills:
                        dynamic_skills.append(title_part)
            break
            
    return dynamic_skills


# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_query(
    query: str,
    dataset_roles: Optional[list] = None,
    dataset_locations: Optional[list] = None,
    dataset_names: Optional[list] = None,
    dataset_skills: Optional[list] = None
) -> Dict[str, Any]:
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
        "name": None,
        "role": None,
        "location": None,
        "experience_min": None,
        "experience_max": None,
        "experience_level": None,
        "skills_text": query,  # full query used for semantic
        "skills": [],
        "known_skills": [],
        "dynamic_skills": []
    }

    doc = None
    if SPACY_AVAILABLE and nlp:
        doc = nlp(query)

    # ── Extract role ─────────────────────────────────────────────────────────
    # Combine static ROLE_KEYWORDS and dataset_roles
    roles_to_check = set(ROLE_KEYWORDS)
    if dataset_roles:
        for r in dataset_roles:
            if isinstance(r, str):
                roles_to_check.add(r.lower().strip())
                
    for role in sorted(roles_to_check, key=len, reverse=True):
        pattern = rf"\b{re.escape(role)}(?:s|es)?\b"
        if re.search(pattern, q):
            # Normalize to the casing from the dataset if present
            matched_role = role.title()
            if dataset_roles:
                for orig_r in dataset_roles:
                    if orig_r.lower().strip() == role:
                        matched_role = orig_r
                        break
            result["role"] = matched_role
            break

    # ── Extract location ─────────────────────────────────────────────────────
    locations_to_check = set(LOCATION_KEYWORDS)
    if dataset_locations:
        for l in dataset_locations:
            if isinstance(l, str):
                locations_to_check.add(l.lower().strip())

    for loc in sorted(locations_to_check, key=len, reverse=True):
        pattern = rf"\b{re.escape(loc)}(?:s)?\b"
        if re.search(pattern, q):
            # Normalize to the casing from the dataset if present
            matched_loc = loc.title()
            if dataset_locations:
                for orig_l in dataset_locations:
                    if orig_l.lower().strip() == loc:
                        matched_loc = orig_l
                        break
            result["location"] = matched_loc
            break

    # If location not matched by keywords, use spaCy GPE/LOC
    if not result["location"] and doc:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                result["location"] = ent.text.title()
                break

    # spaCy NER for name extraction
    if doc:
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                pass  # not useful here
            if ent.label_ == "PERSON":
                ent_lower = ent.text.lower().strip()
                # Do not treat known skills, locations, or roles as names
                is_known_keyword = (
                    ent_lower in [s.lower() for s in SKILL_KEYWORDS] or
                    ent_lower in [l.lower() for l in locations_to_check] or
                    ent_lower in [r.lower() for r in roles_to_check]
                )
                is_extracted_filter = (
                    (result["role"] and ent_lower == result["role"].lower()) or
                    (result["location"] and ent_lower == result["location"].lower()) or
                    (result["skills"] and any(ent_lower == s.lower() for s in result["skills"]))
                )
                if not is_known_keyword and not is_extracted_filter:
                    result["name"] = ent.text.title()

    # Dynamic fallback check against dataset_names
    if not result["name"] and dataset_names:
        for name in sorted(dataset_names, key=len, reverse=True):
            if isinstance(name, str):
                name_lower = name.lower().strip()
                pattern = rf"\b{re.escape(name_lower)}\b"
                if re.search(pattern, q):
                    result["name"] = name
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
    level_match = re.search(r"\b(freshers?|entry.?levels?|juniors?|mid.?levels?|seniors?|leads?|principals?)\b", q)
    if level_match:
        result["experience_level"] = level_match.group(1)

    # ── Extract skills ───────────────────────────────────────────────────────
    # Combine static SKILL_KEYWORDS and dataset_skills
    skills_to_check = set(SKILL_KEYWORDS)
    if dataset_skills:
        for s in dataset_skills:
            if isinstance(s, str):
                # A dataset skill might contain comma-separated entries, so split it
                for part in re.split(r",|;|and", s):
                    part_clean = part.strip().lower()
                    if part_clean:
                        skills_to_check.add(part_clean)

    q_temp = q
    known_skills = []
    for skill in sorted(skills_to_check, key=len, reverse=True):
        start_boundary = r"\b" if skill[0].isalnum() or skill[0] == "_" else ""
        end_boundary = r"\b" if skill[-1].isalnum() or skill[-1] == "_" else ""
        pattern = f"{start_boundary}{re.escape(skill)}{end_boundary}"
        matches = list(re.finditer(pattern, q_temp))
        if matches:
            # Match case from dataset or capitalize
            matched_skill = skill.title()
            if dataset_skills:
                found_orig = False
                for s in dataset_skills:
                    if isinstance(s, str):
                        for part in re.split(r",|;|and", s):
                            if part.strip().lower() == skill:
                                matched_skill = part.strip()
                                found_orig = True
                                break
                        if found_orig:
                            break
            known_skills.append(matched_skill)
            # Replace matched portion with spaces to prevent double matching shorter substrings
            for match in matches:
                start, end = match.span()
                q_temp = q_temp[:start] + " " * (end - start) + q_temp[end:]

    # Extract dynamic/out-of-dictionary skills
    dyn_skills = _extract_dynamic_skills(q_temp, known_skills)
    
    result["skills"] = list(known_skills) + list(dyn_skills)
    result["known_skills"] = list(known_skills)
    result["dynamic_skills"] = list(dyn_skills)

    # Helper to check for whole words (using word boundaries) or phrases
    def _has_word(trigger: str, text: str) -> bool:
        if " " in trigger:
            return trigger in text
        return bool(re.search(rf"\b{re.escape(trigger)}\b", text))

    # ── Determine query type ─────────────────────────────────────────────────
    has_structured_signal = (
        result["name"] is not None or
        result["role"] is not None or
        result["location"] is not None or
        result["experience_min"] is not None or
        result["experience_level"] is not None or
        any(_has_word(word, q) for word in STRUCTURED_TRIGGER_WORDS)
    )

    has_semantic_signal = (
        any(_has_word(word, q) for word in SEMANTIC_TRIGGER_WORDS) or
        len(result["skills"]) > 0
    )

    has_actual_constraints = (
        result["name"] is not None or
        result["role"] is not None or
        result["location"] is not None or
        result["experience_min"] is not None or
        result["experience_level"] is not None
    )

    if has_structured_signal and has_semantic_signal:
        result["query_type"] = QueryType.HYBRID if has_actual_constraints else QueryType.SEMANTIC
    elif has_structured_signal:
        result["query_type"] = QueryType.STRUCTURED if has_actual_constraints else QueryType.SEMANTIC
    elif has_semantic_signal:
        result["query_type"] = QueryType.SEMANTIC
    else:
        result["query_type"] = QueryType.SEMANTIC

    return result
