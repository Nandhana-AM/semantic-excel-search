"""
generate_sample_excel.py
Generates a sample employees.xlsx for Postman / manual testing.
Run: python generate_sample_excel.py
"""

import pandas as pd

data = [
    {"Name": "Alice Johnson",   "Role": "Civil Engineer",        "Location": "Chennai",   "Experience": "5 years",  "Skills": "AutoCAD, Structural Design, Tunneling, RCC"},
    {"Name": "Bob Sharma",      "Role": "Software Engineer",     "Location": "Bangalore", "Experience": "3 years",  "Skills": "Python, FastAPI, Docker, PostgreSQL"},
    {"Name": "Carol Patel",     "Role": "Data Scientist",        "Location": "Mumbai",    "Experience": "7 years",  "Skills": "Machine Learning, Python, TensorFlow, Keras"},
    {"Name": "Dave Rajan",      "Role": "Civil Engineer",        "Location": "Chennai",   "Experience": "10 years", "Skills": "Project Management, Concrete, Bridges, AutoCAD"},
    {"Name": "Eve Kumar",       "Role": "ML Engineer",           "Location": "Bangalore", "Experience": "4 years",  "Skills": "PyTorch, NLP, Transformers, HuggingFace"},
    {"Name": "Frank Menon",     "Role": "DevOps Engineer",       "Location": "Delhi",     "Experience": "6 years",  "Skills": "Kubernetes, Docker, CI/CD, Terraform"},
    {"Name": "Grace Singh",     "Role": "Frontend Developer",    "Location": "Hyderabad", "Experience": "2 years",  "Skills": "React, TypeScript, CSS, Figma"},
    {"Name": "Henry Das",       "Role": "Backend Developer",     "Location": "Pune",      "Experience": "5 years",  "Skills": "Java, Spring Boot, MySQL, REST APIs"},
    {"Name": "Iris Nair",       "Role": "Structural Engineer",   "Location": "Chennai",   "Experience": "8 years",  "Skills": "STAAD Pro, Foundation Design, Steel Structures"},
    {"Name": "James Pillai",    "Role": "Project Manager",       "Location": "Mumbai",    "Experience": "12 years", "Skills": "PMP, Agile, Risk Management, MS Project"},
    {"Name": "Karen Thomas",    "Role": "QA Engineer",           "Location": "Bangalore", "Experience": "3 years",  "Skills": "Selenium, Pytest, Postman, JIRA"},
    {"Name": "Leo Verma",       "Role": "Cloud Architect",       "Location": "Bangalore", "Experience": "9 years",  "Skills": "AWS, Azure, GCP, Microservices, Terraform"},
    {"Name": "Maya Reddy",      "Role": "Data Engineer",         "Location": "Hyderabad", "Experience": "5 years",  "Skills": "Spark, Kafka, Hadoop, Python, Airflow"},
    {"Name": "Nathan Iyer",     "Role": "Civil Engineer",        "Location": "Delhi",     "Experience": "3 years",  "Skills": "Site Supervision, AutoCAD, Surveying"},
    {"Name": "Olivia George",   "Role": "UI/UX Designer",        "Location": "Mumbai",    "Experience": "4 years",  "Skills": "Figma, Adobe XD, User Research, Prototyping"},
]

df = pd.DataFrame(data)
df.to_excel("sample_employees.xlsx", index=False, engine="openpyxl")
print(f"Generated sample_employees.xlsx with {len(df)} rows.")
print(f"Columns: {list(df.columns)}")
