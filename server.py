from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pickle
import re
import os
import numpy as np
from typing import List, Dict
from fastapi import Form
from fastapi import Query

app = FastAPI()

# Global variables for model, tfidf vectorizer, and encoder
__model = None
__tfidf = None
__encoder = None

# Predefined skill sets for different job domains
skillsets = {
    "SAP Developer": [
        "SAP ABAP", "SAP Fiori", "SAP HANA", "SAP S/4HANA", "SAP BASIS",
        "SAP BW", "SAP MM", "SAP SD", "OData Services", "Web Dynpro", "UI5",
        "BAPI", "ALE/IDocs", "SAP GRC", "AWS", "Azure", "Docker"
    ],
    "Network Security Engineer": [
        "Firewalls (Cisco, Palo Alto, Fortinet)", "VPN", "IDS/IPS", "SIEM (Splunk, ELK)",
        "Penetration Testing", "Ethical Hacking", "SOC Operations", "Cloud Security",
        "Threat Hunting", "Zero Trust Security", "AWS", "Azure", "Docker"
    ],
    "Blockchain Developer": [
        "Solidity", "Ethereum", "Smart Contracts", "Hyperledger Fabric", "Corda",
        "Cryptography", "Consensus Mechanisms", "DeFi", "NFT Standards", "Web3.js",
        "IPFS", "Truffle", "Metamask", "AWS", "Azure", "Docker"
    ],
    "Database Developer": [
        "SQL", "NoSQL", "PostgreSQL", "Oracle", "MongoDB", "Data Warehousing",
        "ETL Pipelines", "Query Optimization", "Database Replication",
        "Sharding", "Partitioning", "AWS", "Azure", "Docker"
    ],
    "Web Designing": [
        "HTML", "CSS", "JavaScript", "UI/UX Design", "Figma", "Adobe XD",
        "React.js", "Vue.js", "Tailwind CSS", "Bootstrap", "SEO Optimization",
        "A/B Testing", "Motion UI", "GSAP", "AWS", "Azure", "Docker"
    ],
    "DevOps Engineer": [
        "CI/CD", "Jenkins", "GitHub Actions", "Terraform", "Kubernetes",
        "Docker", "Prometheus", "Grafana", "Helm", "Cloud Security",
        "Chaos Engineering", "AWS", "Azure", "Docker"
    ],
    "Health and Fitness": [
        "Personal Training", "Nutrition", "Exercise Science", "Yoga",
        "Strength Training", "Physical Therapy", "Sports Medicine",
        "Health Coaching", "Wellness Programs"
    ],
    "Python Developer": [
        "Python", "Flask", "Django", "FastAPI", "Pandas", "NumPy",
        "AsyncIO", "API Development", "GraphQL", "Web Scraping",
        "Machine Learning", "AWS", "Azure", "Docker"
    ],
    "Advocate": [
        "Legal Research", "Case Analysis", "Contract Drafting", "Litigation",
        "Corporate Law", "Criminal Law", "Intellectual Property",
        "Negotiation", "Dispute Resolution"
    ],
    "Java Developer": [
        "Core Java", "Spring Boot", "Spring Security", "Hibernate",
        "Microservices Architecture", "REST APIs", "Concurrency",
        "Multithreading", "Docker", "AWS", "Azure"
    ],
    "PMO (Project Management Officer)": [
        "Project Planning", "Agile Methodologies", "Scrum",
        "Risk Management", "Stakeholder Management",
        "JIRA", "Resource Allocation", "AWS", "Azure", "Docker"
    ],
    "Mechanical Engineer": [
        "CAD (AutoCAD, SolidWorks)", "Thermodynamics", "Fluid Mechanics",
        "Material Science", "Finite Element Analysis", "HVAC", "Manufacturing",
        "Quality Control", "AWS", "Azure", "Docker"
    ],
    "Business Analyst": [
        "Requirement Analysis", "Process Mapping", "Business Strategy",
        "Market Research", "Data Analysis", "SQL", "Agile Methodologies",
        "JIRA", "AWS", "Azure", "Docker"
    ],
    "Data Science": [
        "Machine Learning", "Deep Learning", "Data Visualization",
        "Big Data", "Computer Vision", "NLP", "Time Series Forecasting",
        "AutoML", "MLOps", "AWS", "Azure", "Docker"
    ],
    "Sales": [
        "Lead Generation", "Cold Calling", "CRM Tools", "Negotiation",
        "Sales Forecasting", "Market Analysis", "Customer Relationship",
        "Presentation Skills", "AWS", "Azure", "Docker"
    ],
    "Arts": [
        "Painting", "Sculpting", "Illustration", "Photography",
        "Digital Art", "Graphic Design", "Art History",
        "Creative Direction", "Adobe Suite"
    ],
    "ETL Developer": [
        "ETL Pipelines", "Apache Airflow", "Data Warehousing",
        "SQL", "Spark Streaming", "Kafka", "AWS Glue", "AWS", "Azure", "Docker"
    ],
    "DotNet Developer": [
        "C#", "ASP.NET Core", "Entity Framework", "Microservices Architecture",
        "Blazor", "SignalR", "Azure Functions", "GraphQL",
        "AWS", "Azure", "Docker"
    ],
    "Hadoop Developer": [
        "HDFS", "YARN", "MapReduce", "Hive", "Pig", "Spark",
        "Cassandra", "Kafka", "AWS EMR", "AWS", "Azure", "Docker"
    ],
    "Electrical Engineer": [
        "Circuit Design", "PCB Layout", "MATLAB", "Simulink",
        "Power Systems", "Embedded Systems", "IoT Devices",
        "Wireless Communication", "AWS", "Azure", "Docker"
    ],
    "Automation Testing": [
        "Selenium", "Cypress", "Playwright", "JUnit", "TestNG",
        "API Testing", "Postman", "RestAssured", "AWS", "Azure", "Docker"
    ],
    "Operations Manager": [
        "Process Optimization", "Lean Six Sigma", "Supply Chain Management",
        "Strategic Planning", "Risk Assessment", "ERP Systems",
        "Financial Planning", "AWS", "Azure", "Docker"
    ],
    "HR (Human Resources)": [
        "Recruitment", "Employee Relations", "Talent Acquisition",
        "Payroll Management", "HR Analytics", "Performance Management",
        "Onboarding", "AWS", "Azure", "Docker"
    ],
    "Testing": [
        "Manual Testing", "Automated Testing", "Regression Testing",
        "Performance Testing", "Load Testing", "Security Testing",
        "Test Case Management", "AWS", "Azure", "Docker"
    ],
    "Civil Engineer": [
        "Structural Engineering", "Construction Management",
        "AutoCAD", "Revit", "Building Codes", "Geotechnical Engineering",
        "Surveying", "AWS", "Azure", "Docker"
    ]
}


# Load model, TF-IDF vectorizer, and label encoder
def load_model():
    global __model, __tfidf, __encoder
    model_file = "./model.pkl"
    tfidf_file = "./tfidf.pkl"
    encoder_file = "./encoder.pkl"

    with open(model_file, 'rb') as f:
        __model = pickle.load(f)

    with open(tfidf_file, 'rb') as f:
        __tfidf = pickle.load(f)

    with open(encoder_file, 'rb') as f:
        __encoder = pickle.load(f)

# Load the model at startup
load_model()

# Function to clean text (removes URLs, special characters, etc.)
def clean_text(txt: str) -> str:
    clean_txt = re.sub(r"http\S+", "", txt)  # Remove URLs
    clean_txt = re.sub(r"[^\w\s]", " ", clean_txt)  # Remove special characters
    clean_txt = re.sub(r"\s+", " ", clean_txt).strip()  # Remove extra spaces
    return clean_txt.lower()

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_file.file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    return text

# Predict the decision score for a given text and domain
def decision_score(text: str, domain: str) -> float:
    cleaned_text = clean_text(text)
    vectorized_text = __tfidf.transform([cleaned_text]).toarray()

    decision_scores = __model.decision_function(vectorized_text)

    if domain not in __encoder.classes_:
        raise ValueError(f"Domain '{domain}' not found in trained classes: {__encoder.classes_}")

    domain_index = list(__encoder.classes_).index(domain)

    return round(decision_scores[0][domain_index], 3)

# Classify the domain of a given text
def domain_classifier(text: str) -> Dict[str, float]:
    cleaned_text = clean_text(text)
    vectorized_text = __tfidf.transform([cleaned_text]).toarray()

    domain = __model.predict(vectorized_text)[0]  # Extract the predicted domain
    decision_scores = __model.decision_function(vectorized_text)[0].tolist()

    return {"domain": domain, "decision_scores": decision_scores}

# Calculate skill match score using Jaccard similarity
def skills_match_score(text: str, domain: str) -> float:
    found_skills = {skill for skill in skillsets.get(domain, []) if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE)}

    intersection = len(found_skills)
    union = len(skillsets.get(domain, []))

    if union == 0:
        return 0

    jaccard_score = intersection / union
    alpha = 5
    exp_score = 1 - np.exp(-alpha * jaccard_score)

    return round(exp_score, 3)

# Endpoint to classify resumes based on domain
@app.post("/classify/")
async def classify_resumes(files: List[UploadFile] = File(...)):
    results = {}

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        text = extract_text_from_pdf(file)
        clean_txt = clean_text(text)
        classification = domain_classifier(clean_txt)

        results[file.filename] = classification

    return {"classified_domains": results}

# Endpoint to rank resumes based on skills and decision score
@app.post("/rank/")
async def rank_resumes(
    files: List[UploadFile] = File(...),
    target_domain: str = Query(
        ...,
        title="Target Domain",
        description="Select one of the available job domains",
        enum=[
            "SAP Developer", "Network Security Engineer", "Blockchain Developer", "Database Developer",
            "Web Designing", "DevOps Engineer", "Health and Fitness", "Python Developer", "Advocate",
            "Java Developer", "PMO (Project Management Officer)", "Mechanical Engineer", "Business Analyst",
            "Data Science", "Sales", "Arts", "ETL Developer", "DotNet Developer", "Hadoop Developer",
            "Electrical Engineer", "Automation Testing", "Operations Manager", "HR (Human Resources)",
            "Testing", "Civil Engineer"
        ]
    )
):
    results = {}
    w1, w2 = 0.7, 0.3

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        text = extract_text_from_pdf(file)
        clean_txt = clean_text(text)

        score_skills = skills_match_score(clean_txt, target_domain)
        score_text = decision_score(clean_txt, target_domain)

        score = w1 * score_skills + w2 * score_text
        results[file.filename] = score

    # Sort resumes by descending score
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    return {"Ranked Scores": sorted_results}


# Home endpoint to check if API is running
@app.get("/")
def home():
    return {"message": "Resume Domain Classifier API is running!"}
