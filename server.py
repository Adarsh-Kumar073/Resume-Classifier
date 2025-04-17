from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import pickle
import re
import numpy as np
from typing import List, Dict

app = FastAPI()

# Enable CORS for frontend access (Update origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model, tfidf vectorizer, and encoder
__model = None
__tfidf = None
__encoder = None

# Load skill sets for different job domains
skillsets = {
    "SAP Developer": ["SAP ABAP", "SAP Fiori", "SAP HANA", "SAP S/4HANA", "SAP BASIS", "SAP BW", "SAP MM", "SAP SD",
                      "OData Services", "Web Dynpro", "UI5", "BAPI", "ALE/IDocs", "SAP GRC", "AWS", "Azure", "Docker"],
    "Network Security Engineer": ["Firewalls (Cisco, Palo Alto, Fortinet)", "VPN", "IDS/IPS", "SIEM (Splunk, ELK)",
                                  "Penetration Testing", "Ethical Hacking", "SOC Operations", "Cloud Security",
                                  "Threat Hunting", "Zero Trust Security", "AWS", "Azure", "Docker"],
    "Blockchain Developer": ["Solidity", "Ethereum", "Smart Contracts", "Hyperledger Fabric", "Corda",
                             "Cryptography", "Consensus Mechanisms", "DeFi", "NFT Standards", "Web3.js", "IPFS",
                             "Truffle", "Metamask", "AWS", "Azure", "Docker"],
    "Database Developer": ["SQL", "NoSQL", "PostgreSQL", "Oracle", "MongoDB", "Data Warehousing", "ETL Pipelines",
                           "Query Optimization", "Database Replication", "Sharding", "Partitioning", "AWS", "Azure", "Docker"],
    "Web Designing": ["HTML", "CSS", "JavaScript", "UI/UX Design", "Figma", "Adobe XD", "React.js", "Vue.js",
                      "Tailwind CSS", "Bootstrap", "SEO Optimization", "A/B Testing", "Motion UI", "GSAP", "AWS", "Azure", "Docker"],
    "DevOps Engineer": ["CI/CD", "Jenkins", "GitHub Actions", "Terraform", "Kubernetes", "Docker", "Prometheus",
                        "Grafana", "Helm", "Cloud Security", "Chaos Engineering", "AWS", "Azure", "Docker"],
    "Health and Fitness": ["Personal Training", "Nutrition", "Exercise Science", "Yoga", "Strength Training",
                           "Physical Therapy", "Sports Medicine", "Health Coaching", "Wellness Programs"],
    "Python Developer": ["Python", "Flask", "Django", "FastAPI", "Pandas", "NumPy", "AsyncIO", "API Development",
                         "GraphQL", "Web Scraping", "Machine Learning", "AWS", "Azure", "Docker"],
    "Advocate": ["Legal Research", "Case Analysis", "Contract Drafting", "Litigation", "Corporate Law", "Criminal Law",
                 "Intellectual Property", "Negotiation", "Dispute Resolution"],
    "Java Developer": ["Core Java", "Spring Boot", "Spring Security", "Hibernate", "Microservices Architecture",
                       "REST APIs", "Concurrency", "Multithreading", "Docker", "AWS", "Azure"],
    "PMO (Project Management Officer)": ["Project Planning", "Agile Methodologies", "Scrum", "Risk Management",
                                         "Stakeholder Management", "JIRA", "Resource Allocation", "AWS", "Azure", "Docker"],
    "Mechanical Engineer": ["CAD (AutoCAD, SolidWorks)", "Thermodynamics", "Fluid Mechanics", "Material Science",
                            "Finite Element Analysis", "HVAC", "Manufacturing", "Quality Control", "AWS", "Azure", "Docker"],
    "Business Analyst": ["Requirement Analysis", "Process Mapping", "Business Strategy", "Market Research",
                         "Data Analysis", "SQL", "Agile Methodologies", "JIRA", "AWS", "Azure", "Docker"],
    "Data Science": ["Machine Learning", "Deep Learning", "Data Visualization", "Big Data", "Computer Vision",
                     "NLP", "Time Series Forecasting", "AutoML", "MLOps", "AWS", "Azure", "Docker"],
    "Sales": ["Lead Generation", "Cold Calling", "CRM Tools", "Negotiation", "Sales Forecasting", "Market Analysis",
              "Customer Relationship", "Presentation Skills", "AWS", "Azure", "Docker"],
    "Arts": ["Painting", "Sculpting", "Illustration", "Photography", "Digital Art", "Graphic Design", "Art History",
             "Creative Direction", "Adobe Suite"],
    "ETL Developer": ["ETL Pipelines", "Apache Airflow", "Data Warehousing", "SQL", "Spark Streaming", "Kafka",
                      "AWS Glue", "AWS", "Azure", "Docker"],
    "DotNet Developer": ["C#", "ASP.NET Core", "Entity Framework", "Microservices Architecture", "Blazor", "SignalR",
                         "Azure Functions", "GraphQL", "AWS", "Azure", "Docker"],
    "Hadoop Developer": ["HDFS", "YARN", "MapReduce", "Hive", "Pig", "Spark", "Cassandra", "Kafka", "AWS EMR",
                         "AWS", "Azure", "Docker"],
    "Electrical Engineer": ["Circuit Design", "PCB Layout", "MATLAB", "Simulink", "Power Systems", "Embedded Systems",
                            "IoT Devices", "Wireless Communication", "AWS", "Azure", "Docker"],
    "Automation Testing": ["Selenium", "Cypress", "Playwright", "JUnit", "TestNG", "API Testing", "Postman",
                           "RestAssured", "AWS", "Azure", "Docker"],
    "Operations Manager": ["Process Optimization", "Lean Six Sigma", "Supply Chain Management", "Strategic Planning",
                           "Risk Assessment", "ERP Systems", "Financial Planning", "AWS", "Azure", "Docker"],
    "HR (Human Resources)": ["Recruitment", "Employee Relations", "Talent Acquisition", "Payroll Management",
                             "HR Analytics", "Performance Management", "Onboarding", "AWS", "Azure", "Docker"],
    "Testing": ["Manual Testing", "Automated Testing", "Regression Testing", "Performance Testing", "Load Testing",
                "Security Testing", "Test Case Management", "AWS", "Azure", "Docker"],
    "Civil Engineer": ["Structural Engineering", "Construction Management", "AutoCAD", "Revit", "Building Codes",
                       "Geotechnical Engineering", "Surveying", "AWS", "Azure", "Docker"]
}


# Load model, TF-IDF vectorizer, and label encoder
def load_model():
    global __model, __tfidf, __encoder
    with open("./model.pkl", 'rb') as f:
        __model = pickle.load(f)
    with open("./tfidf.pkl", 'rb') as f:
        __tfidf = pickle.load(f)
    with open("./encoder.pkl", 'rb') as f:
        __encoder = pickle.load(f)

load_model()

def clean_text(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt.lower()

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_file.file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")
    return text

def decision_score(text: str, domain: str) -> float:
    cleaned = clean_text(text)
    vector = __tfidf.transform([cleaned]).toarray()
    if domain not in __encoder.classes_:
        raise ValueError(f"Invalid domain: {domain}")
    domain_index = list(__encoder.classes_).index(domain)
    return round(__model.decision_function(vector)[0][domain_index], 3)

def domain_classifier(text: str) -> Dict[str, float]:
    cleaned = clean_text(text)
    vector = __tfidf.transform([cleaned]).toarray()
    domain = __model.predict(vector)
    category = __encoder.inverse_transform(domain)
    # scores = __model.decision_function(vector)[0].tolist()
    return {"domain": category[0]}

def skills_match_score(text: str, domain: str) -> float:
    found = {skill for skill in skillsets.get(domain, []) if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE)}
    intersection = len(found)
    union = len(skillsets.get(domain, []))
    if union == 0:
        return 0
    jaccard = intersection / union
    return round(1 - np.exp(-5 * jaccard), 3)

@app.get("/")
def home():
    return {"message": "Resume Domain Classifier API is running!"}

@app.post("/classify/")
async def classify_resumes(files: List[UploadFile] = File(...)):
    results = {}
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed.")
        text = extract_text_from_pdf(file)
        classification = domain_classifier(clean_text(text))
        results[file.filename] = classification
    return {"classified_domains": results}

@app.post("/rank/")
async def rank_resumes(
    files: List[UploadFile] = File(...),
    target_domain: str = Query(..., title="Target Domain")
):
    results = {}
    w1, w2 = 0.7, 0.3
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files allowed.")
        text = extract_text_from_pdf(file)
        clean_txt = clean_text(text)
        skill_score = skills_match_score(clean_txt, target_domain)
        model_score = decision_score(clean_txt, target_domain)
        final_score = w1 * skill_score + w2 * model_score
        results[file.filename] = round(final_score, 3)

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return {"Ranked Scores": sorted_results}
