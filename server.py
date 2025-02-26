from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import pickle
import re
import os
from typing import List

app = FastAPI()

# Global variables for model, tfidf vectorizer, and encoder
__model = None
__tfidf = None
__encoder = None


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


def clean_text(txt: str) -> str:
    clean_txt = re.sub("http:\\S+\\s", "", txt)
    cleanText = re.sub('RT|cc', ' ', clean_txt)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\\x00-\\x7f]', ' ', cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText.strip()


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


def domain_classifier(text: str):
    cleaned_text = clean_text(text)
    vectorised_text = __tfidf.transform([cleaned_text]).toarray()
    domain = __model.predict(vectorised_text)
    category = __encoder.inverse_transform(domain)
    return category[0]


@app.post("/classify/")
async def classify_resumes(files: List[UploadFile] = File(...)):
    results = {}
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        text = extract_text_from_pdf(file)
        domain = domain_classifier(text)
        results[file.filename] = domain

    return {"classified_domains": results}


@app.get("/")
def home():
    return {"message": "Resume Domain Classifier API is running!"}
