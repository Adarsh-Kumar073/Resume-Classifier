# Resume Domain Classifier API

This is a FastAPI-based application that extracts text from PDF resumes and classifies them into domains using a pre-trained Support Vector Machine (SVM) model.

## Features
- Accepts multiple PDF resume files for classification.
- Extracts text using `pdfplumber`.
- Cleans and preprocesses text using TF-IDF vectorization and Label Encoding.
- Classifies resumes into predefined domains using an SVM model.
- Ranks resumes based on a combination of classification confidence and skillset match score.
- Provides a selection-based input for target domain selection.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7). Install the required dependencies using:
```bash
pip install fastapi uvicorn pdfplumber scikit-learn
```

## Running the API
To start the FastAPI server, run:
```bash
uvicorn main:app --reload
```
Replace `main` with the actual script filename if different.

## API Endpoints
### Home Endpoint
- **GET /**
  - Returns a message indicating the API is running.

### Resume Classification Endpoint
- **POST /classify/**
  - Accepts PDF resume files and returns classified domains.
  - Example request:
    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/classify/' -F 'files=@resume.pdf'
    ```

### Resume Ranking Endpoint
- **POST/rank/**
  -Accepts PDF resumes and a selected target domain.
  -Returns ranked resumes in decreasing order of suitability for the target domain.
  -Example request:
   ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/rank/' -F 'files=@resume1.pdf' -F 'files=@resume2.pdf' -F 'target_domain=Python Developer'
    ```
  

## Model Details
- The classification model is built using **Support Vector Machine (SVM)**.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** is used for text vectorization.
- **Label Encoding** is used to transform domain labels into numerical format.
- The model is trained to classify resumes into different domains based on extracted text features.
