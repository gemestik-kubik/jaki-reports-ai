from fastapi import FastAPI
import pickle
import re
from pydantic import BaseModel
import pandas as pd

# === 1. Load Model Logistic Regression ===
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# === 2. Inisialisasi FastAPI ===
app = FastAPI(title="JAKI Classifier API", version="1.0")

# === 3. Preprocessing Function ===
def clean_text(text: str) -> str:
    """Clean the text data by removing newlines, punctuation, and converting to lowercase."""
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# === 4. Define Input Schema ===
class ReportData(BaseModel):
    content: str
    category_name: str  # This is unused but kept for compatibility
    createdAt: str  # ISO8601 datetime format

# === 5. Endpoint untuk Prediksi ===
@app.post("/predict_proba")
def predict_report_proba(data: ReportData):
    # Preprocessing input
    cleaned_content = clean_text(data.content)

    # Predict probabilities
    y_proba = model.predict_proba([cleaned_content])

    # Label target yang digunakan saat training
    target_labels = model.classes_

    # Konversi hasil probabilitas ke bentuk JSON
    result = {target_labels[i]: round(y_proba[0][i], 4) for i in range(len(target_labels))}

    return {"probabilities": result}