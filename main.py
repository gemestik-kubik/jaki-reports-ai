from fastapi import FastAPI
import pickle
import re
from pydantic import BaseModel
import pandas as pd

# === 1. Load Model CatBoostClassifier ===
with open("catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# === 2. Inisialisasi FastAPI ===
app = FastAPI(title="Reports Classifier API", version="1.0")

# === 3. Preprocessing Function ===
def clean_text(text):
    """Clean the text data by removing newlines, punctuation, and converting to lowercase."""
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# === 4. Define Input Schema ===
class PatientData(BaseModel):
    content: str
    category_name: str
    createdAt: str  

# === 5. Endpoint untuk Prediksi Probabilitas ===
@app.post("/predict_proba")
def predict_disease_proba(data: PatientData):
    # Preprocessing input
    input_data = {
        "content": clean_text(data.content),
        "category_name": data.category_name,
        "createdAt": data.createdAt
    }
    input_df = pd.DataFrame([input_data])

    # Lakukan prediksi probabilitas
    y_proba = model.predict_proba(input_df)

    # Label target yang digunakan saat training
    target_labels = model.classes_

    # Konversi hasil probabilitas ke bentuk JSON
    result = {target_labels[i]: round(y_proba[0][i], 4) for i in range(len(target_labels))}

    return {"probabilities": result}