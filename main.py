from fastapi import FastAPI
import pickle
import re
from pydantic import BaseModel
import pandas as pd

# === 1. Load Model CatBoostClassifier ===
with open("catboost_model.pkl", "rb") as f:
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

def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data to match the training pipeline."""
    # Convert 'createdAt' to datetime
    data['createdAt'] = pd.to_datetime(data['createdAt'], format='ISO8601')

    # Extract useful features from 'createdAt'
    data['year'] = data['createdAt'].year
    data['month'] = data['createdAt'].month
    data['day'] = data['createdAt'].day

    # Clean the 'content' column
    data['content'] = clean_text(data['content'])

    # Create a DataFrame
    input_df = pd.DataFrame([data])

    # Drop unnecessary columns to match the training pipeline
    input_df = input_df.drop(['createdAt'], axis=1)

    return input_df

# === 4. Define Input Schema ===
class ReportData(BaseModel):
    content: str
    category_name: str
    createdAt: str  # ISO8601 datetime format

# === 5. Endpoint untuk Prediksi Probabilitas ===
@app.post("/predict_proba")
def predict_report_proba(data: ReportData):
    # Preprocessing input
    input_data = preprocess_input(data.dict())

    # Lakukan prediksi probabilitas
    y_proba = model.predict_proba(input_data)

    # Label target yang digunakan saat training
    target_labels = model.classes_

    # Konversi hasil probabilitas ke bentuk JSON
    result = {target_labels[i]: round(y_proba[0][i], 4) for i in range(len(target_labels))}

    return {"probabilities": result}