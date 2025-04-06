import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def clean_text(text):
    """Clean the text data by removing newlines, punctuation, and converting to lowercase."""
    text = re.sub(r'\n', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def main():
    # Load the dataset
    df = pd.read_csv('train.csv')

    # Preprocessing
    # Convert 'createdAt' to datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'], format='ISO8601')

    # Extract useful features from 'createdAt'
    df['year'] = df['createdAt'].dt.year
    df['month'] = df['createdAt'].dt.month
    df['day'] = df['createdAt'].dt.day

    # Clean the 'content' column
    df['content'] = df['content'].apply(clean_text)

    # Drop unnecessary columns
    X = df['content']  # Use only the 'content' column for TF-IDF
    y = df['zone_name']  # Target label

    # Create a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),  # Limit to 5000 features
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Train the model
    pipeline.fit(X, y)

    # Save the trained model to a .pkl file
    with open('logreg_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model saved as 'logreg_model.pkl'")

if __name__ == "__main__":
    main()