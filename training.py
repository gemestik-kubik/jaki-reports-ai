import pandas as pd
from catboost import CatBoostClassifier
import re
import pickle

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
    X = df.drop(['id', 'code', 'createdAt', 'zone_name', 'status'], axis=1)  # Drop irrelevant columns
    y = df['zone_name']  # Target label

    # Identify categorical and text features
    categorical_features = ['category_name']
    text_features = ['content']

    # Initialize the CatBoost classifier
    model = CatBoostClassifier(
        iterations=1000,
        cat_features=categorical_features,
        text_features=text_features,
        verbose=100
    )

    # Train the model on the entire dataset
    model.fit(X, y)

    # Save the trained model to a .pkl file
    with open('catboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved as 'catboost_model.pkl'")

if __name__ == "__main__":
    main()