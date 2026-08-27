import pandas as pd
import re
import os

def load_data(filepath):
    print(f"--- 1. Loading IMDb Dataset from {filepath} ---")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows.")
    return df

def perform_eda(df):
    print("\n--- 2. Data Inspection / EDA ---")
    print("DataFrame Info:")
    df.info()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    if 'sentiment' in df.columns:
        print("\nSentiment distribution:")
        print(df['sentiment'].value_counts())
    
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits (keeping only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def data_cleaning(df):
    print("\n--- 3. Data Cleaning ---")
    text_col = 'review' if 'review' in df.columns else 'text'
    if text_col not in df.columns:
        print(f"Error: Neither 'review' nor 'text' found in columns: {df.columns}")
        return df
    
    print("Applying text cleaning...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    print("Data cleaning completed. First 5 cleaned rows:")
    print(df[['cleaned_text']].head())
    return df

def main():
    filepath = '../storage/imdb.csv'
    # Fallback to local storage if running from project root
    if not os.path.exists(filepath):
        filepath = 'storage/imdb.csv'
        
    df = load_data(filepath)
    if df is not None:
        perform_eda(df)
        df = data_cleaning(df)
        
        # Save cleaned dataset
        out_dir = 'data/processed'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'cleaned_imdb.csv')
        df.to_csv(out_path, index=False)
        print(f"\nCleaned dataset saved to {out_path}")

if __name__ == '__main__':
    main()
