import pandas as pd
import numpy as np
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(filepath):
    print(f"--- Loading data from {filepath} ---")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows.")
    return df

def text_preprocessing(text):
    if not isinstance(text, str):
        return ""
    # Assume basic cleaning is done (lowercasing, remove tags), here we ensure it's clean
    text = text.lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    # Basic tokenization by splitting on whitespace
    return text.split()

def main():
    # Attempt to load cleaned data if it exists, else load raw
    filepath = 'data/processed/cleaned_imdb.csv'
    if not os.path.exists(filepath):
        filepath = 'storage/imdb.csv'
        if not os.path.exists(filepath):
            filepath = '../storage/imdb.csv'
            
    df = load_data(filepath)
    if df is None:
        return
        
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else ('review' if 'review' in df.columns else 'text')
    target_col = 'sentiment'
    
    print("\n--- 4. Text Preprocessing ---")
    if text_col not in ['cleaned_text']:
        print("Applying text preprocessing...")
        df['processed_text'] = df[text_col].apply(text_preprocessing)
    else:
        print("Data is already preprocessed (cleaned_text found). Copying to processed_text...")
        df['processed_text'] = df[text_col]
        
    print("\n--- 5. Tokenization ---")
    print("Applying basic tokenization...")
    df['tokens'] = df['processed_text'].apply(tokenize)
    print("Tokenization completed. First 2 rows tokens:")
    print(df['tokens'].head(2))
    
    print("\n--- 6. Train/Validation/Test Split ---")
    # Split 70% train, 15% val, 15% test
    # First split into 70% train and 30% temp
    X = df['processed_text']
    y = df[target_col] if target_col in df.columns else np.zeros(len(df))
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if target_col in df.columns else None)
    # Split temp into 50% val and 50% test (which is 15% of total each)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp if target_col in df.columns else None)
    
    print(f"Train size: {len(X_train)} (70%)")
    print(f"Validation size: {len(X_val)} (15%)")
    print(f"Test size: {len(X_test)} (15%)")
    
    # Save splits
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    pd.DataFrame({'text': X_val, 'sentiment': y_val}).to_csv(os.path.join(out_dir, 'validation.csv'), index=False)
    pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print("\n--- 7. TF-IDF ---")
    print("Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF Training shape: {X_train_tfidf.shape}")
    print(f"TF-IDF Validation shape: {X_val_tfidf.shape}")
    print(f"TF-IDF Test shape: {X_test_tfidf.shape}")
    
    # Save vectorizer
    model_dir = 'data/models'
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"\nTF-IDF Vectorizer saved to {vec_path}")

if __name__ == '__main__':
    main()
