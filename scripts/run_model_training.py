import pandas as pd
import pickle
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_data_and_vectorizer():
    print("Loading datasets and vectorizer...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    with open('data/models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    return train_df, test_df, vectorizer

def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    # Using macro avg for general metrics
    prec = precision_score(y_test, predictions, average='macro', zero_division=0)
    rec = recall_score(y_test, predictions, average='macro', zero_division=0)
    f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
    
    print(f"\n--- Model: {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def main():
    if not os.path.exists('data/processed/train.csv'):
        print("Processed data not found. Please run preprocessing script first.")
        return

    train_df, test_df, vectorizer = load_data_and_vectorizer()
    
    X_train_text = train_df['text'].fillna('')
    y_train = train_df['sentiment']
    X_test_text = test_df['text'].fillna('')
    y_test = test_df['sentiment']
    
    print("Transforming text data with TF-IDF...")
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(random_state=42, max_iter=2000)
    }
    
    metrics_results = {}
    
    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        model_path = f"data/models/{name.lower().replace(' ', '_')}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        metrics_results[name] = evaluate_model(name, model, X_test, y_test)
        
    print("\n--- 11. Model Comparison ---")
    best_model = None
    best_f1 = 0
    
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 46)
    for name, metrics in metrics_results.items():
        print(f"{name:<20} | {metrics['accuracy']:<10.4f} | {metrics['f1']:<10.4f}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = name
            
    print(f"\nBest Model based on F1-Score: {best_model} ({best_f1:.4f})")
    
    # Save metrics
    with open('data/models/metrics.json', 'w') as f:
        json.dump(metrics_results, f, indent=4)
        
if __name__ == '__main__':
    main()
