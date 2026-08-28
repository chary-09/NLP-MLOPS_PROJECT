import json
import os
import pickle
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("--- 12. Model Evaluation & 13. Best Model Selection ---")
    metrics_path = 'data/models/metrics.json'
    if not os.path.exists(metrics_path):
        print("Metrics file not found. Please run model training first.")
        return
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    best_model_name = None
    best_f1 = -1
    
    for name, metric in metrics.items():
        print(f"{name}: F1-Score = {metric['f1']:.4f}")
        if metric['f1'] > best_f1:
            best_f1 = metric['f1']
            best_model_name = name
            
    print(f"\nBest Model Selected: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    print("\n--- 14. Model Saving ---")
    model_filename = f"data/models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    best_model_path = 'data/models/best_model.pkl'
    
    # Save a copy as best_model.pkl
    with open(model_filename, 'rb') as f_in:
        model = pickle.load(f_in)
    with open(best_model_path, 'wb') as f_out:
        pickle.dump(model, f_out)
        
    print(f"Saved {best_model_name} as {best_model_path}")
    
    print("\n--- 15. Prediction Script & 16. Testing ---")
    # Load Vectorizer and Best Model
    with open('data/models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
        
    test_sentences = [
        "I absolutely loved this movie, it was fantastic and thrilling!",
        "This was the worst experience of my life, completely terrible and boring.",
        "It was okay, not great but not too bad either."
    ]
    
    print("Running predictions on test sentences:")
    for sentence in test_sentences:
        cleaned = clean_text(sentence)
        transformed = vectorizer.transform([cleaned])
        prediction = best_model.predict(transformed)[0]
        # Depending on how the dataset was structured, it might be 0/1 or string labels
        # Assuming typical sentiment dataset:
        print(f"Sentence: '{sentence}'")
        print(f"Predicted Sentiment: {prediction}\n")

if __name__ == '__main__':
    main()
