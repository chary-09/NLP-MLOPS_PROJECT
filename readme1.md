# NLP Sentiment Analysis MLOps Platform

## Phase 1: NLP and Machine Learning

This project implements the basic sentiment analysis pipeline:

```text
Dataset -> Clean Text -> Tokenization -> TF-IDF -> Train Models
				-> Evaluate Models -> Save Best Model -> Predict Sentiment
```

Phase 1 focuses only on NLP and machine learning. The dashboard, FastAPI, database, explainability, monitoring, Docker, and CI/CD features are planned for later phases.

## What Has Been Implemented

- Loads sentiment data from `data/raw/sentiment.csv`.
- Cleans text by converting it to lowercase and removing URLs, punctuation, numbers, and extra spaces.
- Tokenizes cleaned text into individual words.
- Converts text into TF-IDF features using unigrams and bigrams.
- Trains three machine learning models:
	- Logistic Regression
	- Multinomial Naive Bayes
	- Linear SVM
- Compares models using accuracy, precision, recall, and F1-score.
- Creates a confusion matrix for each model evaluation.
- Selects the model with the best weighted F1-score.
- Saves the selected model, TF-IDF vectorizer, and evaluation metrics.
- Predicts `POSITIVE`, `NEGATIVE`, or `NEUTRAL` sentiment for new text.
- Includes automated tests for preprocessing, tokenization, TF-IDF, training, evaluation, and prediction.

## Project Structure

```text
data/
├── raw/
│   └── sentiment.csv              # Input dataset
└── models/                        # Generated model files

src/
├── nlp/
│   ├── preprocessor.py            # Text cleaning
│   ├── tokenizer.py               # Tokenization
│   └── feature_extraction.py      # TF-IDF vectorizer
└── model/
		├── trainer.py                 # Three model trainers
		├── predictor.py               # Sentiment prediction
		├── evaluator.py               # Metrics and confusion matrix
		└── model_utils.py             # Save/load artifacts

scripts/
├── train_model.py                 # Train and save the best model
└── run_prediction.py              # Predict sentiment from the command line

tests/
├── test_nlp.py                    # NLP tests
└── test_model.py                  # Model tests

notebooks/
├── 01_EDA.ipynb
├── 02_Data_Preprocessing.ipynb
├── 03_Model_Training.ipynb
└── 04_Model_Evaluation.ipynb
```

## Dataset Format

The input file must be named `sentiment.csv` and contain these two columns:

```csv
text,sentiment
I love this product,positive
This service is disappointing,negative
It works as expected,neutral
```

The supported labels are `positive`, `negative`, and `neutral`.

## Installation on Windows

Open PowerShell in the project directory:

```powershell
cd "D:\sonu collage\KL University profile\3rd year\NLP\NLP-PDNC project"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in PowerShell or use the Python executable directly:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Train the Models

Run:

```powershell
python scripts\train_model.py
```

The script trains all three models, evaluates them, selects the best model by weighted F1-score, and creates these files in `data/models/`:

- `sentiment_model.pkl` - the selected classifier
- `tfidf_vectorizer.pkl` - the fitted TF-IDF vectorizer
- `metrics.json` - metrics and confusion matrices for all models

Example output:

```text
Best model: logistic_regression
logistic_regression: accuracy=1.000, precision=1.000, recall=1.000, f1=1.000
naive_bayes: accuracy=1.000, precision=1.000, recall=1.000, f1=1.000
linear_svm: accuracy=1.000, precision=1.000, recall=1.000, f1=1.000
```

The included dataset contains only a few examples, so these metrics are training-set results and should not be treated as production accuracy. Add a larger, representative dataset before relying on the model.

## Predict Sentiment

After training, run:

```powershell
python scripts\run_prediction.py "I really love this product!"
```

Example output:

```text
Sentiment: POSITIVE
Confidence: 50%
```

The confidence value is an estimate from the selected classifier. With the small sample dataset, confidence may be limited even when the predicted sentiment is correct.

If no text is provided, the script uses `I really love this product!` as the default example:

```powershell
python scripts\run_prediction.py
```

## Run Tests

Run the full test suite with:

```powershell
python -m pytest -q
```

The current test suite covers the Phase 1 implementation and should report:

```text
13 passed
```

To run only the Phase 1 NLP and model tests:

```powershell
python -m pytest -q tests\test_nlp.py tests\test_model.py
```

## Run the Notebooks

Open the `notebooks` folder in VS Code and run the notebooks in this order:

1. `01_EDA.ipynb` - inspect the dataset and sentiment distribution.
2. `02_Data_Preprocessing.ipynb` - clean text and inspect tokens.
3. `03_Model_Training.ipynb` - train the three classifiers.
4. `04_Model_Evaluation.ipynb` - compare metrics and review model performance.

## Makefile Commands

If `make` is available, these shortcuts can be used:

```powershell
make install   # Install dependencies
make train     # Train and save models
make test      # Run tests
```

## Phase 2 and Phase 3

The following features are intentionally outside Phase 1:

- Streamlit dashboard
- FastAPI prediction service
- Database integration
- SHAP and LIME explanations
- Monitoring and data drift detection
- Docker and CI/CD
