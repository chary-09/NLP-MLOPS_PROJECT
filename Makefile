.PHONY: install test train api dashboard

install:
	pip install -r requirements.txt

test:
	pytest

train:
	python scripts/train_model.py

api:
	uvicorn src.api.main:app --reload

dashboard:
	streamlit run src/dashboard/app.py
