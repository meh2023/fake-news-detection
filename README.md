# Fake News Detection

A machine learning project that classifies news text as **real** or **fake** using TF-IDF features and logistic regression, with a Streamlit web interface for live predictions.

## Features

- Text preprocessing (cleaning, stopword removal, lemmatization)
- TF-IDF + logistic regression pipeline
- Training script with accuracy metrics
- Interactive web app for analyzing headlines and articles

## Project structure

```
fake news detection/
├── app.py              # Streamlit web interface
├── train.py            # Train and save the model
├── requirements.txt
├── data/
│   └── news_dataset.csv
├── models/             # Saved model (created after training)
└── src/
    ├── preprocess.py
    ├── train_model.py
    └── predict.py
```

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python train.py
```

## Run the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## How it works

1. **Preprocessing** — Lowercases text, removes URLs and punctuation, drops stopwords, and lemmatizes words.
2. **Features** — TF-IDF vectors with unigrams and bigrams (up to 10,000 features).
3. **Classifier** — Logistic regression with balanced class weights.
4. **Prediction** — Returns a label (`real` or `fake`) plus confidence scores.

## Dataset

The included `data/news_dataset.csv` has `text` and `label` columns. You can replace it with your own data as long as labels are `real` or `fake`.

For larger datasets, consider [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) or the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip).

## Limitations

This model learns patterns in writing style, not factual truth. Sensational real headlines or well-written misinformation may be misclassified. Always verify claims with trusted fact-checking sources.

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions
