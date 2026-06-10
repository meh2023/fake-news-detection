import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_nltk_ready = False


def _ensure_nltk_data() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    for package in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)
    _nltk_ready = True


def clean_text(text: str) -> str:
    """Normalize news text for feature extraction."""
    if not isinstance(text, str):
        return ""

    _ensure_nltk_data()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)
