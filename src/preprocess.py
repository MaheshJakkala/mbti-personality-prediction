"""
preprocess.py
-------------
Text cleaning and preprocessing utilities for the MBTI Personality Prediction project.
Handles URL removal, stopword filtering, and optional lemmatization/stemming.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# Download required NLTK data on first use
def _ensure_nltk_data():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
        nltk.download("wordnet")

_ensure_nltk_data()

URL_RE = re.compile(
    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
    r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+"
    r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»\u201c\u201d\u2018\u2019]))"
)
SYMBOLS_RE = re.compile(r"[^a-zA-Z]")


def get_mbti_stopwords(mbti_types):
    """
    Generate a list of stopwords from MBTI type labels to prevent data leakage.

    Args:
        mbti_types (list): List of MBTI type strings e.g. ['INFJ', 'ENTP', ...]

    Returns:
        list: Lowercase type tokens and their plurals
    """
    lower = [t.lower() for t in mbti_types]
    return lower + [t + "s" for t in lower]


def clean_text(data, add_stopwords=None, lem_stem=None):
    """
    Convert raw post text into a list of clean tokens.

    Steps:
      1. Lowercase
      2. Replace URLs with 'urlstr'
      3. Remove non-alphabetic characters
      4. Remove stopwords (NLTK English + optional extras)
      5. Optional: lemmatize or stem

    Args:
        data (str): Raw input text (may contain '|||' separators).
        add_stopwords (list, optional): Extra stopwords to filter out.
        lem_stem (str, optional): One of 'lem', 'stemp' (Porter), 'stems' (Snowball).

    Returns:
        list[str]: List of processed tokens.
    """
    base_stopwords = set(stopwords.words("english"))
    if add_stopwords:
        base_stopwords.update(add_stopwords)

    text = data.lower()
    text = URL_RE.sub("urlstr", text)
    text = SYMBOLS_RE.sub(" ", text)

    tokens = [w for w in text.split() if w not in base_stopwords]

    if lem_stem == "lem":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    elif lem_stem == "stemp":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    elif lem_stem == "stems":
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def tokens_to_string(tokens):
    """Join token list back to a single string."""
    return " ".join(tokens)


def preprocess_series(series, add_stopwords=None, lem_stem=None):
    """
    Apply clean_text to a pandas Series of raw post strings.

    Args:
        series (pd.Series): Series of raw text posts.
        add_stopwords (list, optional): Additional stopwords.
        lem_stem (str, optional): Lemmatization/stemming strategy.

    Returns:
        pd.Series: Series of cleaned strings.
    """
    cleaned = series.map(
        lambda x: tokens_to_string(clean_text(x, add_stopwords=add_stopwords, lem_stem=lem_stem))
    )
    return cleaned
