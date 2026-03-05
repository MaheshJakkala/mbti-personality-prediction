"""
tests/test_predict.py
---------------------
Unit tests for the preprocessing and inference pipeline.
Run with: pytest tests/
"""

import pytest
from src.preprocess import clean_text, tokens_to_string, get_mbti_stopwords


MBTI_TYPES = [
    "INFJ", "ENTP", "INTP", "INTJ", "ENTJ", "ENFJ",
    "INFP", "ENFP", "ISFP", "ISTP", "ISFJ", "ISTJ",
    "ESTP", "ESFP", "ESTJ", "ESFJ"
]


class TestCleanText:
    def test_removes_urls(self):
        text = "Check this out http://example.com it's great"
        tokens = clean_text(text)
        assert "http" not in " ".join(tokens)
        assert "example" not in " ".join(tokens)

    def test_removes_special_characters(self):
        text = "Hello!!! This is a test... #hashtag @mention"
        tokens = clean_text(text)
        for token in tokens:
            assert token.isalpha(), f"Token '{token}' contains non-alpha characters"

    def test_lowercase(self):
        text = "Hello World MBTI Test"
        tokens = clean_text(text)
        for token in tokens:
            assert token == token.lower(), f"Token '{token}' is not lowercase"

    def test_removes_stopwords(self):
        text = "the quick brown fox jumps over the lazy dog"
        tokens = clean_text(text)
        assert "the" not in tokens
        assert "over" not in tokens

    def test_custom_stopwords(self):
        text = "infj is the rarest personality type"
        tokens = clean_text(text, add_stopwords=["infj", "infjs"])
        assert "infj" not in tokens

    def test_empty_string(self):
        tokens = clean_text("")
        assert tokens == []

    def test_url_only(self):
        text = "http://www.google.com"
        tokens = clean_text(text)
        # URL placeholder 'urlstr' should be present or empty depending on stopword config
        assert isinstance(tokens, list)

    def test_lemmatization(self):
        text = "running dogs jumping cats"
        tokens_lem = clean_text(text, lem_stem="lem")
        assert isinstance(tokens_lem, list)
        assert len(tokens_lem) > 0

    def test_porter_stemming(self):
        text = "running jumps happily"
        tokens_stem = clean_text(text, lem_stem="stemp")
        assert isinstance(tokens_stem, list)

    def test_snowball_stemming(self):
        text = "running jumps happily"
        tokens_stem = clean_text(text, lem_stem="stems")
        assert isinstance(tokens_stem, list)


class TestTokensToString:
    def test_basic_join(self):
        tokens = ["hello", "world"]
        result = tokens_to_string(tokens)
        assert result == "hello world"

    def test_empty_list(self):
        result = tokens_to_string([])
        assert result == ""

    def test_single_token(self):
        result = tokens_to_string(["introvert"])
        assert result == "introvert"


class TestMbtiStopwords:
    def test_generates_lowercase(self):
        stopwords = get_mbti_stopwords(MBTI_TYPES)
        for sw in stopwords:
            assert sw == sw.lower()

    def test_includes_plurals(self):
        stopwords = get_mbti_stopwords(["INFJ"])
        assert "infj" in stopwords
        assert "infjs" in stopwords

    def test_length(self):
        stopwords = get_mbti_stopwords(MBTI_TYPES)
        assert len(stopwords) == len(MBTI_TYPES) * 2


class TestIntegration:
    def test_full_preprocessing_pipeline(self):
        """Simulate real social media post preprocessing."""
        post = (
            "As an INFJ I often feel misunderstood... "
            "http://reddit.com/r/mbti I scored 96% introverted! "
            "ENTPs always challenge me 😅 #personality"
        )
        tokens = clean_text(post, add_stopwords=get_mbti_stopwords(MBTI_TYPES))
        result = tokens_to_string(tokens)

        assert isinstance(result, str)
        assert len(result) > 0
        # MBTI types should be filtered
        assert "infj" not in result
        assert "entp" not in result
        # URLs should be gone
        assert "reddit" not in result
