"""Advanced sentiment analysis engine.

Implements multiple sentiment analysis approaches:
1. VADER - optimized for social media and financial text
2. TextBlob - general purpose NLP sentiment
3. NER-based entity extraction for financial context
4. Ensemble scoring combining multiple models

Includes text preprocessing: stopword removal, lemmatization, NER.
"""

import re
import logging
from typing import Optional

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass


class SentimentAnalyzer:
    """Multi-model sentiment analysis engine for financial text."""

    # Financial domain-specific lexicon adjustments for VADER
    FINANCIAL_LEXICON = {
        "bullish": 2.5,
        "bearish": -2.5,
        "upgrade": 2.0,
        "downgrade": -2.0,
        "outperform": 1.8,
        "underperform": -1.8,
        "buy": 1.5,
        "sell": -1.5,
        "bankruptcy": -3.5,
        "default": -3.0,
        "surge": 2.0,
        "plunge": -2.5,
        "crash": -3.0,
        "rally": 2.0,
        "dividend": 1.0,
        "layoffs": -2.0,
        "restructuring": -1.0,
        "acquisition": 0.5,
        "merger": 0.5,
        "lawsuit": -1.5,
        "fraud": -3.5,
        "innovation": 1.5,
        "expansion": 1.5,
        "recession": -2.5,
        "growth": 1.5,
        "decline": -1.5,
        "loss": -2.0,
        "profit": 2.0,
        "revenue": 0.5,
        "debt": -1.0,
        "overvalued": -1.5,
        "undervalued": 1.5,
    }

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        # Augment VADER lexicon with financial terms
        self.vader.lexicon.update(self.FINANCIAL_LEXICON)
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            self.stop_words = set()

    def preprocess(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove special chars but keep sentence structure
        text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(self, text: str) -> str:
        """Lemmatize text tokens."""
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        lemmatized = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(lemmatized)

    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using NLTK NER."""
        entities = []
        try:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
                chunks = nltk.ne_chunk(tagged)
                for chunk in chunks:
                    if hasattr(chunk, "label"):
                        entity_text = " ".join(c[0] for c in chunk)
                        entities.append({
                            "text": entity_text,
                            "type": chunk.label(),
                        })
        except Exception as e:
            logger.debug(f"NER extraction failed: {e}")
        return entities

    def analyze_vader(self, text: str) -> dict:
        """Analyze sentiment using VADER with financial lexicon."""
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "compound": compound,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def analyze_textblob(self, text: str) -> dict:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        return {
            "label": label,
            "polarity": polarity,
            "subjectivity": subjectivity,
        }

    def analyze_ensemble(self, text: str) -> dict:
        """Combine VADER and TextBlob for ensemble sentiment scoring.

        The ensemble score is a weighted average:
        - VADER compound: 60% weight (better for financial/social text)
        - TextBlob polarity: 40% weight (general purpose baseline)

        Returns unified sentiment result.
        """
        preprocessed = self.preprocess(text)
        lemmatized = self.lemmatize(preprocessed)

        vader_result = self.analyze_vader(preprocessed)
        textblob_result = self.analyze_textblob(lemmatized)

        # Weighted ensemble
        vader_weight = 0.6
        textblob_weight = 0.4

        # Normalize VADER compound (-1,1) and TextBlob polarity (-1,1) are same scale
        ensemble_polarity = (
            vader_result["compound"] * vader_weight
            + textblob_result["polarity"] * textblob_weight
        )

        if ensemble_polarity > 0.05:
            label = "Positive"
        elif ensemble_polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        entities = self.extract_entities(text)

        return {
            "label": label,
            "polarity": round(ensemble_polarity, 4),
            "subjectivity": round(textblob_result["subjectivity"], 4),
            "vader_compound": round(vader_result["compound"], 4),
            "textblob_polarity": round(textblob_result["polarity"], 4),
            "entities": entities,
            "vader_detail": {
                "positive": vader_result["positive"],
                "negative": vader_result["negative"],
                "neutral": vader_result["neutral"],
            },
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze sentiment for a batch of texts."""
        return [self.analyze_ensemble(text) for text in texts]

    def get_sentence_sentiments(self, text: str) -> list[dict]:
        """Get sentiment for each sentence in the text."""
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            sentences = text.split(".")
        results = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                result = self.analyze_ensemble(sentence)
                result["sentence"] = sentence
                results.append(result)
        return results


# Module-level singleton
sentiment_analyzer = SentimentAnalyzer()
