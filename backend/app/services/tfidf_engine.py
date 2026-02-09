"""TF-IDF importance modeling for financial text analysis.

Identifies the most important financial terms in news articles
and generates feature vectors for ML model input.

Key capabilities:
- Custom financial vocabulary with domain-specific terms
- TF-IDF vectorization pipeline
- Feature extraction and weighting
- Term importance ranking
"""

import logging
import re
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Critical financial terms to track
FINANCIAL_VOCABULARY = [
    "bankruptcy", "acquisition", "merger", "layoffs", "expansion", "debt",
    "revenue", "profit", "loss", "earnings", "dividend", "stock", "share",
    "market", "trading", "investor", "growth", "decline", "surge", "plunge",
    "crash", "rally", "bull", "bear", "volatility", "inflation", "recession",
    "interest", "rate", "bond", "equity", "asset", "liability", "margin",
    "forecast", "guidance", "outlook", "upgrade", "downgrade", "overweight",
    "underweight", "outperform", "underperform", "buyback", "repurchase",
    "restructuring", "spinoff", "ipo", "delisting", "compliance", "regulation",
    "sec", "audit", "fraud", "investigation", "settlement", "fine", "penalty",
    "patent", "innovation", "disruption", "competition", "monopoly", "antitrust",
    "supply", "demand", "shortage", "surplus", "tariff", "sanction", "subsidy",
    "hedge", "derivative", "option", "futures", "commodity", "currency",
    "crypto", "blockchain", "ai", "cloud", "saas", "subscription",
    "ebitda", "cashflow", "operating", "gross", "net", "income",
    "quarter", "annual", "fiscal", "report", "filing", "disclosure",
    "ceo", "cfo", "board", "executive", "management", "leadership",
    "partnership", "alliance", "contract", "deal", "agreement",
]


class TFIDFEngine:
    """TF-IDF vectorization engine for financial text analysis."""

    def __init__(self, max_features: int = 200, ngram_range: tuple = (1, 2)):
        """Initialize TF-IDF engine.

        Args:
            max_features: Maximum number of features in vocabulary
            ngram_range: Range of n-grams (default unigrams + bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        )
        self.is_fitted = False
        self.feature_names = []

    def preprocess_for_tfidf(self, text: str) -> str:
        """Preprocess text specifically for TF-IDF analysis."""
        text = text.lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit(self, documents: list[str]):
        """Fit the TF-IDF vectorizer on a corpus of documents.

        Args:
            documents: List of text documents
        """
        preprocessed = [self.preprocess_for_tfidf(doc) for doc in documents if doc]
        if not preprocessed:
            logger.warning("No documents to fit TF-IDF on")
            return

        self.vectorizer.fit(preprocessed)
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        self.is_fitted = True
        logger.info(f"TF-IDF fitted with {len(self.feature_names)} features")

    def transform(self, documents: list[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors.

        Args:
            documents: List of text documents

        Returns:
            TF-IDF matrix (n_documents x n_features)
        """
        if not self.is_fitted:
            self.fit(documents)

        preprocessed = [self.preprocess_for_tfidf(doc) for doc in documents]
        return self.vectorizer.transform(preprocessed).toarray()

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        preprocessed = [self.preprocess_for_tfidf(doc) for doc in documents if doc]
        if not preprocessed:
            return np.array([])

        matrix = self.vectorizer.fit_transform(preprocessed).toarray()
        self.feature_names = list(self.vectorizer.get_feature_names_out())
        self.is_fitted = True
        return matrix

    def get_top_terms(self, text: str, top_n: int = 10) -> list[dict]:
        """Get the most important terms in a single document.

        Args:
            text: Input text
            top_n: Number of top terms to return

        Returns:
            List of {term, score} dicts sorted by importance
        """
        if not self.is_fitted:
            self.fit([text])

        preprocessed = self.preprocess_for_tfidf(text)
        vector = self.vectorizer.transform([preprocessed]).toarray()[0]

        term_scores = list(zip(self.feature_names, vector))
        term_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {"term": term, "score": round(float(score), 4)}
            for term, score in term_scores[:top_n]
            if score > 0
        ]

    def get_financial_term_scores(self, text: str) -> dict:
        """Score specifically financial vocabulary terms in the text.

        Returns a dict mapping financial terms to their TF-IDF scores,
        only for terms that appear in the text.
        """
        preprocessed = self.preprocess_for_tfidf(text)
        words = set(preprocessed.split())

        scores = {}
        for term in FINANCIAL_VOCABULARY:
            if term in words:
                # Calculate simple TF for this term
                tf = preprocessed.split().count(term)
                doc_length = len(preprocessed.split())
                scores[term] = round(tf / max(doc_length, 1), 4)

        return scores

    def get_corpus_important_terms(
        self, documents: list[str], top_n: int = 20
    ) -> list[dict]:
        """Get the most important terms across a corpus.

        Args:
            documents: List of text documents
            top_n: Number of top terms

        Returns:
            List of {term, avg_score, doc_frequency} dicts
        """
        matrix = self.fit_transform(documents)
        if matrix.size == 0:
            return []

        avg_scores = np.mean(matrix, axis=0)
        doc_frequencies = np.sum(matrix > 0, axis=0)

        terms = []
        for i, name in enumerate(self.feature_names):
            if avg_scores[i] > 0:
                terms.append({
                    "term": name,
                    "avg_score": round(float(avg_scores[i]), 4),
                    "doc_frequency": int(doc_frequencies[i]),
                })

        terms.sort(key=lambda x: x["avg_score"], reverse=True)
        return terms[:top_n]

    def get_feature_vector(self, text: str) -> dict:
        """Get a compact feature vector for ML input.

        Combines TF-IDF scores with financial term indicators.
        """
        tfidf_scores = {}
        if self.is_fitted:
            preprocessed = self.preprocess_for_tfidf(text)
            vector = self.vectorizer.transform([preprocessed]).toarray()[0]
            # Use only non-zero features to keep it compact
            for i, score in enumerate(vector):
                if score > 0:
                    tfidf_scores[self.feature_names[i]] = round(float(score), 4)

        financial_scores = self.get_financial_term_scores(text)

        return {
            "tfidf_features": tfidf_scores,
            "financial_term_scores": financial_scores,
            "tfidf_magnitude": round(float(np.sqrt(sum(v**2 for v in tfidf_scores.values()))), 4) if tfidf_scores else 0.0,
            "financial_term_count": len(financial_scores),
        }


# Module-level singleton
tfidf_engine = TFIDFEngine()
