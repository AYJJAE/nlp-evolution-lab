"""
TF-IDF Baseline Classifier (Bag-of-Words paradigm)
Module 1: Evolution of NLP

Represents text as high-dimensional sparse term-frequency vectors
weighted by inverse document frequency, classified with Logistic Regression.
Demonstrates lack of word-order awareness and susceptibility to negation failure.
"""

import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from models.benchmark_data import TRAIN_DATA

class TFIDFClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 1), # Classic unigram Bag of Words
            stop_words=None,    # Keep stop words visible so students see their weights
            token_pattern=r"(?u)\b\w+\b"
        )
        self.classifier = LogisticRegression(C=2.0, max_iter=200, random_state=42)
        self._train()
        
    def _train(self):
        texts = [item[0] for item in TRAIN_DATA]
        labels = [item[1] for item in TRAIN_DATA]
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.coefficients = self.classifier.coef_[0]

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
            
        start_time = time.perf_counter()
        
        # Transform input
        vec = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(vec)[0]
        prediction_idx = int(self.classifier.predict(vec)[0])
        
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        pred_label = "Positive" if prediction_idx == 1 else "Negative"
        confidence = float(probabilities[prediction_idx])
        
        # Extract active tokens and their contribution weights
        feature_indices = vec.indices
        tfidf_values = vec.data
        token_contributions = []
        
        for idx, val in zip(feature_indices, tfidf_values):
            word = self.feature_names[idx]
            coef = float(self.coefficients[idx])
            contrib = float(val * coef)
            token_contributions.append({
                "token": word,
                "tfidf": round(float(val), 4),
                "weight": round(coef, 4),
                "impact": "positive" if contrib > 0 else "negative",
                "contribution": round(contrib, 4)
            })
            
        # Sort tokens by absolute contribution
        token_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        # Educational explanation
        if any(neg in text.lower().split() for neg in ["not", "never", "no", "hardly", "cannot"]):
            pedagogical_note = (
                "TF-IDF treats text as an unordered bag of tokens. When negations like 'not' or 'hardly' appear, "
                "the model cannot bind the negation operator to neighboring sentiment words because token sequence order is discarded."
            )
        else:
            pedagogical_note = (
                "TF-IDF weighs word frequency against inverse document frequency. Prediction is driven by the dot product "
                "of sparse term frequencies and learned linear coefficients without lexical semantics or context."
            )
            
        return {
            "model_type": "TF-IDF + Logistic Regression",
            "paradigm": "Bag-of-Words (Sparse Count / Frequency)",
            "prediction": pred_label,
            "prediction_code": prediction_idx,
            "confidence": round(confidence, 4),
            "score_label": "Calibrated Logistic Probability",
            "scores": {
                "Negative": round(float(probabilities[0]), 4),
                "Positive": round(float(probabilities[1]), 4)
            },
            "inference_time_ms": max(inference_time_ms, 0.05),
            "token_contributions": token_contributions[:8], # Top contributing tokens
            "total_tokens_matched": len(token_contributions),
            "pedagogical_note": pedagogical_note
        }

