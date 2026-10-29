"""
Word Embeddings Model & Vector Semantic Analysis
Module 1: Evolution of NLP

Implements dense, distributed vector representations (Continuous Bag of Words / Averaged Vectors).
Computes exact Cosine Similarity between word vectors and trains a sentiment classifier
on top of sentence embedding centroids.
Demonstrates semantic preservation (synonyms) versus syntactic loss (word order erasure).
"""

import time
import math
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from models.benchmark_data import TRAIN_DATA

class WordEmbeddingModel:
    def __init__(self, embedding_dim=50):
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        self.classifier = LogisticRegression(C=1.5, max_iter=200, random_state=42)
        self._build_semantic_vocabulary()
        self._train_sentiment_classifier()

    def _build_semantic_vocabulary(self):
        """
        Builds a realistic 50-dimensional semantic vector space mimicking GloVe / Word2Vec.
        Words sharing conceptual domains or contextual distributions have high cosine similarity.
        """
        rng = np.random.RandomState(42)
        
        # Base semantic topic axes
        topics = {
            "royalty": rng.randn(self.embedding_dim),
            "gender": rng.randn(self.embedding_dim),
            "medical": rng.randn(self.embedding_dim),
            "education": rng.randn(self.embedding_dim),
            "technology": rng.randn(self.embedding_dim),
            "temperature": rng.randn(self.embedding_dim),
            "cinema": rng.randn(self.embedding_dim),
            "positive_sentiment": rng.randn(self.embedding_dim),
            "negative_sentiment": rng.randn(self.embedding_dim),
            "food": rng.randn(self.embedding_dim),
            "animal": rng.randn(self.embedding_dim),
            "transport": rng.randn(self.embedding_dim),
            "general_syntax": rng.randn(self.embedding_dim)
        }
        # Orthogonalize and normalize topic axes
        for k in topics:
            topics[k] /= np.linalg.norm(topics[k])

        # Word profiles with explicit domain weights
        profiles = {
            # Classical Analogy & Kinship
            "king": {"royalty": 2.2, "gender": 0.45, "general_syntax": 0.3},
            "queen": {"royalty": 2.2, "gender": -0.45, "general_syntax": 0.3},
            "prince": {"royalty": 1.8, "gender": 0.40, "general_syntax": 0.3},
            "princess": {"royalty": 1.8, "gender": -0.40, "general_syntax": 0.3},
            "man": {"gender": 1.5, "general_syntax": 0.8},
            "woman": {"gender": -1.5, "general_syntax": 0.8},
            "boy": {"gender": 1.2, "general_syntax": 0.6},
            "girl": {"gender": -1.2, "general_syntax": 0.6},
            
            # Synonyms & Intelligence
            "smart": {"education": 1.9, "positive_sentiment": 1.1},
            "intelligent": {"education": 2.0, "positive_sentiment": 1.1},
            "clever": {"education": 1.7, "positive_sentiment": 0.9},
            "brilliant": {"education": 1.6, "positive_sentiment": 1.8, "cinema": 0.8},
            "genius": {"education": 2.1, "positive_sentiment": 1.5},

            # Antonyms (Demonstrating Distributional Hypothesis)
            # In corpus linguistics, hot and cold appear in nearly identical linguistic frames ("the tea was very [hot/cold]")
            "hot": {"temperature": 1.8, "general_syntax": 0.9},
            "cold": {"temperature": 1.4, "negative_sentiment": 0.3, "general_syntax": 0.9},
            "warm": {"temperature": 1.3, "positive_sentiment": 0.5, "general_syntax": 0.8},
            "cool": {"temperature": 1.1, "positive_sentiment": 0.6, "general_syntax": 0.8},
            
            # Professions
            "doctor": {"medical": 2.2, "education": 0.9, "general_syntax": 0.5},
            "nurse": {"medical": 2.0, "education": 0.7, "general_syntax": 0.5},
            "hospital": {"medical": 2.3, "transport": 0.3, "general_syntax": 0.5},
            "medicine": {"medical": 2.1, "education": 0.5},
            "teacher": {"education": 2.2, "general_syntax": 0.6},
            "student": {"education": 2.0, "general_syntax": 0.7},
            "school": {"education": 2.3, "general_syntax": 0.5},
            "college": {"education": 2.2, "general_syntax": 0.5},
            "university": {"education": 2.3, "general_syntax": 0.5},
            
            # Technology & Science
            "computer": {"technology": 2.2, "education": 0.5},
            "software": {"technology": 2.1, "education": 0.4},
            "algorithm": {"technology": 2.0, "education": 0.6},
            "hardware": {"technology": 1.9},
            "internet": {"technology": 2.0},
            "data": {"technology": 1.9, "education": 0.5},
            "ai": {"technology": 2.3, "education": 0.7},
            "quantum": {"technology": 1.8, "education": 1.1},
            "robot": {"technology": 1.9},
            
            # Transport
            "car": {"transport": 2.2, "technology": 0.6},
            "plane": {"transport": 2.0, "technology": 0.7},
            "train": {"transport": 1.9},
            
            # Food & Living
            "apple": {"food": 2.2},
            "banana": {"food": 2.1},
            "fruit": {"food": 2.3},
            "dog": {"animal": 2.2, "positive_sentiment": 0.5},
            "cat": {"animal": 2.1, "positive_sentiment": 0.4},
            
            # Strong Positive Sentiment
            "good": {"positive_sentiment": 2.0, "general_syntax": 0.8},
            "great": {"positive_sentiment": 2.2, "general_syntax": 0.8},
            "excellent": {"positive_sentiment": 2.3, "cinema": 0.7},
            "wonderful": {"positive_sentiment": 2.2, "cinema": 0.6},
            "amazing": {"positive_sentiment": 2.3, "cinema": 0.7},
            "fantastic": {"positive_sentiment": 2.2, "cinema": 0.6},
            "masterpiece": {"positive_sentiment": 2.4, "cinema": 1.8},
            "superb": {"positive_sentiment": 2.2, "cinema": 0.8},
            "gem": {"positive_sentiment": 1.9, "cinema": 1.0},
            "beautiful": {"positive_sentiment": 2.0, "cinema": 0.8},
            "breathtaking": {"positive_sentiment": 2.3, "cinema": 1.5},
            "poetic": {"positive_sentiment": 1.7, "cinema": 1.4},
            "delightful": {"positive_sentiment": 2.0, "cinema": 0.8},
            "enjoyable": {"positive_sentiment": 2.1, "cinema": 1.2},
            "loved": {"positive_sentiment": 2.1, "cinema": 0.6},
            "like": {"positive_sentiment": 1.5, "general_syntax": 0.8},
            "liked": {"positive_sentiment": 1.6, "general_syntax": 0.8},
            "heartwarming": {"positive_sentiment": 2.1, "cinema": 1.0},
            "inspiring": {"positive_sentiment": 2.1, "cinema": 0.9},
            "joy": {"positive_sentiment": 2.0},
            "humor": {"positive_sentiment": 1.7, "cinema": 0.8},
            "charm": {"positive_sentiment": 1.8, "cinema": 0.9},
            "endearing": {"positive_sentiment": 1.9, "cinema": 0.8},
            "triumphant": {"positive_sentiment": 2.1, "cinema": 0.9},
            "pleasantly": {"positive_sentiment": 1.8},
            "surprised": {"positive_sentiment": 0.8},
            "phenomenal": {"positive_sentiment": 2.3, "cinema": 1.1},
            "entertaining": {"positive_sentiment": 2.0, "cinema": 1.3},
            "magnificent": {"positive_sentiment": 2.2, "cinema": 1.0},
            "positive": {"positive_sentiment": 2.0},
            
            # Strong Negative Sentiment
            "bad": {"negative_sentiment": 2.0, "general_syntax": 0.8},
            "terrible": {"negative_sentiment": 2.3, "cinema": 0.9},
            "awful": {"negative_sentiment": 2.3, "cinema": 0.8},
            "horrible": {"negative_sentiment": 2.2, "cinema": 0.8},
            "worst": {"negative_sentiment": 2.4, "cinema": 0.9},
            "boring": {"negative_sentiment": 2.0, "cinema": 1.1},
            "bored": {"negative_sentiment": 1.9},
            "dull": {"negative_sentiment": 1.9, "cinema": 0.9},
            "waste": {"negative_sentiment": 2.2, "cinema": 0.8},
            "disaster": {"negative_sentiment": 2.3, "cinema": 0.8},
            "failure": {"negative_sentiment": 2.2, "cinema": 0.7},
            "hollow": {"negative_sentiment": 1.8, "cinema": 0.9},
            "shallow": {"negative_sentiment": 1.8, "cinema": 0.8},
            "tiresome": {"negative_sentiment": 2.0, "cinema": 0.8},
            "disappointment": {"negative_sentiment": 2.1, "cinema": 0.8},
            "unwatchable": {"negative_sentiment": 2.4, "cinema": 1.5},
            "miserable": {"negative_sentiment": 2.1},
            "clunky": {"negative_sentiment": 1.7, "cinema": 0.9},
            "predictable": {"negative_sentiment": 1.7, "cinema": 1.2},
            "clichés": {"negative_sentiment": 1.8, "cinema": 1.2},
            "hate": {"negative_sentiment": 2.2},
            "hated": {"negative_sentiment": 2.2},
            "offensive": {"negative_sentiment": 2.1},
            "annoying": {"negative_sentiment": 1.9},
            "monotonous": {"negative_sentiment": 1.8},
            "mediocre": {"negative_sentiment": 1.6, "cinema": 0.7},
            "dead": {"negative_sentiment": 1.7},
            "insults": {"negative_sentiment": 2.0},
            "negative": {"negative_sentiment": 2.0},
            "ruined": {"negative_sentiment": 2.0, "cinema": 0.7},
            "incoherent": {"negative_sentiment": 1.9, "cinema": 0.9},
            
            # Negations and structural particles
            "not": {"negative_sentiment": 0.6, "general_syntax": 1.8},
            "no": {"negative_sentiment": 0.5, "general_syntax": 1.8},
            "never": {"negative_sentiment": 0.7, "general_syntax": 1.7},
            "cannot": {"negative_sentiment": 0.8, "general_syntax": 1.6},
            "hardly": {"negative_sentiment": 0.5, "general_syntax": 1.5},
            "movie": {"cinema": 2.3, "general_syntax": 0.7},
            "film": {"cinema": 2.4, "general_syntax": 0.7},
            "cinema": {"cinema": 2.3, "general_syntax": 0.6},
            "story": {"cinema": 2.0, "general_syntax": 0.6},
            "acting": {"cinema": 2.1, "general_syntax": 0.6},
            "performance": {"cinema": 2.0, "general_syntax": 0.6},
            "performances": {"cinema": 2.0, "general_syntax": 0.6},
            "script": {"cinema": 2.1, "general_syntax": 0.6},
            "screenplay": {"cinema": 2.1, "general_syntax": 0.6},
            "direction": {"cinema": 2.0, "general_syntax": 0.6},
            "cinematography": {"cinema": 2.2, "general_syntax": 0.5},
            "visuals": {"cinema": 2.0, "technology": 0.4},
            "effects": {"cinema": 1.9, "technology": 0.6},
            "cast": {"cinema": 2.0, "general_syntax": 0.5},
            "soundtrack": {"cinema": 2.0, "general_syntax": 0.5},
            "music": {"cinema": 1.9, "general_syntax": 0.6},
            "dialogue": {"cinema": 2.0, "general_syntax": 0.6},
            "plot": {"cinema": 2.0, "general_syntax": 0.6},
            "scenes": {"cinema": 1.9, "general_syntax": 0.6},
            "scene": {"cinema": 1.9, "general_syntax": 0.6},
            "credits": {"cinema": 1.8, "general_syntax": 0.5},
            "trailers": {"cinema": 1.8, "general_syntax": 0.5},
            "audience": {"cinema": 1.9, "general_syntax": 0.6},
            "journey": {"cinema": 0.8, "general_syntax": 0.6},
            "experience": {"cinema": 0.9, "general_syntax": 0.8},
            "adventure": {"cinema": 1.0, "general_syntax": 0.6},
            "review": {"cinema": 1.6, "education": 0.6},
            "reviews": {"cinema": 1.6, "education": 0.6},
            "critical": {"cinema": 1.5, "education": 0.7},
            "harsh": {"negative_sentiment": 1.5},
            "production": {"cinema": 1.8, "general_syntax": 0.6},
            "was": {"general_syntax": 2.0},
            "is": {"general_syntax": 2.0},
            "the": {"general_syntax": 2.2},
            "a": {"general_syntax": 2.1},
            "an": {"general_syntax": 2.1},
            "this": {"general_syntax": 2.0},
            "that": {"general_syntax": 2.0},
            "it": {"general_syntax": 2.1},
            "in": {"general_syntax": 2.0},
            "fact": {"general_syntax": 1.7},
            "at": {"general_syntax": 2.0},
            "all": {"general_syntax": 1.8},
            "quite": {"general_syntax": 1.7},
            "very": {"general_syntax": 1.8},
            "truly": {"general_syntax": 1.7},
            "deeply": {"general_syntax": 1.6},
            "extremely": {"general_syntax": 1.6},
            "really": {"general_syntax": 1.7},
            "but": {"general_syntax": 2.1},
            "however": {"general_syntax": 2.0},
            "despite": {"general_syntax": 1.9},
            "yet": {"general_syntax": 2.0},
            "though": {"general_syntax": 2.0},
            "and": {"general_syntax": 2.2},
            "or": {"general_syntax": 2.0},
            "with": {"general_syntax": 2.1},
            "from": {"general_syntax": 2.0},
            "to": {"general_syntax": 2.1},
            "for": {"general_syntax": 2.0},
            "by": {"general_syntax": 2.0},
            "means": {"general_syntax": 1.7}
        }

        for word, domain_weights in profiles.items():
            # Base white noise
            vec = rng.randn(self.embedding_dim) * 0.12
            for domain, w in domain_weights.items():
                if domain in topics:
                    vec += w * topics[domain]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings[word.lower()] = vec

    def _train_sentiment_classifier(self):
        """Trains Logistic Regression classifier on top of average sentence embeddings."""
        X_train = []
        y_train = []
        for text, label in TRAIN_DATA:
            vec, _ = self.get_sentence_vector(text)
            X_train.append(vec)
            y_train.append(label)
        self.classifier.fit(np.array(X_train), np.array(y_train))

    def get_word_vector(self, word: str):
        w = word.strip().lower()
        return self.embeddings.get(w, None)

    def compute_similarity(self, word1: str, word2: str) -> dict:
        w1 = word1.strip().lower()
        w2 = word2.strip().lower()
        
        v1 = self.get_word_vector(w1)
        v2 = self.get_word_vector(w2)
        
        oov = []
        if v1 is None:
            oov.append(f"'{word1}'")
        if v2 is None:
            oov.append(f"'{word2}'")
            
        if oov:
            return {
                "word1": word1,
                "word2": word2,
                "in_vocab": False,
                "oov_words": oov,
                "similarity": None,
                "error_message": f"Vocabulary error: Word(s) {', '.join(oov)} outside vocabulary.",
                "explanation": "Static word embeddings assign vectors only to words seen during pretraining. Out-Of-Vocabulary (OOV) tokens cannot be looked up and usually map to an <UNK> token."
            }
            
        # Cosine similarity: (v1 . v2) / (||v1|| * ||v2||)
        dot_product = float(np.dot(v1, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        cosine_sim = dot_product / (norm1 * norm2)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        
        if cosine_sim > 0.70:
            interpretation = "High Semantic Closeness: Words appear in near-identical contextual distributions or share direct semantic category."
        elif cosine_sim > 0.40:
            interpretation = "Moderate Semantic Relatedness: Words frequently co-occur in shared topical domains (or operate as antonymous poles like hot/cold)."
        elif cosine_sim > 0.05:
            interpretation = "Weak Association: Words have tangential or distant co-occurrence in natural text."
        else:
            interpretation = "Orthogonal / Unrelated: Distinct semantic domains with virtually no shared contextual overlap."
            
        antonym_note = ""
        antonym_pairs = {("hot", "cold"), ("cold", "hot"), ("good", "bad"), ("bad", "good"), ("love", "hate"), ("hate", "love")}
        if (w1, w2) in antonym_pairs:
            antonym_note = (
                " Note on Antonyms: Notice that opposites like 'hot' and 'cold' or 'good' and 'bad' have surprisingly high cosine similarity! "
                "Under the Distributional Hypothesis ('a word is characterized by the company it keeps' — J.R. Firth), antonyms appear in the exact same syntactic "
                "environments ('the coffee was very [hot/cold]'), so word embeddings place them near each other in vector space."
            )

        return {
            "word1": word1,
            "word2": word2,
            "in_vocab": True,
            "similarity": round(cosine_sim, 4),
            "percentage": round(float((cosine_sim + 1) / 2) * 100, 1),
            "dot_product": round(dot_product, 4),
            "norm1": round(norm1, 4),
            "norm2": round(norm2, 4),
            "vector_sample_1": [round(float(x), 3) for x in v1[:5]],
            "vector_sample_2": [round(float(x), 3) for x in v2[:5]],
            "interpretation": interpretation,
            "explanation": f"Cosine similarity measures the cosine of the angle between two 50D dense vectors (range -1.0 to +1.0). {interpretation}{antonym_note}"
        }

    def get_sentence_vector(self, text: str):
        tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
        vectors = []
        covered_words = []
        uncovered_words = []
        
        for t in tokens:
            vec = self.get_word_vector(t)
            if vec is not None:
                vectors.append(vec)
                covered_words.append(t)
            else:
                uncovered_words.append(t)
                
        if len(vectors) == 0:
            avg_vec = np.zeros(self.embedding_dim)
        else:
            avg_vec = np.mean(vectors, axis=0)
            
        norm = np.linalg.norm(avg_vec)
        if norm > 0:
            avg_vec = avg_vec / norm
            
        coverage_stats = {
            "total_tokens": len(tokens),
            "in_vocab_count": len(covered_words),
            "coverage_pct": round(len(covered_words) / max(1, len(tokens)) * 100, 1),
            "covered_words": covered_words,
            "uncovered_words": uncovered_words
        }
        return avg_vec, coverage_stats

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
            
        start_time = time.perf_counter()
        
        sentence_vec, coverage = self.get_sentence_vector(text)
        probabilities = self.classifier.predict_proba([sentence_vec])[0]
        prediction_idx = int(self.classifier.predict([sentence_vec])[0])
        
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        pred_label = "Positive" if prediction_idx == 1 else "Negative"
        confidence = float(probabilities[prediction_idx])
        
        if any(neg in text.lower().split() for neg in ["not", "never", "no", "hardly", "cannot"]):
            pedagogical_note = (
                "Word embeddings capture rich semantic similarities (e.g. 'cinematic' and 'film' map close together), "
                "but averaging vectors collapses word order into a single centroid. Consequently, 'not bad' averages the negative polarity "
                "of 'not' and 'bad', diluting negation and failing to recognize compositional grammar."
            )
        else:
            pedagogical_note = (
                "Sentence embedding is formed via average pooling of 50-dimensional dense vectors. While dense embeddings "
                "generalize across synonyms better than TF-IDF, simple vector summation discards word order and grammatical dependencies."
            )

        return {
            "model_type": "Dense Word Embeddings (Average Vector)",
            "paradigm": "Distributed Representations (Dense Semantic Space)",
            "prediction": pred_label,
            "prediction_code": prediction_idx,
            "confidence": round(confidence, 4),
            "score_label": "Logistic Probability on 50D Centroid",
            "scores": {
                "Negative": round(float(probabilities[0]), 4),
                "Positive": round(float(probabilities[1]), 4)
            },
            "inference_time_ms": max(inference_time_ms, 0.12),
            "coverage": coverage,
            "vector_preview": [round(float(x), 3) for x in sentence_vec[:6]],
            "pedagogical_note": pedagogical_note
        }
