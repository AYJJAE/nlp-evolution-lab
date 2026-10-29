"""
Benchmark Runner & Evaluation Suite
Module 1: Evolution of NLP

Evaluates TF-IDF, Word Embeddings, and Transformer on held-out evaluation examples.
Strict separation of train and test sets.
Computes accuracy, latency, and detailed failure analysis.
"""

import io
import csv
import time
from models.benchmark_data import EVALUATION_DATA

def run_benchmark(tfidf_model, embedding_model, transformer_model):
    """
    Evaluates all 3 models on the identical held-out test split.
    Returns quantitative metrics and qualitative misclassification audits.
    """
    results = {
        "tfidf": {"correct": 0, "total": len(EVALUATION_DATA), "total_time": 0.0, "predictions": []},
        "embedding": {"correct": 0, "total": len(EVALUATION_DATA), "total_time": 0.0, "predictions": []},
        "transformer": {"correct": 0, "total": len(EVALUATION_DATA), "total_time": 0.0, "predictions": []}
    }
    
    detailed_rows = []

    for item in EVALUATION_DATA:
        text = item["text"]
        true_label = item["label"]
        true_str = "Positive" if true_label == 1 else "Negative"
        category = item["category"]
        
        # 1. TF-IDF
        t0 = time.perf_counter()
        pred_tfidf = tfidf_model.predict(text)
        t_tfidf = (time.perf_counter() - t0) * 1000
        results["tfidf"]["total_time"] += t_tfidf
        tfidf_correct = (pred_tfidf["prediction_code"] == true_label)
        if tfidf_correct:
            results["tfidf"]["correct"] += 1
            
        # 2. Embedding
        t0 = time.perf_counter()
        pred_emb = embedding_model.predict(text)
        t_emb = (time.perf_counter() - t0) * 1000
        results["embedding"]["total_time"] += t_emb
        emb_correct = (pred_emb["prediction_code"] == true_label)
        if emb_correct:
            results["embedding"]["correct"] += 1
            
        # 3. Transformer
        t0 = time.perf_counter()
        pred_tr = transformer_model.predict(text)
        t_tr = (time.perf_counter() - t0) * 1000
        results["transformer"]["total_time"] += t_tr
        tr_correct = (pred_tr["prediction_code"] == true_label)
        if tr_correct:
            results["transformer"]["correct"] += 1
            
        detailed_rows.append({
            "id": item["id"],
            "text": text,
            "category": category,
            "ground_truth": true_str,
            "ground_truth_code": true_label,
            "rationale": item["rationale"],
            "tfidf": {
                "prediction": pred_tfidf["prediction"],
                "confidence": pred_tfidf["confidence"],
                "correct": tfidf_correct,
                "latency_ms": round(t_tfidf, 2)
            },
            "embedding": {
                "prediction": pred_emb["prediction"],
                "confidence": pred_emb["confidence"],
                "correct": emb_correct,
                "latency_ms": round(t_emb, 2)
            },
            "transformer": {
                "prediction": pred_tr["prediction"],
                "confidence": pred_tr["confidence"],
                "correct": tr_correct,
                "latency_ms": round(t_tr, 2)
            }
        })

    total = len(EVALUATION_DATA)
    summary = {
        "total_samples": total,
        "models": {
            "tfidf": {
                "name": "TF-IDF + Logistic Regression",
                "paradigm": "Bag-of-Words (Sparse Count)",
                "accuracy": round((results["tfidf"]["correct"] / total) * 100, 1),
                "correct_count": results["tfidf"]["correct"],
                "error_count": total - results["tfidf"]["correct"],
                "avg_latency_ms": round(results["tfidf"]["total_time"] / total, 2)
            },
            "embedding": {
                "name": "Word Embeddings (Average Vector)",
                "paradigm": "Distributed Representations (Dense Space)",
                "accuracy": round((results["embedding"]["correct"] / total) * 100, 1),
                "correct_count": results["embedding"]["correct"],
                "error_count": total - results["embedding"]["correct"],
                "avg_latency_ms": round(results["embedding"]["total_time"] / total, 2)
            },
            "transformer": {
                "name": "PyTorch Self-Attention Transformer",
                "paradigm": "Contextualized Multi-Head Self-Attention",
                "accuracy": round((results["transformer"]["correct"] / total) * 100, 1),
                "correct_count": results["transformer"]["correct"],
                "error_count": total - results["transformer"]["correct"],
                "avg_latency_ms": round(results["transformer"]["total_time"] / total, 2)
            }
        },
        "detailed_rows": detailed_rows,
        "pedagogical_disclaimer": (
            "Scientific & Methodological Caveat: This held-out evaluation suite (20 curated sentences) highlights "
            "structural architectural differences—especially on negation inversion and contrastive clauses—where "
            "Bag of Words and static embeddings systematically stumble. However, a tiny dataset cannot establish "
            "general statistical model superiority across broader distribution shifts. Real-world evaluation requires "
            "massive test suites (e.g. GLUE, SST-5, IMDB) with cross-validation and significance testing."
        )
    }
    return summary

def generate_benchmark_csv(summary: dict) -> str:
    """Generates RFC 4180 compliant CSV string of the benchmark results."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "ID",
        "Sentence",
        "Category",
        "Ground_Truth",
        "TFIDF_Prediction",
        "TFIDF_Confidence",
        "TFIDF_Correct",
        "Embedding_Prediction",
        "Embedding_Confidence",
        "Embedding_Correct",
        "Transformer_Prediction",
        "Transformer_Confidence",
        "Transformer_Correct",
        "Linguistic_Rationale"
    ])
    
    for row in summary["detailed_rows"]:
        writer.writerow([
            row["id"],
            row["text"],
            row["category"],
            row["ground_truth"],
            row["tfidf"]["prediction"],
            row["tfidf"]["confidence"],
            "PASS" if row["tfidf"]["correct"] else "FAIL",
            row["embedding"]["prediction"],
            row["embedding"]["confidence"],
            "PASS" if row["embedding"]["correct"] else "FAIL",
            row["transformer"]["prediction"],
            row["transformer"]["confidence"],
            "PASS" if row["transformer"]["correct"] else "FAIL",
            row["rationale"]
        ])
        
    return output.getvalue()

