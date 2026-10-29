"""
NLP Evolution Lab - Backend Application Server
Module 1: Evolution of NLP (Bag of Words -> Word Embeddings -> Transformers)
Flask REST API & Static Asset Delivery
"""

import os
import time
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

from models.tfidf_model import TFIDFClassifier
from models.embedding_model import WordEmbeddingModel
from models.transformer_model import RealTransformerClassifier
from models.benchmark_runner import run_benchmark, generate_benchmark_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

# Initialize NLP Models
print(">> Initializing NLP Evolution Lab Models...")
t0 = time.time()
tfidf_model = TFIDFClassifier()
embedding_model = WordEmbeddingModel()
transformer_model = RealTransformerClassifier()
print(f">> All 3 NLP models successfully initialized in {round(time.time() - t0, 2)}s.")

# Precompute benchmark results on boot for instant UI load
print(">> Precomputing held-out evaluation benchmark...")
benchmark_cache = run_benchmark(tfidf_model, embedding_model, transformer_model)
print(">> Benchmark suite ready.")

# Curated Illustrative Attention Examples for teaching
ILLUSTRATIVE_ATTENTION_PRESETS = {
    "negation": {
        "text": "This movie was not bad at all",
        "tokens": ["[CLS]", "this", "movie", "was", "not", "bad", "at", "all", "[SEP]"],
        "layers": [
            {
                "layer_index": 0,
                "layer_name": "Layer 1 (Lexical & Local Syntactic Alignment)",
                "heads": [
                    {
                        "head_index": 0,
                        "head_name": "Head 1: Sequential Positional Focus",
                        "description": "Each token attends primarily to adjacent neighboring tokens (i-1 and i+1).",
                        "matrix": [
                            [0.70, 0.15, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01],
                            [0.10, 0.60, 0.20, 0.04, 0.02, 0.02, 0.01, 0.00, 0.01],
                            [0.05, 0.25, 0.50, 0.15, 0.02, 0.01, 0.01, 0.00, 0.01],
                            [0.02, 0.04, 0.14, 0.55, 0.20, 0.03, 0.01, 0.00, 0.01],
                            [0.01, 0.02, 0.02, 0.15, 0.40, 0.35, 0.03, 0.01, 0.01],
                            [0.01, 0.01, 0.02, 0.05, 0.35, 0.45, 0.08, 0.02, 0.01],
                            [0.01, 0.01, 0.01, 0.02, 0.05, 0.15, 0.50, 0.23, 0.02],
                            [0.01, 0.01, 0.01, 0.01, 0.02, 0.06, 0.28, 0.55, 0.05],
                            [0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.80]
                        ]
                    },
                    {
                        "head_index": 1,
                        "head_name": "Head 2: Negation Inversion Tracker",
                        "description": "High cross-attention between negation particle 'not' and target adjective 'bad'.",
                        "matrix": [
                            [0.45, 0.05, 0.05, 0.05, 0.15, 0.15, 0.03, 0.02, 0.05],
                            [0.05, 0.30, 0.45, 0.05, 0.05, 0.05, 0.02, 0.01, 0.02],
                            [0.04, 0.20, 0.40, 0.10, 0.10, 0.12, 0.02, 0.01, 0.01],
                            [0.02, 0.05, 0.10, 0.30, 0.25, 0.22, 0.03, 0.01, 0.02],
                            [0.02, 0.02, 0.04, 0.07, 0.15, 0.62, 0.04, 0.03, 0.01],
                            [0.02, 0.02, 0.05, 0.05, 0.65, 0.16, 0.02, 0.01, 0.02],
                            [0.02, 0.02, 0.02, 0.03, 0.08, 0.22, 0.35, 0.24, 0.02],
                            [0.02, 0.02, 0.02, 0.02, 0.05, 0.18, 0.25, 0.42, 0.02],
                            [0.10, 0.03, 0.03, 0.04, 0.18, 0.22, 0.05, 0.05, 0.30]
                        ]
                    }
                ]
            },
            {
                "layer_index": 1,
                "layer_name": "Layer 2 (Semantic Synthesis & Classification Polling)",
                "heads": [
                    {
                        "head_index": 0,
                        "head_name": "Head 1: [CLS] Global Aggregation",
                        "description": "[CLS] token queries informative phrase boundaries ('not bad', 'movie') to form sentence representation.",
                        "matrix": [
                            [0.12, 0.05, 0.18, 0.05, 0.28, 0.24, 0.03, 0.02, 0.03],
                            [0.05, 0.15, 0.45, 0.05, 0.15, 0.10, 0.02, 0.01, 0.02],
                            [0.08, 0.12, 0.35, 0.08, 0.18, 0.14, 0.02, 0.01, 0.02],
                            [0.05, 0.05, 0.12, 0.20, 0.28, 0.24, 0.03, 0.01, 0.02],
                            [0.04, 0.02, 0.08, 0.06, 0.22, 0.52, 0.03, 0.02, 0.01],
                            [0.05, 0.03, 0.07, 0.05, 0.55, 0.20, 0.02, 0.01, 0.02],
                            [0.03, 0.02, 0.05, 0.04, 0.12, 0.24, 0.25, 0.22, 0.03],
                            [0.03, 0.02, 0.04, 0.03, 0.10, 0.22, 0.20, 0.33, 0.03],
                            [0.15, 0.05, 0.12, 0.05, 0.25, 0.25, 0.04, 0.04, 0.05]
                        ]
                    }
                ]
            }
        ]
    }
}

# ----------------- ROUTES ----------------- #

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route("/api/sentiment/predict", methods=["POST"])
def predict_sentiment():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({
                "success": False,
                "error": "Input sentence is empty. Please enter at least one word."
            }), 400
            
        if len(text) > 800:
            return jsonify({
                "success": False,
                "error": "Input sentence is too long (maximum 800 characters for laboratory evaluation)."
            }), 400

        # Run real models
        tfidf_res = tfidf_model.predict(text)
        emb_res = embedding_model.predict(text)
        tr_res = transformer_model.predict(text)
        
        return jsonify({
            "success": True,
            "sentence": text,
            "results": {
                "tfidf": tfidf_res,
                "embedding": emb_res,
                "transformer": tr_res
            },
            "comparability_disclaimer": (
                "⚠️ Score Comparability Note: Model scores originate from disparate mathematical frameworks "
                "(TF-IDF calibrated Logistic probability vs. CBOW Centroid Logistic probability vs. Transformer Softmax output). "
                "Because they are not identically calibrated, numeric scores reflect intra-model confidence and cannot be "
                "directly compared as equal probabilities across architectures."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/similarity", methods=["POST"])
def word_similarity():
    try:
        data = request.get_json(force=True)
        w1 = data.get("word1", "").strip()
        w2 = data.get("word2", "").strip()
        
        if not w1 or not w2:
            return jsonify({
                "success": False,
                "error": "Please provide both Word 1 and Word 2 for similarity calculation."
            }), 400
            
        sim_data = embedding_model.compute_similarity(w1, w2)
        return jsonify({"success": True, "data": sim_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/attention", methods=["POST"])
def attention_explorer():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "This movie was not bad at all").strip()
        mode = data.get("mode", "real") # "real" or "illustrative"
        layer = int(data.get("layer", 0))
        head = int(data.get("head", 0))

        if mode == "illustrative":
            preset = ILLUSTRATIVE_ATTENTION_PRESETS["negation"]
            num_layers = len(preset["layers"])
            selected_layer = min(layer, num_layers - 1)
            layer_obj = preset["layers"][selected_layer]
            selected_head = min(head, len(layer_obj["heads"]) - 1)
            head_obj = layer_obj["heads"][selected_head]
            
            return jsonify({
                "success": True,
                "mode": "illustrative",
                "label": "Illustrative Demo",
                "tokens": preset["tokens"],
                "layer": selected_layer,
                "head": selected_head,
                "layer_name": layer_obj["layer_name"],
                "head_name": head_obj["head_name"],
                "description": head_obj["description"],
                "attention_matrix": head_obj["matrix"],
                "num_layers": num_layers,
                "num_heads": len(layer_obj["heads"]),
                "pedagogical_caveat": (
                    "Important Research Caveat (Jain & Wallace 2019 / Wiegreffe & Pinter 2019): "
                    "Attention weights visualize the distribution of intermediate information flow between tokens, "
                    "but attention weights do not constitute a complete or faithful causal explanation of the final classification decision."
                )
            })

        # Real Transformer Attention Computation
        tr_res = transformer_model.predict(text)
        tokens = tr_res["tokens"]
        num_layers = tr_res["num_layers"]
        num_heads = tr_res["num_heads"]
        
        clamped_layer = max(0, min(layer, num_layers - 1))
        clamped_head = max(0, min(head, num_heads - 1))
        
        matrix = tr_res["attentions"][clamped_layer][clamped_head]

        return jsonify({
            "success": True,
            "mode": "real",
            "label": "Live PyTorch Computed Attention",
            "tokens": tokens,
            "layer": clamped_layer,
            "head": clamped_head,
            "layer_name": f"Encoder Layer {clamped_layer + 1} of {num_layers}",
            "head_name": f"Self-Attention Head {clamped_head + 1} of {num_heads}",
            "description": f"Genuine softmax attention matrix A = Softmax((Q*K^T)/sqrt({transformer_model.d_model // num_heads})) computed via PyTorch forward pass.",
            "attention_matrix": matrix,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "pedagogical_caveat": (
                "Important Research Caveat: Attention weights show where the self-attention mechanism "
                "routed intermediate representations, but attention weights alone do not fully explain or justify a model's final decision."
            )
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    try:
        return jsonify({
            "success": True,
            "benchmark": benchmark_cache
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/export-benchmark-csv", methods=["GET"])
def export_benchmark_csv():
    try:
        csv_content = generate_benchmark_csv(benchmark_cache)
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=nlp_evolution_lab_benchmark.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/demo-scenarios", methods=["GET"])
def get_demo_scenarios():
    scenarios = [
        {
            "id": "scenario-1",
            "title": "Scenario 1: The Curse of Bag-of-Words & Negation Failure",
            "sentence": "This movie was not bad at all, in fact I quite liked it.",
            "learning_objective": "Demonstrate why BoW / TF-IDF collapses when polarity words are syntactically negated.",
            "walkthrough": [
                "1. Enter the sentence in the Text Playground.",
                "2. Observe TF-IDF prediction: TF-IDF extracts 'bad' as a dominant term and predicts Negative (or low confidence).",
                "3. Observe Word Embeddings: Average vector pools 'not' and 'bad', diluting the positive sentiment.",
                "4. Observe Transformer: Predicts Positive because self-attention links 'not' directly to 'bad' to invert polarity.",
                "5. Open Attention Explorer to inspect Head 2 attending between 'not' and 'bad'."
            ]
        },
        {
            "id": "scenario-2",
            "title": "Scenario 2: The Distributional Hypothesis & Antonym Paradox",
            "word_pair": ["hot", "cold"],
            "learning_objective": "Explore why antonyms possess high cosine similarity in static word embedding spaces.",
            "walkthrough": [
                "1. Navigate to Word Similarity panel and test 'hot' vs 'cold'.",
                "2. Observe a relatively high cosine similarity (~0.62) despite them having opposite semantic definitions.",
                "3. Explain the Distributional Hypothesis: In natural language corpora, antonyms share near-identical syntactic frames ('the room is very [hot/cold]').",
                "4. Contrast this with 'king' vs 'queen' (analogous semantic shift) and 'banana' vs 'quantum' (orthogonal, near-zero similarity)."
            ]
        },
        {
            "id": "scenario-3",
            "title": "Scenario 3: Contrastive Clauses & Multi-Token Discourse",
            "sentence": "The movie has stunning visuals, but the story is completely hollow.",
            "learning_objective": "Illustrate how discourse markers ('but', 'however') shift sentiment weight in hierarchical language.",
            "walkthrough": [
                "1. Enter the sentence in Text Playground.",
                "2. TF-IDF sees both 'stunning' (positive) and 'hollow' (negative), resulting in ambiguous conflict.",
                "3. Average embeddings blend positive visuals with negative storyline into a diluted centroid.",
                "4. The Transformer correctly prioritizes the adversative 'but' clause, classifying the overall sentiment as Negative.",
                "5. Check Results panel error analysis table to observe this failure pattern across the held-out evaluation set."
            ]
        }
    ]
    return jsonify({"success": True, "scenarios": scenarios})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f">> Starting NLP Evolution Lab Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

