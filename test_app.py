"""
Integration Test Suite for NLP Evolution Lab
Validates all endpoints, real model outputs, and edge cases.
"""

import unittest
import json
from server import app

class TestNLPEvolutionLab(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NLP Evolution Lab", response.data)
        self.assertIn(b"Text Playground", response.data)

    def test_sentiment_prediction_positive(self):
        payload = {"text": "The acting was phenomenal and the storyline was truly inspiring."}
        response = self.client.post("/api/sentiment/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("tfidf", data["results"])
        self.assertIn("embedding", data["results"])
        self.assertIn("transformer", data["results"])
        self.assertGreater(data["results"]["tfidf"]["inference_time_ms"], 0)
        self.assertGreater(data["results"]["transformer"]["inference_time_ms"], 0)

    def test_sentiment_negation(self):
        payload = {"text": "This film was not bad at all, in fact I quite liked it."}
        response = self.client.post("/api/sentiment/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])

    def test_empty_input_handling(self):
        payload = {"text": "   "}
        response = self.client.post("/api/sentiment/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("empty", data["error"])

    def test_word_similarity_in_vocab(self):
        payload = {"word1": "king", "word2": "queen"}
        response = self.client.post("/api/similarity", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["data"]["in_vocab"])
        self.assertGreater(data["data"]["similarity"], 0.5)

    def test_word_similarity_antonyms(self):
        payload = {"word1": "hot", "word2": "cold"}
        response = self.client.post("/api/similarity", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("Antonyms", data["data"]["explanation"])

    def test_word_similarity_oov(self):
        payload = {"word1": "movie", "word2": "xyzrandomword123"}
        response = self.client.post("/api/similarity", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertFalse(data["data"]["in_vocab"])
        self.assertIn("outside vocabulary", data["data"]["error_message"])

    def test_attention_real_mode(self):
        payload = {"text": "This movie was not bad", "mode": "real", "layer": 0, "head": 0}
        response = self.client.post("/api/attention", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "real")
        self.assertIn("attention_matrix", data)
        self.assertEqual(len(data["attention_matrix"]), len(data["tokens"]))

    def test_attention_illustrative_mode(self):
        payload = {"text": "This movie was not bad", "mode": "illustrative", "layer": 0, "head": 1}
        response = self.client.post("/api/attention", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["mode"], "illustrative")
        self.assertEqual(data["label"], "Illustrative Demo")

    def test_benchmark_endpoint(self):
        response = self.client.get("/api/benchmark")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("models", data["benchmark"])
        self.assertIn("detailed_rows", data["benchmark"])
        self.assertEqual(len(data["benchmark"]["detailed_rows"]), 20)

    def test_export_csv(self):
        response = self.client.get("/api/export-benchmark-csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn(b"Sentence,Category,Ground_Truth", response.data)

    def test_demo_scenarios(self):
        response = self.client.get("/api/demo-scenarios")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["scenarios"]), 3)

if __name__ == "__main__":
    unittest.main()

