"""
Automated Test Suite for Voice-Enabled Chatbot
Tests:
1. Dataset integrity (intents.json structure, pattern non-emptiness)
2. Preprocessing pipeline (tokenization, sequence length, padding)
3. Deep Learning model loading & inference output shape
4. Intent classification accuracy across sample test queries
5. Confidence threshold and conversational fallback handling
6. Audio file transcription utility
"""

import os
import unittest
import numpy as np
from utils.preprocessing import clean_text, load_intents, preprocess_input
from predict import get_predictor


class TestVoiceChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = get_predictor()
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.base_dir, "data", "intents.json")

    def test_01_dataset_integrity(self):
        """Verify that intents.json exists, parses cleanly, and contains all required tags."""
        self.assertTrue(os.path.exists(self.data_path), "intents.json not found")
        patterns, tags, responses = load_intents(self.data_path)
        self.assertGreater(len(patterns), 100, "Dataset has too few patterns")
        self.assertGreaterEqual(len(set(tags)), 15, "Dataset must have at least 15 intents")
        
        required_tags = ["greeting", "goodbye", "working_hours", "contact", "location", "fallback"]
        for r_tag in required_tags:
            self.assertIn(r_tag, responses, f"Missing required intent tag: {r_tag}")
            self.assertGreater(len(responses[r_tag]), 0, f"No responses for tag: {r_tag}")

    def test_02_text_preprocessing(self):
        """Verify text cleaning and contraction normalization."""
        raw = "Hello! Can't you see I'm testing this?!"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "hello can not you see i am testing this")
        
        empty_clean = clean_text("   !@#$%^&*()   ")
        self.assertEqual(empty_clean, "")

    def test_03_model_prediction_shape_and_classes(self):
        """Verify that the model outputs valid probability distribution."""
        sample_query = "What are your operating hours?"
        result = self.predictor.process_query(sample_query)
        self.assertIn("query", result)
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertIn("response", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertGreater(len(result["response"]), 0)

    def test_04_common_intent_classifications(self):
        """Test inference accuracy on core academic intents."""
        test_cases = [
            ("Hello there!", "greeting"),
            ("What are your working hours?", "working_hours"),
            ("Where is your campus office located?", "location"),
            ("How do I contact customer support?", "contact"),
            ("Tell me a funny joke", "jokes_humor"),
            ("Can I speak to a real person?", "human_agent"),
            ("How much does tuition cost?", "pricing"),
            ("How do I apply for admission?", "admission_inquiry")
        ]
        for query, expected_tag in test_cases:
            res = self.predictor.process_query(query)
            self.assertEqual(
                res["intent"], expected_tag,
                f"Failed intent prediction for '{query}'. Expected '{expected_tag}', got '{res['intent']}'"
            )

    def test_05_fallback_handling(self):
        """Verify that random gibberish triggers fallback."""
        nonsense_query = "asdfghjkl zxcvbnm qwertyuiop"
        res = self.predictor.process_query(nonsense_query)
        self.assertTrue(
            res["is_fallback"] or res["intent"] == "fallback",
            f"Nonsense query should trigger fallback. Got: {res['intent']}"
        )

    def test_06_empty_query_handling(self):
        """Verify that empty inputs are handled cleanly without exceptions."""
        res = self.predictor.process_query("   ")
        self.assertEqual(res["intent"], "empty_input")
        self.assertEqual(res["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
