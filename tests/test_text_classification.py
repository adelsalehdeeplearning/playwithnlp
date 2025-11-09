"""Unit tests for text classification module."""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_classification import TextClassifier, DataLoader, TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor."""
    
    def test_lowercase(self):
        """Test lowercase conversion."""
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
        result = preprocessor.preprocess("HELLO World")
        self.assertEqual(result, "hello world")
    
    def test_remove_punctuation(self):
        """Test punctuation removal."""
        preprocessor = TextPreprocessor(lowercase=False, remove_punctuation=True)
        result = preprocessor.preprocess("Hello, World!")
        self.assertEqual(result, "Hello World")
    
    def test_remove_extra_spaces(self):
        """Test extra space removal."""
        preprocessor = TextPreprocessor(lowercase=False, remove_punctuation=False, 
                                       remove_extra_spaces=True)
        result = preprocessor.preprocess("Hello    World   ")
        self.assertEqual(result, "Hello World")
    
    def test_list_processing(self):
        """Test processing a list of texts."""
        preprocessor = TextPreprocessor()
        texts = ["HELLO!", "WORLD!"]
        results = preprocessor.preprocess(texts)
        self.assertEqual(results, ["hello", "world"])


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader."""
    
    def test_load_from_lists(self):
        """Test loading data from lists."""
        loader = DataLoader()
        texts = ["text1", "text2"]
        labels = ["label1", "label2"]
        X, y = loader.load_from_lists(texts, labels)
        self.assertEqual(X, texts)
        self.assertEqual(y, labels)
    
    def test_train_test_split(self):
        """Test train-test split."""
        loader = DataLoader()
        texts = ["text1", "text2", "text3", "text4"]
        labels = ["label1", "label2", "label1", "label2"]
        loader.load_from_lists(texts, labels)
        X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.25)
        self.assertEqual(len(X_train) + len(X_test), 4)
        self.assertEqual(len(y_train) + len(y_test), 4)


class TestTextClassifier(unittest.TestCase):
    """Test cases for TextClassifier."""
    
    def setUp(self):
        """Set up test data."""
        self.texts = [
            "this is positive", "this is negative",
            "very positive", "very negative",
            "positive example", "negative example"
        ]
        self.labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = TextClassifier(model_type='naive_bayes')
        self.assertEqual(classifier.model_type, 'naive_bayes')
        self.assertFalse(classifier.is_trained)
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with self.assertRaises(ValueError):
            TextClassifier(model_type='invalid_model')
    
    def test_train_and_predict(self):
        """Test training and prediction."""
        classifier = TextClassifier(model_type='naive_bayes')
        train_info = classifier.train(self.texts, self.labels)
        
        self.assertTrue(classifier.is_trained)
        self.assertIn('train_accuracy', train_info)
        self.assertGreater(train_info['train_accuracy'], 0)
        
        predictions = classifier.predict(["positive example", "negative example"])
        self.assertEqual(len(predictions), 2)
    
    def test_predict_before_training(self):
        """Test prediction before training raises error."""
        classifier = TextClassifier(model_type='naive_bayes')
        with self.assertRaises(ValueError):
            classifier.predict(["test"])
    
    def test_evaluate(self):
        """Test model evaluation."""
        classifier = TextClassifier(model_type='naive_bayes')
        classifier.train(self.texts, self.labels)
        
        results = classifier.evaluate(self.texts[:2], self.labels[:2])
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        self.assertIn('confusion_matrix', results)


if __name__ == '__main__':
    unittest.main()
