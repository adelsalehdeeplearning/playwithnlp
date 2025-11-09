"""Integration tests for the complete text classification workflow."""

import unittest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_classification import TextClassifier, DataLoader, TextPreprocessor


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Set up test data."""
        self.texts = [
            "Great product, highly recommended!",
            "Terrible quality, very disappointed.",
            "Excellent service and fast delivery.",
            "Poor performance, not worth the money.",
            "Amazing quality, exceeded expectations!",
            "Waste of money, complete failure.",
            "Fantastic experience, will buy again.",
            "Horrible service, never coming back.",
        ]
        self.labels = ["positive", "negative", "positive", "negative",
                      "positive", "negative", "positive", "negative"]
    
    def test_full_workflow(self):
        """Test complete classification workflow."""
        # 1. Preprocess
        preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
        clean_texts = preprocessor.preprocess(self.texts)
        
        # 2. Load and split data
        data_loader = DataLoader()
        data_loader.load_from_lists(clean_texts, self.labels)
        X_train, X_test, y_train, y_test = data_loader.train_test_split(test_size=0.25)
        
        # 3. Train classifier
        classifier = TextClassifier(model_type='naive_bayes')
        train_info = classifier.train(X_train, y_train)
        
        # Verify training
        self.assertTrue(classifier.is_trained)
        self.assertIn('train_accuracy', train_info)
        self.assertGreater(train_info['train_accuracy'], 0)
        
        # 4. Make predictions
        predictions = classifier.predict(X_test)
        self.assertEqual(len(predictions), len(X_test))
        
        # 5. Evaluate
        results = classifier.evaluate(X_test, y_test)
        self.assertIn('accuracy', results)
        self.assertIn('classification_report', results)
        
        # 6. Save and load model
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            classifier.save(temp_path)
            
            # Create new classifier and load
            new_classifier = TextClassifier()
            new_classifier.load(temp_path)
            
            # Verify loaded model works
            new_predictions = new_classifier.predict(X_test)
            self.assertEqual(predictions, new_predictions)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_all_model_types(self):
        """Test all supported model types."""
        preprocessor = TextPreprocessor()
        clean_texts = preprocessor.preprocess(self.texts)
        
        data_loader = DataLoader()
        data_loader.load_from_lists(clean_texts, self.labels)
        X_train, X_test, y_train, y_test = data_loader.train_test_split(test_size=0.25)
        
        for model_type in ['naive_bayes', 'logistic_regression', 'svm']:
            with self.subTest(model_type=model_type):
                classifier = TextClassifier(model_type=model_type)
                train_info = classifier.train(X_train, y_train)
                
                self.assertTrue(classifier.is_trained)
                self.assertGreater(train_info['train_accuracy'], 0)
                
                predictions = classifier.predict(X_test)
                self.assertEqual(len(predictions), len(X_test))


if __name__ == '__main__':
    unittest.main()
