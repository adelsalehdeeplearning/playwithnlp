"""Text classification model implementation."""

from typing import List, Optional, Dict, Any
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TextClassifier:
    """Text classification model using scikit-learn."""
    
    SUPPORTED_MODELS = {
        'naive_bayes': MultinomialNB,
        'logistic_regression': LogisticRegression,
        'svm': LinearSVC
    }
    
    def __init__(self, model_type: str = 'naive_bayes', 
                 max_features: int = 5000,
                 **model_kwargs):
        """
        Initialize the text classifier.
        
        Args:
            model_type: Type of model to use ('naive_bayes', 'logistic_regression', 'svm')
            max_features: Maximum number of features for TF-IDF vectorizer
            **model_kwargs: Additional keyword arguments for the model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type must be one of {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Initialize the model with default parameters
        if model_type == 'logistic_regression':
            model_kwargs.setdefault('max_iter', 1000)
            model_kwargs.setdefault('random_state', 42)
        elif model_type == 'svm':
            model_kwargs.setdefault('random_state', 42)
            model_kwargs.setdefault('max_iter', 1000)
        
        self.model = self.SUPPORTED_MODELS[model_type](**model_kwargs)
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on the given texts and labels.
        
        Args:
            texts: List of text strings for training
            labels: List of corresponding labels
            
        Returns:
            Dictionary containing training information
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Train the model
        self.model.fit(X, labels)
        self.is_trained = True
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X)
        train_accuracy = accuracy_score(labels, train_predictions)
        
        return {
            'train_accuracy': train_accuracy,
            'n_samples': len(texts),
            'n_features': X.shape[1]
        }
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for the given texts.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            List of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for the given texts.
        
        Args:
            texts: List of text strings to classify
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model type '{self.model_type}' does not support probability prediction")
        
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate the classifier on the given texts and labels.
        
        Args:
            texts: List of text strings for evaluation
            labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist()
        }
    
    def save(self, filepath: str):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_type': self.model_type,
            'max_features': self.max_features,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.model_type = model_data['model_type']
        self.max_features = model_data['max_features']
        self.is_trained = model_data['is_trained']
