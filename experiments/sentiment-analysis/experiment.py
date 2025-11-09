"""
Sentiment Analysis Experiment

This script demonstrates sentiment analysis using a pre-trained transformer model
from Hugging Face. It classifies text as positive or negative sentiment.
"""

from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    A simple sentiment analyzer using pre-trained transformers.
    """
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model from Hugging Face
        """
        print(f"Loading model: {model_name}...")
        self.classifier = pipeline("sentiment-analysis", model=model_name)
        print("Model loaded successfully!")
    
    def analyze(self, text):
        """
        Analyze the sentiment of a given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Dictionary containing label and score
        """
        result = self.classifier(text)[0]
        return result
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of dictionaries containing labels and scores
        """
        results = self.classifier(texts)
        return results
    
    def print_result(self, text, result):
        """
        Pretty print the sentiment analysis result.
        
        Args:
            text (str): Original input text
            result (dict): Analysis result
        """
        label = result['label']
        score = result['score']
        print(f"\nText: '{text}'")
        print(f"Sentiment: {label}")
        print(f"Confidence: {score:.4f}")
        print("-" * 60)


def main():
    """
    Main function to demonstrate sentiment analysis.
    """
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Example texts to analyze
    example_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The movie was quite entertaining and well-made.",
        "I'm disappointed with the quality and service.",
        "This book changed my life! Highly recommended!",
        "Terrible customer support, very frustrating.",
        "The weather is nice today.",
        "I'm not sure how I feel about this.",
    ]
    
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Analyze each text
    for text in example_texts:
        result = analyzer.analyze(text)
        analyzer.print_result(text, result)
    
    # Batch analysis example
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS")
    print("=" * 60)
    
    batch_texts = [
        "Great product!",
        "Awful experience.",
        "It's okay."
    ]
    
    batch_results = analyzer.analyze_batch(batch_texts)
    
    for text, result in zip(batch_texts, batch_results):
        analyzer.print_result(text, result)


if __name__ == "__main__":
    main()
