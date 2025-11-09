"""
Text Classification Experiment

This script demonstrates text classification using zero-shot classification
with transformers, which can classify text into categories without training.
"""

from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class TextClassifier:
    """
    A text classifier using zero-shot classification with transformers.
    """
    
    def __init__(self):
        """
        Initialize the zero-shot classifier.
        """
        print("Loading zero-shot classification model...")
        self.classifier = pipeline("zero-shot-classification", 
                                   model="facebook/bart-large-mnli")
        print("Model loaded successfully!")
    
    def classify(self, text, candidate_labels):
        """
        Classify text into one of the candidate labels.
        
        Args:
            text (str): Input text to classify
            candidate_labels (list): List of possible categories
            
        Returns:
            dict: Classification results with labels and scores
        """
        result = self.classifier(text, candidate_labels)
        return result
    
    def print_result(self, text, result):
        """
        Pretty print the classification result.
        
        Args:
            text (str): Original input text
            result (dict): Classification result
        """
        print(f"\nText: '{text[:80]}{'...' if len(text) > 80 else ''}'")
        print(f"Top prediction: {result['labels'][0]} (confidence: {result['scores'][0]:.4f})")
        print("All predictions:")
        for label, score in zip(result['labels'], result['scores']):
            bar = "█" * int(score * 30)
            print(f"  {label:15s}: {bar} {score:.4f}")
        print("-" * 70)


def main():
    """
    Main function to demonstrate text classification.
    """
    # Initialize classifier
    classifier = TextClassifier()
    
    # Define categories for classification
    categories = ["technology", "sports", "politics", "entertainment", "science"]
    
    # Example texts from different domains
    example_texts = [
        "Apple announced the new iPhone 15 with improved camera features and longer battery life.",
        "The Lakers won the championship game with a score of 102-98 in overtime.",
        "The Senate passed a new bill regarding climate change policy with bipartisan support.",
        "The latest Marvel movie broke box office records, earning over $500 million in its opening weekend.",
        "Researchers discovered a new species of deep-sea fish in the Mariana Trench.",
        "Google released an update to its search algorithm to improve result accuracy.",
        "The Olympic games will be held in Paris next summer with athletes from around the world.",
        "Scientists successfully tested a new vaccine that shows promising results against malaria.",
    ]
    
    print("\n" + "=" * 70)
    print("TEXT CLASSIFICATION DEMONSTRATION")
    print("=" * 70)
    print(f"\nCategories: {', '.join(categories)}\n")
    
    # Classify each text
    for text in example_texts:
        result = classifier.classify(text, categories)
        classifier.print_result(text, result)
    
    # Custom classification with different categories
    print("\n" + "=" * 70)
    print("CUSTOM CATEGORIES EXAMPLE")
    print("=" * 70)
    
    custom_text = "The new restaurant downtown serves amazing Italian cuisine with fresh pasta."
    custom_categories = ["food", "travel", "health", "business", "education"]
    
    result = classifier.classify(custom_text, custom_categories)
    classifier.print_result(custom_text, result)


if __name__ == "__main__":
    main()
