"""Example usage of the text classification system."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_classification import TextClassifier, DataLoader, TextPreprocessor
import pandas as pd


def main():
    """Run a complete text classification example."""
    
    print("=" * 60)
    print("Text Classification Example")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample dataset...")
    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time and money.",
        "The product works great and exceeded my expectations.",
        "Poor quality, broke after one week.",
        "Amazing service! Will definitely come back again.",
        "Worst customer service I've ever experienced.",
        "The book was incredibly well-written and engaging.",
        "Boring and poorly written, couldn't finish it.",
        "Excellent quality for the price. Highly recommend!",
        "Not worth the money. Very disappointed.",
        "Outstanding performance! Better than I expected.",
        "Complete disaster. Would not recommend to anyone.",
        "Great experience overall. Very satisfied with my purchase.",
        "Awful. Nothing worked as advertised.",
        "Love it! Best purchase I've made this year.",
        "Horrible quality. Returned immediately.",
    ]
    
    sample_labels = [
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative"
    ]
    
    print(f"   - Total samples: {len(sample_texts)}")
    print(f"   - Classes: {set(sample_labels)}")
    
    # Initialize components
    print("\n2. Initializing components...")
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_extra_spaces=True
    )
    
    data_loader = DataLoader(text_column='text', label_column='label')
    
    # Preprocess texts
    print("\n3. Preprocessing texts...")
    preprocessed_texts = preprocessor.preprocess(sample_texts)
    print(f"   - Original: '{sample_texts[0]}'")
    print(f"   - Preprocessed: '{preprocessed_texts[0]}'")
    
    # Load data and split
    print("\n4. Splitting data into train/test sets...")
    data_loader.load_from_lists(preprocessed_texts, sample_labels)
    X_train, X_test, y_train, y_test = data_loader.train_test_split(
        test_size=0.3, random_state=42
    )
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Train different models
    models = ['naive_bayes', 'logistic_regression', 'svm']
    
    for model_type in models:
        print(f"\n5. Training {model_type} model...")
        classifier = TextClassifier(model_type=model_type, max_features=1000)
        
        train_info = classifier.train(X_train, y_train)
        print(f"   - Training accuracy: {train_info['train_accuracy']:.4f}")
        print(f"   - Number of features: {train_info['n_features']}")
        
        print(f"\n6. Evaluating {model_type} model...")
        eval_results = classifier.evaluate(X_test, y_test)
        print(f"   - Test accuracy: {eval_results['accuracy']:.4f}")
        
        print("\n   Classification Report:")
        report = eval_results['classification_report']
        for label in ['negative', 'positive']:
            if label in report:
                metrics = report[label]
                print(f"      {label.capitalize()}:")
                print(f"         Precision: {metrics['precision']:.4f}")
                print(f"         Recall: {metrics['recall']:.4f}")
                print(f"         F1-score: {metrics['f1-score']:.4f}")
        
        print("\n7. Making predictions on new texts...")
        new_texts = [
            "This is absolutely wonderful!",
            "Very bad experience, would not recommend."
        ]
        preprocessed_new = preprocessor.preprocess(new_texts)
        predictions = classifier.predict(preprocessed_new)
        
        for text, pred in zip(new_texts, predictions):
            print(f"   - Text: '{text}'")
            print(f"     Prediction: {pred}")
        
        # Save model
        model_path = f'examples/{model_type}_model.pkl'
        print(f"\n8. Saving model to {model_path}...")
        classifier.save(model_path)
        print(f"   - Model saved successfully!")
        
        print("\n" + "=" * 60)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
