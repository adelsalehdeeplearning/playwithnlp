# PlayWithNLP - Text Classification

A Python library for text classification tasks using Natural Language Processing (NLP) techniques.

## Features

- **Multiple Classification Models**: Support for Naive Bayes, Logistic Regression, and SVM
- **Text Preprocessing**: Built-in text cleaning and normalization utilities
- **Easy Data Loading**: Simple data loaders for various input formats
- **Model Persistence**: Save and load trained models
- **Comprehensive Evaluation**: Detailed metrics including accuracy, precision, recall, and F1-score

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adelsalehdeeplearning/playwithnlp.git
cd playwithnlp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.text_classification import TextClassifier, DataLoader, TextPreprocessor

# 1. Prepare your data
texts = [
    "This is a positive example",
    "This is a negative example",
    # ... more examples
]
labels = ["positive", "negative"]

# 2. Preprocess texts
preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=True)
clean_texts = preprocessor.preprocess(texts)

# 3. Load and split data
data_loader = DataLoader()
data_loader.load_from_lists(clean_texts, labels)
X_train, X_test, y_train, y_test = data_loader.train_test_split(test_size=0.2)

# 4. Train a classifier
classifier = TextClassifier(model_type='naive_bayes')
train_info = classifier.train(X_train, y_train)
print(f"Training accuracy: {train_info['train_accuracy']:.4f}")

# 5. Evaluate the model
results = classifier.evaluate(X_test, y_test)
print(f"Test accuracy: {results['accuracy']:.4f}")

# 6. Make predictions
new_texts = ["This is amazing!", "This is terrible"]
predictions = classifier.predict(preprocessor.preprocess(new_texts))
print(predictions)

# 7. Save the model
classifier.save('my_model.pkl')
```

## Running Examples

Run the example script to see the full workflow:

```bash
python examples/text_classification_example.py
```

## Running Tests

Run the unit tests:

```bash
python -m pytest tests/
# or
python tests/test_text_classification.py
```

## Project Structure

```
playwithnlp/
├── src/
│   └── text_classification/
│       ├── __init__.py
│       ├── classifier.py       # Main classification model
│       ├── data_loader.py      # Data loading utilities
│       └── preprocessing.py    # Text preprocessing utilities
├── examples/
│   └── text_classification_example.py
├── tests/
│   └── test_text_classification.py
├── requirements.txt
└── README.md
```

## API Reference

### TextClassifier

Main class for text classification.

**Parameters:**
- `model_type` (str): Type of model ('naive_bayes', 'logistic_regression', 'svm')
- `max_features` (int): Maximum number of TF-IDF features
- `**model_kwargs`: Additional model-specific parameters

**Methods:**
- `train(texts, labels)`: Train the classifier
- `predict(texts)`: Predict labels for new texts
- `evaluate(texts, labels)`: Evaluate model performance
- `save(filepath)`: Save the trained model
- `load(filepath)`: Load a trained model

### TextPreprocessor

Text preprocessing utilities.

**Parameters:**
- `lowercase` (bool): Convert text to lowercase
- `remove_punctuation` (bool): Remove punctuation marks
- `remove_numbers` (bool): Remove numeric characters
- `remove_extra_spaces` (bool): Remove extra whitespace

**Methods:**
- `preprocess(text)`: Preprocess text or list of texts

### DataLoader

Data loading and splitting utilities.

**Parameters:**
- `text_column` (str): Name of text column
- `label_column` (str): Name of label column

**Methods:**
- `load_from_csv(filepath)`: Load data from CSV
- `load_from_dataframe(df)`: Load data from DataFrame
- `load_from_lists(texts, labels)`: Load data from lists
- `train_test_split(test_size, random_state)`: Split data into train/test sets

## Supported Models

1. **Naive Bayes** (`naive_bayes`): Fast and efficient for text classification
2. **Logistic Regression** (`logistic_regression`): Linear model with good performance
3. **Support Vector Machine** (`svm`): Powerful for binary and multi-class classification

## Requirements

- Python >= 3.7
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- nltk >= 3.8.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.