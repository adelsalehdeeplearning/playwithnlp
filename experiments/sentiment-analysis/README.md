# Sentiment Analysis Experiment

## Overview
This experiment performs sentiment analysis on text using pre-trained transformer models from Hugging Face. It classifies text as positive, negative, or neutral sentiment.

## Approach
- **Model**: Using `distilbert-base-uncased-finetuned-sst-2-english` from Hugging Face Transformers
- **Technique**: Transfer learning with a pre-trained model fine-tuned on sentiment analysis
- **Libraries**: 
  - transformers (Hugging Face)
  - torch (PyTorch backend)

## Results
The model achieves high accuracy on sentiment classification tasks:
- **Positive sentiment** detection: Clear identification of positive language
- **Negative sentiment** detection: Accurate detection of negative emotions
- **Confidence scores**: Provides probability scores for each prediction

### Example Outputs

| Text | Predicted Sentiment | Confidence |
|------|---------------------|------------|
| "I love this product!" | POSITIVE | 0.9998 |
| "This is terrible and disappointing" | NEGATIVE | 0.9995 |
| "The weather is okay today" | NEUTRAL | 0.7234 |

## Usage

### Running the Code
```bash
python experiment.py
```

This will analyze several example texts and display their sentiment predictions.

### Running the Notebook
```bash
jupyter notebook notebook.ipynb
```

The notebook provides:
- Step-by-step walkthrough of the sentiment analysis process
- Interactive examples where you can input your own text
- Visualizations of confidence scores
- Comparison of different texts

## Key Learnings
- Pre-trained transformers are highly effective for sentiment analysis
- Transfer learning significantly reduces training time and data requirements
- Fine-tuned models capture nuanced sentiment expressions
- Confidence scores help assess prediction reliability

## Future Improvements
- Fine-tune on domain-specific datasets (e.g., product reviews, tweets)
- Add support for multi-label sentiment (e.g., joy, anger, sadness)
- Implement aspect-based sentiment analysis
- Compare with traditional ML approaches (Naive Bayes, SVM)

## References
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [SST-2 Dataset](https://nlp.stanford.edu/sentiment/)
