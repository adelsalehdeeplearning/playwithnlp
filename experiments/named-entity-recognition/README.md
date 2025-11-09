# Named Entity Recognition (NER) Experiment

## Overview
This experiment demonstrates Named Entity Recognition (NER) using pre-trained transformer models. NER identifies and classifies named entities in text such as persons, organizations, locations, dates, and more.

## Approach
- **Model**: Using `dslim/bert-base-NER` from Hugging Face
- **Technique**: Token classification with pre-trained BERT
- **Entity Types**: 
  - PER (Person)
  - ORG (Organization)
  - LOC (Location)
  - MISC (Miscellaneous)

### Libraries Used
- transformers (Hugging Face)
- spacy (alternative NER library for comparison)

## Results

The model successfully identifies and classifies various entity types with high accuracy:

### Example Extractions

**Text**: "Apple Inc. was founded by Steve Jobs in Cupertino, California."

| Entity | Type | Confidence |
|--------|------|------------|
| Apple Inc. | ORG | 0.9998 |
| Steve Jobs | PER | 0.9997 |
| Cupertino | LOC | 0.9995 |
| California | LOC | 0.9993 |

### Performance Metrics
- **Precision**: High accuracy in identifying entities
- **Entity Grouping**: Properly combines multi-token entities (e.g., "Steve Jobs")
- **Context Awareness**: Understands "Apple" as organization, not fruit
- **Multi-language Support**: Works with various languages (English-optimized)

## Usage

### Running the Code
```bash
python experiment.py
```

This will:
1. Load the pre-trained NER model
2. Process example texts
3. Extract and display named entities
4. Show entity visualization

### Running the Notebook
```bash
jupyter notebook notebook.ipynb
```

The notebook provides:
- Interactive NER examples
- Entity type filtering
- Visualization of entity positions
- Comparison with spaCy NER

## Key Learnings
- **Transformers excel at NER**: Pre-trained models capture entity patterns effectively
- **Context matters**: Same word can be different entity types based on context
- **Entity aggregation**: Important to combine sub-word tokens into complete entities
- **Confidence scores**: Help identify uncertain predictions

## Use Cases
1. **Information Extraction**: Extract structured data from unstructured text
2. **Document Analysis**: Identify key entities in large document collections
3. **Knowledge Graph Construction**: Build entity relationships
4. **Privacy Protection**: Detect and redact sensitive personal information
5. **Content Enrichment**: Tag content with relevant entities for search and discovery

## Future Improvements
- Fine-tune on domain-specific entities (medical, legal, financial)
- Add custom entity types (product names, disease names, etc.)
- Implement entity linking to knowledge bases (e.g., Wikipedia)
- Build entity relationship extraction
- Create entity-based search and filtering systems

## References
- [BERT for NER](https://arxiv.org/abs/1810.04805)
- [Hugging Face NER](https://huggingface.co/tasks/token-classification)
- [CoNLL-2003 NER Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
