"""
Named Entity Recognition (NER) Experiment

This script demonstrates NER using pre-trained transformer models
to identify and classify named entities in text.
"""

from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


class NamedEntityRecognizer:
    """
    A named entity recognizer using pre-trained transformers.
    """
    
    def __init__(self, model_name="dslim/bert-base-NER"):
        """
        Initialize the NER model.
        
        Args:
            model_name (str): Name of the pre-trained NER model
        """
        print(f"Loading NER model: {model_name}...")
        self.ner = pipeline("ner", model=model_name, aggregation_strategy="simple")
        print("Model loaded successfully!")
    
    def extract_entities(self, text):
        """
        Extract named entities from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            list: List of entity dictionaries
        """
        entities = self.ner(text)
        return entities
    
    def print_entities(self, text, entities):
        """
        Pretty print the extracted entities.
        
        Args:
            text (str): Original input text
            entities (list): List of extracted entities
        """
        print(f"\nText: '{text}'")
        print(f"\nFound {len(entities)} entities:")
        
        if not entities:
            print("  No entities found.")
            return
        
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['entity_group']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        # Print entities grouped by type
        for entity_type, entity_list in sorted(entities_by_type.items()):
            print(f"\n  {entity_type} ({len(entity_list)}):")
            for entity in entity_list:
                word = entity['word']
                score = entity['score']
                start = entity['start']
                end = entity['end']
                print(f"    - '{word}' (confidence: {score:.4f}, position: {start}-{end})")
        
        print("-" * 70)
    
    def highlight_entities(self, text, entities):
        """
        Create a text visualization with highlighted entities.
        
        Args:
            text (str): Original input text
            entities (list): List of extracted entities
            
        Returns:
            str: Text with entity markers
        """
        if not entities:
            return text
        
        # Sort entities by start position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        highlighted = text
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            word = entity['word']
            entity_type = entity['entity_group']
            
            # Replace entity with marked version
            marked = f"[{word}]({entity_type})"
            highlighted = highlighted[:start] + marked + highlighted[end:]
        
        return highlighted


def main():
    """
    Main function to demonstrate NER.
    """
    # Initialize the recognizer
    recognizer = NamedEntityRecognizer()
    
    # Example texts with various entities
    example_texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Barack Obama was the 44th President of the United States and lived in Washington D.C.",
        "Google announced a new AI research lab in London, led by Demis Hassabis.",
        "The Eiffel Tower in Paris attracts millions of visitors every year.",
        "Microsoft CEO Satya Nadella spoke at the conference in Seattle.",
        "The Amazon rainforest spans across Brazil, Peru, and Colombia.",
        "Einstein developed the theory of relativity while working in Switzerland.",
    ]
    
    print("\n" + "=" * 70)
    print("NAMED ENTITY RECOGNITION DEMONSTRATION")
    print("=" * 70)
    
    # Process each text
    for text in example_texts:
        entities = recognizer.extract_entities(text)
        recognizer.print_entities(text, entities)
        
        # Show highlighted version
        highlighted = recognizer.highlight_entities(text, entities)
        print(f"\nHighlighted: {highlighted}\n")
    
    # Statistics
    print("\n" + "=" * 70)
    print("ENTITY STATISTICS")
    print("=" * 70)
    
    all_entities = []
    entity_type_counts = {}
    
    for text in example_texts:
        entities = recognizer.extract_entities(text)
        all_entities.extend(entities)
        
        for entity in entities:
            entity_type = entity['entity_group']
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
    
    print(f"\nTotal entities found: {len(all_entities)}")
    print(f"\nEntity type distribution:")
    for entity_type, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")
    
    avg_confidence = sum(e['score'] for e in all_entities) / len(all_entities)
    print(f"\nAverage confidence: {avg_confidence:.4f}")


if __name__ == "__main__":
    main()
