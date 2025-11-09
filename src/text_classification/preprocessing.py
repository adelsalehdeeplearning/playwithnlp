"""Text preprocessing utilities for text classification."""

import re
import string
from typing import List, Union


class TextPreprocessor:
    """Preprocessor for cleaning and normalizing text data."""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True,
                 remove_numbers: bool = False, remove_extra_spaces: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_extra_spaces: Remove extra whitespace
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces
    
    def preprocess(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Preprocess text or list of texts.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Preprocessed text or list of texts
        """
        if isinstance(text, list):
            return [self._preprocess_single(t) for t in text]
        return self._preprocess_single(text)
    
    def _preprocess_single(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces
        if self.remove_extra_spaces:
            text = ' '.join(text.split())
        
        return text.strip()
