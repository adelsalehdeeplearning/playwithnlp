"""Data loader for text classification tasks."""

import pandas as pd
from typing import Tuple, List, Optional
import numpy as np


class DataLoader:
    """Load and manage text classification datasets."""
    
    def __init__(self, text_column: str = 'text', label_column: str = 'label'):
        """
        Initialize the data loader.
        
        Args:
            text_column: Name of the column containing text data
            label_column: Name of the column containing labels
        """
        self.text_column = text_column
        self.label_column = label_column
        self.data = None
        
    def load_from_csv(self, filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        self.data = pd.read_csv(filepath)
        texts = self.data[self.text_column].tolist()
        labels = self.data[self.label_column].tolist()
        return texts, labels
    
    def load_from_dataframe(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Load data from a pandas DataFrame.
        
        Args:
            df: DataFrame containing text and label columns
            
        Returns:
            Tuple of (texts, labels)
        """
        self.data = df
        texts = df[self.text_column].tolist()
        labels = df[self.label_column].tolist()
        return texts, labels
    
    def load_from_lists(self, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Load data from lists.
        
        Args:
            texts: List of text strings
            labels: List of label strings
            
        Returns:
            Tuple of (texts, labels)
        """
        self.data = pd.DataFrame({
            self.text_column: texts,
            self.label_column: labels
        })
        return texts, labels
    
    def train_test_split(self, test_size: float = 0.2, 
                        random_state: Optional[int] = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_from_* methods first.")
        
        from sklearn.model_selection import train_test_split
        
        texts = self.data[self.text_column].tolist()
        labels = self.data[self.label_column].tolist()
        
        return train_test_split(texts, labels, test_size=test_size, 
                              random_state=random_state)
