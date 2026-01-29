import torch
import pandas as pd
from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

class ArabicTextTokenizer:
    def __init__(self, model_name: str, data_column: str, max_len: int = 256):
        self.model_name = model_name
        self.data_column = data_column
        self.max_len = max_len

        # Initialize preprocessing
        self.preprocessor = ArabertPreprocessor(model_name)
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=None)

    def preprocess_text(self, text: str) -> str:
        return self.preprocessor.preprocess(text)

    def tokenize_text(self, text: str) -> dict:
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply preprocessing
        df[self.data_column] = df[self.data_column].apply(self.preprocess_text)
        # Apply tokenization
        df['tokenized_column'] = df[self.data_column].apply(self.tokenize_text)
        return df

    def transform_texts(self, texts: pd.Series) -> pd.Series:
        return texts.apply(lambda x: self.tokenize_text(self.preprocess_text(x)))
