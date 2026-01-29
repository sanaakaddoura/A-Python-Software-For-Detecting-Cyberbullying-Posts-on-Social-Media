import pandas as pd
import numpy as np
import string
import re
import emoji
import pyarabic.araby as araby
from typing import Optional

class ArabicTextPreprocessor:
    def __init__(self, data_column: str):
        self.data_column = data_column

    @staticmethod
    def normalize_arabic(text: str) -> str:
        text = re.sub("[إأآا]", "ا", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("گ", "ك", text)
        return text

    @staticmethod
    def remove_repeating_char(text: str) -> str:
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    @staticmethod
    def data_cleaning(series: pd.Series) -> pd.Series:
        clean = series.copy()
        clean = clean.apply(lambda r: re.sub(r'\.+?(?=\B|$)', '', r))
        clean = clean.apply(lambda r: re.sub(r'\n', ' ', r))
        clean = clean.apply(lambda r: r.replace('#', ' ').replace('_', ' '))
        clean = clean.apply(lambda r: re.sub(r'@[a-zA-Z0-9_]+', ' ', r))
        clean = clean.apply(lambda r: re.sub(r'https?\S+(?=\s|$)', 'www', r))
        clean = clean.apply(lambda r: emoji.replace_emoji(r, replace=""))
        clean = clean.apply(lambda r: araby.strip_diacritics(r))
        clean = clean.apply(lambda r: ' '.join(r.split()))
        return clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove special characters except spaces and words
        df[self.data_column] = df[self.data_column].str.replace(r'[^\w\s]+', '', regex=True)
        df[self.data_column] = df[self.data_column].str.replace(r'\s+', ' ', regex=True)

        # Normalize Arabic characters
        df[self.data_column] = df[self.data_column].apply(self.normalize_arabic)

        # Remove repeated characters
        df[self.data_column] = df[self.data_column].apply(self.remove_repeating_char)

        # General cleaning
        df[self.data_column] = self.data_cleaning(df[self.data_column])

        # Drop duplicates and NaNs
        df.drop_duplicates(subset=self.data_column, inplace=True)
        df.dropna(subset=[self.data_column], inplace=True)

        return df