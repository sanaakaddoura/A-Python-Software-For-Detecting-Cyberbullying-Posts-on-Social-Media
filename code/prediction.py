# dataframe_tokenized_predictor.py
import torch
import pandas as pd
from typing import List
from transformers import AutoModelForSequenceClassification

class DataFramePredictor:
    """
    Run predictions on a DataFrame containing pre-tokenized transformer inputs.
    """

    def __init__(
        self,
        model_name: str,
        tokenized_column: str,
        labels: List[str],
        device: str | None = None
    ):
        self.model_name = model_name
        self.tokenized_column = tokenized_column
        self.labels = labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            token=None
        ).to(self.device)

        self.model.eval()

    def _stack_batch(self, tokenized_rows: List[dict]) -> dict:
        """
        Stack tokenized row dictionaries into a batch.
        """
        batch = {}
        for key in tokenized_rows[0].keys():
            batch[key] = torch.cat(
                [row[key] for row in tokenized_rows], dim=0
            ).to(self.device)
        return batch

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add prediction outputs to the DataFrame.

        Adds:
        - predicted_class (int)
        - predicted_label (str)
        - confidence (float)
        """
        tokenized_rows = df[self.tokenized_column].tolist()
        inputs = self._stack_batch(tokenized_rows)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1).values

        df = df.copy()
        df["predicted_class"] = predicted_class.cpu().numpy()
        df["predicted_label"] = [
            self.labels[i] for i in df["predicted_class"]
        ]
        df["confidence"] = confidence.cpu().numpy()

        return df
