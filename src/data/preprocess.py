
import re
import pandas as pd
from typing import Union

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ,.?!'-]", " ", text)
    return text.strip()

def map_rating_to_sentiment(rating: Union[int, float]) -> int:
    # 0=negative, 1=neutral, 2=positive
    try:
        r = float(rating)
    except Exception:
        return 1
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "review_text" in df.columns:
        df["review_text"] = df["review_text"].map(clean_text)
    if "rating" in df.columns:
        df["sentiment"] = df["rating"].map(map_rating_to_sentiment)
    return df.dropna(subset=["review_text"])
