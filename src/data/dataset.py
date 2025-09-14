
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Union

@dataclass
class ReviewColumns:
    text: str = "review_text"
    rating: str = "rating"
    product_id: str = "product_id"
    product_title: str = "product_title"
    category: str = "category"

def load_reviews_csv(path: Union[str, Path], cols: ReviewColumns = ReviewColumns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Auto-map common column names
    rename_map = {}
    for guess, target in [
        ("reviews.text", cols.text),
        ("reviewText", cols.text),
        ("text", cols.text),
        ("stars", cols.rating),
        ("reviews.rating", cols.rating),
        ("rating", cols.rating),
        ("asin", cols.product_id),
        ("product_id", cols.product_id),
        ("name", cols.product_title),
        ("title", cols.product_title),
        ("categories", cols.category),
        ("category", cols.category),
    ]:
        if guess in df.columns and target not in df.columns:
            rename_map[guess] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
