
from __future__ import annotations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import numpy as np

def cluster_products(df: pd.DataFrame, method: str = "tfidf", n_clusters: int = 5, max_features: int = 20000) -> pd.DataFrame:
    texts_series = None
    if 'product_title' in df.columns:
        texts_series = df['product_title']
    elif 'review_text' in df.columns:
        texts_series = df['review_text']
    else:
        texts_series = pd.Series([""] * len(df))
    texts = texts_series.fillna('').astype(str).tolist()

    if method == 'sbert':
        emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').encode(texts, show_progress_bar=False)
        X = np.array(emb)
    else:
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
        X = vec.fit_transform(texts)

    km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    labels = km.fit_predict(X)
    out = df.copy()
    out['cluster'] = labels
    return out
