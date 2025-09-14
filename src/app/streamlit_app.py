import glob
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.dataset import load_reviews_csv
from src.data.preprocess import prepare_dataframe
from src.models.clustering import cluster_products
from src.models.summarizer import summarize_category


import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="NLP Reviews Demo", layout="wide")
st.title("🛍️ Automated Customer Reviews — Demo App")
st.caption("Sentiment • Clustering • Summarization")

# --------------------------- Constants ---------------------------
SENTIMENT_MODELS = {
    "Twitter RoBERTa (3-class)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "NLPTown BERT (1–5 stars)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "DistilBERT SST-2 (2-class)": "distilbert-base-uncased-finetuned-sst-2-english",
}

SUMMARIZERS = [
    "sshleifer/distilbart-cnn-12-6",  # fast & decent
    "facebook/bart-large-cnn",        # higher quality, slower
    "t5-small",                       # tiny; we add 'summarize:' prefix in summarize_category
]

# --------------------------- Data loader ---------------------------
st.sidebar.header("Data")
source = st.sidebar.radio("Source", ["Upload", "From data/raw"], index=1)

@st.cache_data(show_spinner=False)
def _df_from_path(path: str):
    return prepare_dataframe(load_reviews_csv(path))

@st.cache_data(show_spinner=False)
def _df_from_buffer(buf):
    return prepare_dataframe(load_reviews_csv(buf))

df = None
if source == "Upload":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv", "csv.gz"])
    if uploaded is not None:
        df = _df_from_buffer(uploaded)
else:
    files = sorted(glob.glob("data/raw/*.csv")) + sorted(glob.glob("data/raw/*.csv.gz"))
    if not files:
        st.warning("No local CSVs found in data/raw/. Drop a file there or switch to Upload.")
        st.stop()
    sel = st.sidebar.selectbox("Pick a local file", files)
    if sel:
        df = _df_from_path(sel)

if df is None:
    st.info("Upload or select a CSV with columns like: review_text, rating, product_title, category.")
    st.stop()

# Optional sampling for responsiveness on huge files
max_rows = st.sidebar.number_input("Use at most N rows (0 = all)", min_value=0, value=100_000, step=10_000)
if max_rows and max_rows > 0 and len(df) > max_rows:
    df = df.sample(n=max_rows, random_state=42)
    st.caption(f"Using a sample of {len(df):,} rows for speed.")

st.success(f"Loaded {len(df):,} rows.")

# --------------------------- Helpers ---------------------------
@st.cache_resource(show_spinner=True)
def load_clf_safely(model_id: str):
    from transformers import pipeline
    try:
        return pipeline("sentiment-analysis", model=model_id)
    except Exception as e:
        st.warning(
            f"Couldn't load `{model_id}`. Falling back to DistilBERT SST-2.\n\nReason: {e}"
        )
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def normalize_label(pred):
    lbl = str(pred.get("label", "")).lower()
    # NLPTown "X star(s)" → neg/neu/pos
    if "star" in lbl:
        try:
            n = int(lbl.split()[0])
            return "negative" if n <= 2 else ("neutral" if n == 3 else "positive")
        except Exception:
            pass
    if lbl in {"positive", "negative", "neutral"}:
        return lbl
    if "pos" in lbl: return "positive"
    if "neg" in lbl: return "negative"
    return lbl or "unknown"

# --------------------------- Tabs ---------------------------
tab1, tab2, tab3 = st.tabs(["Classification", "Clustering", "Summaries"])

# ===== Classification =====
with tab1:
    st.subheader("Sentiment Classification")
    choice = st.selectbox("Model", list(SENTIMENT_MODELS.keys()), index=2)  # default to small model
    clf = load_clf_safely(SENTIMENT_MODELS[choice])
    txt = st.text_area("Try a review:", "This product works great and the battery lasts forever!")
    if st.button("Classify"):
        pred = clf(txt)[0]
        st.json({"raw": pred, "normalized": normalize_label(pred)})

# ===== Clustering =====
with tab2:
    st.subheader("Product Category Clustering (Meta-categories)")
    n_clusters = st.slider("Clusters", 4, 12, 6)
    method = st.selectbox("Embedding", ["tfidf", "sbert"], index=0)

    if st.button("Run Clustering"):
        clustered = cluster_products(df, method=method, n_clusters=n_clusters)
        cols = [c for c in ["product_title", "category", "cluster"] if c in clustered.columns]
        st.dataframe(clustered[cols].head(100), use_container_width=True)

        # Download clustered CSV
        csv_bytes = clustered.to_csv(index=False).encode("utf-8")
        st.download_button("Download clustered CSV", csv_bytes, file_name="clustered.csv", mime="text/csv")

        # Label suggestions (TF-IDF only)
        st.markdown("#### Cluster label suggestions")
        if method == "tfidf":
            # Build TF-IDF over titles (fallback to review_text)
            if "product_title" in clustered.columns:
                texts = clustered["product_title"].fillna("").astype(str).tolist()
            else:
                texts = clustered["review_text"].fillna("").astype(str).tolist()

            vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
            X = vec.fit_transform(texts)
            terms = np.array(vec.get_feature_names_out())

            for c in sorted(clustered["cluster"].unique()):
                idx = (clustered["cluster"] == c).to_numpy().nonzero()[0]
                if len(idx) == 0:
                    continue
                centroid = X[idx].mean(axis=0).A1
                top = terms[centroid.argsort()[::-1][:8]]
                st.write(f"**Cluster {c}:** " + ", ".join(top))
        else:
            st.info("Label suggestions are shown for TF-IDF mode. Switch embedding to TF-IDF.")

# ===== Summaries =====
with tab3:
    st.subheader("Review Summaries → Recommendation Article")
    cat_col_candidates = [c for c in ["category", "product_title"] if c in df.columns]
    if not cat_col_candidates:
        st.warning("No category-like columns found. Add a 'category' or 'product_title' column.")
    else:
        cat_col = st.selectbox("Choose category column", cat_col_candidates)
        cat_values = sorted(df[cat_col].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Category value", cat_values)

        sum_model = st.selectbox("Summarizer model", SUMMARIZERS, index=0)

        if st.button("Generate Summary"):
            try:
                with st.status("Generating summary…", expanded=True) as status:
                    status.write("Loading model (first load may take a bit)…")
                    cat_df = df[df[cat_col].astype(str) == str(selected)]

                    status.write(f"Preparing stats for '{selected}'…")
                    article = summarize_category(cat_df, str(selected), model_name=sum_model)

                    status.update(label="Done", state="complete")
                st.markdown("### ✍️ Article")
                st.write(article)

                out_dir = Path("artifacts/summaries")
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{str(selected).replace('/', '_')}.md"
                out_file.write_text(article, encoding="utf-8")
                st.success(f"Saved to {out_file}")

                st.download_button(
                    "Download article (.md)",
                    article.encode("utf-8"),
                    file_name=f"{str(selected).replace('/', '_')}.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error("Failed to generate summary.")
                st.exception(e)
