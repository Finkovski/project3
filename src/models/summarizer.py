from __future__ import annotations
from functools import lru_cache
import pandas as pd
from transformers import pipeline

DEFAULT_PROMPT = (
    "You are a product reviewer. Given product-level stats and representative snippets, write a short buyer's guide. "
    "Include: (1) top 3 products and key differences, (2) top complaints for each, (3) the worst product and why to avoid it. "
    "Keep it under 180 words, objective but concise."
)

@lru_cache(maxsize=4)
def get_summarizer(model_name: str):
    # caches the HF pipeline across reruns
    return pipeline("summarization", model=model_name)

def summarize_category(
    df: pd.DataFrame,
    category_name: str,
    model_name: str = "sshleifer/distilbart-cnn-12-6",
    max_input_tokens: int = 1024,
    max_output_tokens: int = 256,
    max_rows_for_stats: int = 20000,  # guard for huge categories
) -> str:
    # limit rows used for stats so we don't spend time grouping millions
    if len(df) > max_rows_for_stats:
        df = df.sample(n=max_rows_for_stats, random_state=42)

    by_product = (
        df.groupby("product_title", dropna=False)
          .agg(avg_rating=("rating", "mean"), reviews=("review_text", "count"))
          .sort_values(["avg_rating", "reviews"], ascending=[False, False])
          .head(10)
          .reset_index()
    )
    top = by_product.head(3).to_dict(orient="records")
    worst = by_product.tail(1).to_dict(orient="records")
    text_snips = (
        df["review_text"].dropna().astype(str).head(50).str.slice(0, 200).tolist()
    )

    payload = {"category": category_name, "top": top, "worst": worst, "snippets": text_snips[:10]}
    src = DEFAULT_PROMPT + "\n\nDATA:\n" + str(payload)

    # T5 models need a prefix
    if model_name.lower().startswith("t5"):
        src = "summarize: " + src

    gen = get_summarizer(model_name)
    out = gen(src, max_length=max_output_tokens, min_length=80, do_sample=False)[0]["summary_text"]
    return out
