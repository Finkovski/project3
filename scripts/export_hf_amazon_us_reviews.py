import argparse
from typing import Optional
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def main(subset: str, out_path: str, limit: Optional[int]) -> None:
    ds = load_dataset("amazon_us_reviews", subset, split="train")  # datasets 2.x

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    df = ds.to_pandas()
    out = pd.DataFrame({
        "review_text": df["review_body"],
        "rating": df["star_rating"],
        "product_title": df.get("product_title", pd.Series([""] * len(df))),
        "category": subset.replace("_v1_00", "").replace("_", " "),
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="Wireless_v1_00")  # pick one from the list below
    ap.add_argument("--out", default="data/raw/amazon_wireless.csv")
    ap.add_argument("--limit", type=int, default=20000)    # cap rows to keep it light
    args = ap.parse_args()
    main(args.subset, args.out, args.limit)
