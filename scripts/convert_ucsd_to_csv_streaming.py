import argparse, json, gzip, sys, csv
from pathlib import Path

def open_text(path):
    if path == "-":
        return sys.stdin
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def main(inp, out, category):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["review_text","rating","product_id","product_title","category"])
        with open_text(inp) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                review = obj.get("reviewText") or obj.get("review_text") or ""
                rating = obj.get("overall") or obj.get("rating")
                asin = obj.get("asin") or ""
                title = obj.get("summary") or obj.get("title") or asin
                w.writerow([review, rating, asin, title, category])
                if i % 100000 == 0:
                    print(f"wrote {i:,} rows...", file=sys.stderr)
    print(f"Done -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="path to *_5.json(.gz) or '-' for stdin")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--category", default="Unknown")
    args = ap.parse_args()
    main(args.inp, args.out, args.category)
