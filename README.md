# Automated Customer Reviews — NLP Demo

Sentiment analysis, product clustering, and auto-generated buyer-guide summaries for Amazon-style reviews.
Built with Streamlit, Hugging Face Transformers, scikit-learn, and pandas.

> Data: UCSD Amazon Reviews dataset (category splits).
> Models: `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment; BART/T5 for summarization.

---

## What it does

- Sentiment classification (neg/neu/pos) with a fine-tuned transformer
- Clustering of product titles into meta-categories (TF-IDF / SBERT)
- Summaries → one-click buyer-guide article per category
- Streamlit app with upload or local file selection
- Artifacts: confusion matrix image, clustered CSV, Markdown summaries, optional saved model

---

## Project structure

```
nlp_reviews_project/
├─ src/
│  ├─ app/
│  │  ├─ streamlit_app.py         # main app
│  │  └─ gradio_app.py            # optional
│  ├─ data/
│  │  ├─ dataset.py               # CSV loader
│  │  └─ preprocess.py            # rating → sentiment, cleaning
│  ├─ models/
│  │  ├─ classifier.py            # training (freeze/unfreeze, class-weights)
│  │  ├─ clustering.py            # TF-IDF & SBERT + KMeans
│  │  └─ summarizer.py            # category summarization
│  └─ utils/
│     ├─ metrics.py               # accuracy/precision/recall/F1 + confusion
│     └─ viz.py                   # confusion matrix plot
├─ scripts/
│  ├─ convert_ucsd_to_csv_streaming.py
│  ├─ train_classifier.py         # CLI with --limit, --freeze-base, etc.
│  ├─ run_clustering.py
│  └─ generate_summaries.py
├─ configs/default.yaml
├─ requirements.txt
└─ .gitignore
```

---

## Quickstart

```bash
# 1) create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) env for imports & tokenizers
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
```

Start the app:

```bash
streamlit run src/app/streamlit_app.py
```

Tip (big files): app sidebar → From data/raw lets you select local CSVs without the 200 MB upload cap.

---

## Get data (UCSD Amazon Reviews)

Example categories (smaller splits):

```bash
# Video Games
curl -L https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz | gunzip -c | python scripts/convert_ucsd_to_csv_streaming.py --in -       --out data/raw/amazon_video_games.csv       --category "Video Games"

# Musical Instruments
curl -L https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz | gunzip -c | python scripts/convert_ucsd_to_csv_streaming.py --in -       --out data/raw/amazon_musical_instruments.csv       --category "Musical Instruments"

# Office Products
curl -L https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz | gunzip -c | python scripts/convert_ucsd_to_csv_streaming.py --in -       --out data/raw/amazon_office_products.csv       --category "Office Products"
```

Each command streams and converts JSON-lines to CSV with columns:
`review_text, rating, product_id, product_title, category`.

---

## Streamlit usage

1. Sidebar → Source = From data/raw → pick a CSV.
2. Optionally set “Use at most N rows” (100k–200k is snappy on CPU).
3. Tabs:
   - Classification: try a review, pick a model
   - Clustering: choose K (about 6), embedding (TF-IDF for speed), Run
   - Summaries: choose `category` → Generate Summary (saves to `artifacts/summaries/*.md`)

---

## Train the classifier (fast presets)

Head-only, balanced subset, 2 epochs, and save the model:

```bash
python scripts/train_classifier.py   --csv data/raw/amazon_video_games.csv   --limit-per-class 6000   --epochs 2   --freeze-base   --save-model-dir artifacts/model
```

Other useful flags:

- `--unfreeze-last-n 2`  (train last 2 encoder layers for a small accuracy bump)
- `--class-weights`      (use when training on imbalanced data; not needed for balanced `--limit-per-class`)
- `--label-smoothing 0.05` (light regularization)

Outputs:
- `artifacts/confusion_matrix.png`
- saved model under `artifacts/model/` (if `--save-model-dir` used)

Sanity-check the saved model:

```bash
python - <<'PY'
from transformers import pipeline
clf = pipeline("text-classification", model="artifacts/model", tokenizer="artifacts/model")
print(clf("This game is fantastic, but battery life is short."))
print(clf("Average at best—works, but lots of bugs."))
print(clf("Terrible build quality, stopped working in a week."))
PY
```

---

## Clustering (meta-categories)

From the app (recommended): TF-IDF, K ≈ 6 → Run → Download clustered CSV

Or CLI:

```bash
python scripts/run_clustering.py --csv data/raw/amazon_video_games.csv
```

You will get `data/processed/clustered.csv` and label suggestions in-app.

---

## Summaries (buyer-guide articles)

From the app → Summaries:
- `category` column → choose a value (for example, Video Games)
- summarizer `sshleifer/distilbart-cnn-12-6` for speed
- Generate Summary → saved to `artifacts/summaries/<category>.md`

CLI batch (one per category value):

```bash
python scripts/generate_summaries.py   --csv data/raw/amazon_video_games.csv   --category_col category
```

---

## Deliverables

Typical artifacts to share:

```
artifacts/
├─ confusion_matrix.png
├─ model/                      # optional if you used --save-model-dir
└─ summaries/
   ├─ Video Games.md
   ├─ Musical Instruments.md
   └─ Office Products.md
data/processed/
└─ clustered.csv
```

Keep the repo slim: these paths are ignored by `.gitignore`. Zip them for hand-in:

```bash
mkdir -p deliverables
cp artifacts/confusion_matrix.png deliverables/
cp artifacts/summaries/*.md deliverables/ 2>/dev/null || true
cp data/processed/clustered.csv deliverables/ 2>/dev/null || true
zip -r nlp_reviews_deliverable.zip deliverables
```

---

## Troubleshooting

- `ModuleNotFoundError: src` → `export PYTHONPATH="$PWD"` before running.
- Streamlit 200 MB upload limit → use From data/raw in the sidebar.
- macOS MPS `pin_memory` warning → harmless.
- LibreSSL warning from urllib3 → harmless for local runs.

---

## .gitignore policy

The repo intentionally ignores:

```
.venv/ __pycache__/ data/raw/ data/processed/ artifacts/ deliverables/
*.csv *.csv.gz *.json.gz *.parquet *.zip
```

If you want to include a tiny sample CSV for demos, put it under `data/samples/` and adjust `.gitignore` accordingly.

---

## Acknowledgements

- UCSD Amazon Reviews dataset — Julian McAuley et al.
- Hugging Face Transformers and Datasets
- CardiffNLP RoBERTa sentiment model
- Streamlit

---

## License

MIT. See the `LICENSE` file.

---

## Reproducibility check

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"; export TOKENIZERS_PARALLELISM=false
streamlit run src/app/streamlit_app.py
```
