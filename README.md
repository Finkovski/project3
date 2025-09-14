
# NLP | Automated Customer Reviews — Project Skeleton

This repository is a ready-to-run scaffold for the **Ironhack NLP Business Case**. It includes code stubs and a Streamlit app that exposes:
1) **Sentiment classification**, 2) **Product-category clustering**, 3) **Review summarization**.

## Quickstart

```bash
# 1) Create and activate a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Download a dataset (see data/README.md)
#    Place CSV(s) into data/raw/

# 4) Run the Streamlit app
streamlit run src/app/streamlit_app.py
```

> If you prefer **Gradio**, see `src/app/gradio_app.py`.

## Repo Layout

```
nlp_reviews_project/
├─ src/
│  ├─ app/                # Streamlit/Gradio UI
│  ├─ data/               # Data loaders & preprocessing
│  ├─ models/             # ML/NLP model code
│  ├─ utils/              # Metrics & viz helpers
│  └─ config.py           # Config dataclasses & paths
├─ scripts/               # CLI entrypoints
├─ data/                  # Put raw/processed data here (gitignored)
├─ notebooks/             # EDA and experiments
├─ reports/               # PDF/MD reports, PPT outline
├─ configs/               # YAML configs
├─ requirements.txt
├─ setup.cfg              # Linting/format settings
└─ README.md
```

## What’s Implemented

- **Classification**: HF Transformers (DistilBERT by default) with training/eval stubs and a confusion matrix plot.
- **Clustering**: TF‑IDF + KMeans baseline, or Sentence-Transformers embeddings (configurable).
- **Summarization**: BART/T5 pipeline; prompts accept per‑category stats.
- **App**: Minimal Streamlit UI to demo all three components.

## Datasets

- Kaggle: *Consumer Reviews of Amazon Products*
- UCSD: *Amazon Product Reviews*
- See `data/README.md` for download & expected CSV schema.

## Deliverables Hints

- Export generated **blog posts** to `artifacts/summaries/*.md`
- Create **PDF** using your report in `reports/`.
- Use `reports/presentation_outline.md` to build your PPT.
- Deploy app on **Hugging Face Spaces** or **Streamlit Community Cloud**.

## Commands

```bash
# Train classifier
python scripts/train_classifier.py --config configs/default.yaml

# Run clustering
python scripts/run_clustering.py --config configs/default.yaml

# Generate summaries (reads clustering & metrics)
python scripts/generate_summaries.py --config configs/default.yaml
```

## Notes
- All modules are type‑hinted and documented.
- Swap models in `configs/default.yaml`.
- For faster tests, set `debug.small_sample: true` in config.
