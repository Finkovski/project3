# Slide 1 — Title
**Automated Customer Reviews: Sentiment + Clustering + Summaries**  
Jan Hemayatkar-Fink

**What’s inside**
- Classify reviews (neg/neu/pos)
- Cluster products into meta-categories
- Generate buyer-guide summaries
- Live demo app (Streamlit)

---

# Slide 2 — Problem & Goal
**Problem**
- Thousands of reviews; manual analysis is slow and noisy.

**Goal**
- Automate insights: sentiment trends, category structure, and concise recommendation articles.

**Deliverables**
- Classifier, clustering, summarizer + app + evaluation.

---

# Slide 3 — Dataset
**Source:** UCSD Amazon Reviews “categoryFilesSmall” (Video Games, Musical Instruments, Office Products).  
**Why this one?** Official assignment allows Kaggle Amazon Product Reviews and UCSD dataset (larger) — both acceptable.  
**Fields used:** review_text, rating, product_title, category, product_id  
**Scale (per category):** hundreds of thousands of reviews (sampled for speed)

*Speaker note:* The brief lists Kaggle as primary and UCSD as larger; additional datasets are allowed. We used UCSD category splits.  

---

# Slide 4 — Preprocessing
- Map stars → sentiment: 1–2 = Neg, 3 = Neu, 4–5 = Pos
- Clean text lightly; keep emojis/punctuation that help sentiment
- Train/test split = 80/20 (stratified)
- Clustering text: product titles → TF-IDF
- Summarization corpus grouped by category

---

# Slide 5 — Sentiment Model & Training
**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`  
**Setup:**
- Max length 256, batch 16, epochs 2
- Encoder frozen (fast & stable)
- Balanced sample: `--limit-per-class 6000` (≈18k total)

*Speaker note:* We start from a strong base… minimal fine-tuning to reach good performance on star-derived labels.

---

# Slide 6 — Sentiment Results
**Test (~3.6k examples):**
- Accuracy **0.691**
- Precision (weighted) **0.685**
- Recall (weighted) **0.691**
- F1 (weighted) **0.686**

**Per-class**
- Neg: P 0.696, R 0.756, F1 0.725
- Neu: P 0.602, R 0.513, F1 0.554
- Pos: P 0.758, R 0.805, F1 0.781

*Visual:* insert image `reports/images/confusion_matrix.png`

---

# Slide 7 — Error Analysis (What’s hard?)
- Neutral is hardest (overlaps with both Neg and Pos)
- Class boundary often fuzzy with star-derived labels
- Ideas to improve:
  - Unfreeze last 1–2 transformer layers
  - Class weights / focal loss on imbalance
  - Add domain-specific data augmentation

---

# Slide 8 — Clustering Approach
**Goal:** 4–6 meta-categories  
**Method:** TF-IDF on product titles + KMeans (K=6)  
**Output:** `data/processed/clustered.csv`  
**Snapshot:** `reports/cluster_counts.txt` (counts + avg rating)

*Speaker note:* Clusters reflect themes like “E-book readers”, “Accessories”, “Batteries”, etc.

---

# Slide 9 — Summarization
**Model:** `sshleifer/distilbart-cnn-12-6`  
**Per category article includes:**
- Top 3 products + key differences
- Top complaints per product
- Worst product & why to avoid
**Artifacts:** `reports/summaries/*.md` (e.g., Video Games.md)

*Speaker note:* These are compact buyer-guide style pieces you can publish.

---

# Slide 10 — Live Demo (Streamlit)
**App tabs**
- Classification: pick model, test a review
- Clustering: select K & embedding; view cluster labels
- Summaries: choose category → generate article

**Data sources**
- Upload CSV or select from local `data/raw` (faster, avoids 200MB cap)

*Visuals:* 2–3 screenshots of the app.

---

# Slide 11 — Limitations & Risks
- Neutral class noisy with star mapping
- Long reviews truncated to 256 tokens (trade-off)
- Model bias from training data
- Summaries are extractive/abstractive—fact-check for critical use

---

# Slide 12 — Impact & Next Steps
**Impact**
- Fast sentiment signals
- Category-level structure
- “Ready to publish” summaries

**Next**
- Slightly deeper fine-tuning; add class weights
- Try SBERT embeddings for clustering
- Deploy on Hugging Face Spaces
- Add guardrails & fact-checking to summaries
