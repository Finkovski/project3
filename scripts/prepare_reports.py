import shutil, pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
reports = root / "reports"
(reports / "images").mkdir(parents=True, exist_ok=True)
(reports / "summaries").mkdir(parents=True, exist_ok=True)

# 1) confusion matrix -> reports/images/
cm = root / "artifacts" / "confusion_matrix.png"
if cm.exists():
    shutil.copy2(cm, reports / "images" / "confusion_matrix.png")
else:
    print("NOTE: artifacts/confusion_matrix.png not found. Run train_classifier.py first.")

# 2) cluster counts (+ avg rating) -> reports/cluster_counts.txt
cl_csv = root / "data" / "processed" / "clustered.csv"
if cl_csv.exists():
    df = pd.read_csv(cl_csv)
    out = reports / "cluster_counts.txt"
    with open(out, "w") as f:
        f.write("Counts per cluster:\n")
        f.write(df["cluster"].value_counts().sort_index().to_string())
        if "rating" in df.columns:
            f.write("\n\nAvg rating per cluster:\n")
            f.write(df.groupby("cluster")["rating"].mean().round(2).to_string())
    print("Wrote:", out)
else:
    print("NOTE: data/processed/clustered.csv not found. Run run_clustering.py first.")

# 3) copy summaries -> reports/summaries/
summ_dir = root / "artifacts" / "summaries"
if summ_dir.exists():
    wrote_any = False
    for p in summ_dir.glob("*.md"):
        shutil.copy2(p, reports / "summaries" / p.name)
        wrote_any = True
    if wrote_any:
        print("Copied summaries to reports/summaries/")
    else:
        print("NOTE: no *.md files in artifacts/summaries/. Generate summaries first.")
else:
    print("NOTE: artifacts/summaries/ does not exist. Generate summaries first.")

print("Done. Open the reports/ folder and commit what you need.")
