import argparse
from pathlib import Path
import pandas as pd

from src.config import load_config
from src.data.dataset import load_reviews_csv
from src.data.preprocess import prepare_dataframe
from src.models.classifier import train_and_eval
from src.utils.viz import plot_confusion_matrix


def stratified_sample(df: pd.DataFrame, n_per_class: int = None, total_limit: int = None, seed: int = 42):
    out = df
    if n_per_class is not None and "sentiment" in out.columns:
        parts = []
        for s, sub in out.groupby("sentiment"):
            parts.append(sub.sample(min(len(sub), n_per_class), random_state=seed))
        out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if total_limit is not None and len(out) > total_limit:
        out = out.sample(total_limit, random_state=seed).reset_index(drop=True)
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--csv", default="data/raw/sample.csv")
parser.add_argument("--limit", type=int, default=None, help="Total max rows to train on (after stratify).")
parser.add_argument("--limit-per-class", type=int, default=None, help="Max rows per class (stratified).")
parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
parser.add_argument("--freeze-base", action="store_true", help="Freeze transformer base; train head only.")
parser.add_argument("--unfreeze-last-n", type=int, default=0, help="Unfreeze last N encoder layers (e.g., 2).")
parser.add_argument("--class-weights", action="store_true", help="Use class-weighted loss.")
parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor (0–0.2).")
parser.add_argument("--save-model-dir", default=None, help="Directory to save fine-tuned model/tokenizer.")
parser.add_argument("--save-cm-path", default=None, help="Optional path to save confusion matrix PNG.")
args = parser.parse_args()

cfg = load_config(args.config)
df = load_reviews_csv(args.csv)
df = prepare_dataframe(df)

# stratified/limited sampling for speed
df_train = stratified_sample(df, n_per_class=args.limit_per_class, total_limit=args.limit)

# override epochs if provided
epochs = args.epochs if args.epochs is not None else cfg.classification.epochs

res = train_and_eval(
    df_train,
    model_name=cfg.classification.model_name,
    max_length=cfg.classification.max_length,
    train_size=cfg.classification.train_size,
    epochs=epochs,
    batch_size=cfg.classification.batch_size,
    lr=cfg.classification.learning_rate,
    freeze_base=args.freeze_base,
    unfreeze_last_n=args.unfreeze_last_n,
    use_class_weights=args.class_weights,
    label_smoothing=args.label_smoothing,
    save_dir=args.save_model_dir,
)

print("Metrics:", res.metrics)

# Save confusion matrix
fig = plot_confusion_matrix(res.cm)
art_dir = Path(cfg.paths.artifacts_dir); art_dir.mkdir(parents=True, exist_ok=True)
out_path = Path(args.save_cm_path) if args.save_cm_path else art_dir / "confusion_matrix.png"
fig.savefig(out_path)
print("Saved:", out_path)
