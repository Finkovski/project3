
import argparse
from pathlib import Path
from src.config import load_config
from src.data.dataset import load_reviews_csv
from src.models.summarizer import summarize_category

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--csv', default='data/raw/sample.csv')
parser.add_argument('--category_col', default='category')
args = parser.parse_args()

cfg = load_config(args.config)
df = load_reviews_csv(args.csv)

art_dir = Path(cfg.paths.artifacts_dir) / 'summaries'
art_dir.mkdir(parents=True, exist_ok=True)

for cat, sub in df.groupby(args.category_col):
    text = summarize_category(sub, str(cat), model_name=cfg.summarization.model_name, max_input_tokens=cfg.summarization.max_input_tokens, max_output_tokens=cfg.summarization.max_output_tokens)
    out = art_dir / f"{str(cat).replace('/', '_')}.md"
    out.write_text(text, encoding='utf-8')
    print("Wrote:", out)
