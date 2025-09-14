
import argparse
from pathlib import Path
from src.config import load_config
from src.data.dataset import load_reviews_csv
from src.models.clustering import cluster_products

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--csv', default='data/raw/sample.csv')
args = parser.parse_args()

cfg = load_config(args.config)
df = load_reviews_csv(args.csv)
out_df = cluster_products(df, method=cfg.clustering.method, n_clusters=cfg.clustering.n_clusters, max_features=cfg.clustering.max_features)
Path(cfg.paths.processed_dir).mkdir(parents=True, exist_ok=True)
out_path = Path(cfg.paths.processed_dir) / 'clustered.csv'
out_df.to_csv(out_path, index=False)
print("Saved:", out_path)
