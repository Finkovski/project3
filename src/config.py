
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    raw_data_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    models_dir: Path

@dataclass
class DebugCfg:
    small_sample: bool = False
    sample_size: int = 5000

@dataclass
class ClassifierCfg:
    model_name: str
    max_length: int = 256
    train_size: float = 0.8
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 5e-5

@dataclass
class ClusteringCfg:
    n_clusters: int = 5
    method: str = "tfidf"  # or "sbert"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_features: int = 20000

@dataclass
class SummarizationCfg:
    model_name: str = "facebook/bart-large-cnn"
    max_input_tokens: int = 1024
    max_output_tokens: int = 256

@dataclass
class Config:
    paths: Paths
    debug: DebugCfg
    classification: ClassifierCfg
    clustering: ClusteringCfg
    summarization: SummarizationCfg

def load_config(path: str) -> "Config":
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(
        paths=Paths(**cfg["paths"]),
        debug=DebugCfg(**cfg["debug"]),
        classification=ClassifierCfg(**cfg["classification"]),
        clustering=ClusteringCfg(**cfg["clustering"]),
        summarization=SummarizationCfg(**cfg["summarization"]),
    )
