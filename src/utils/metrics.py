
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
