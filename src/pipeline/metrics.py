import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Handle T5 model output which can be a tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    prediction_scores = sigmoid(predictions)
    predictions = (prediction_scores > 0.5).astype(int)

    # Multi-label metrics
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, prediction_scores)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }
