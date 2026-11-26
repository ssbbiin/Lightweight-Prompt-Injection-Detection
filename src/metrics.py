import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


def calculate_metrics(labels, probabilities, threshold=0.5):

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels, predictions, zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(labels, probabilities)
        ),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
