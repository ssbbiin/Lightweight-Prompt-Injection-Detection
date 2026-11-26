import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path("outputs/evaluation")
OUT = Path("outputs/analysis")

OUT.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(ROOT / "test_results.csv")

test = pd.read_csv(
    "data/processed/test.csv",
    keep_default_na=False
)

combined = test.copy()

combined.insert(
    0, "test_index", np.arange(len(test))
)

for experiment in results["experiment"]:

    pred = pd.read_csv(
        ROOT / experiment / "test_predictions.csv",
        keep_default_na=False
    )

    assert pred["text"].equals(test["text"])
    assert pred["label"].equals(test["label"])

    short_id = experiment.split("_")[0]

    combined[f"{short_id}_probability"] = (
        pred["attack_probability"]
    )

    combined[f"{short_id}_prediction"] = (
        pred["prediction_05"]
    )

    combined[f"{short_id}_error"] = (
        pred["prediction_05"] != pred["label"]
    )

    if "hard_benign_proxy" not in combined.columns:
        combined["hard_benign_proxy"] = (
            pred["hard_benign_proxy"]
        )

error_cols = [
    c for c in combined.columns
    if c.endswith("_error")
]

combined["error_count"] = (
    combined[error_cols].sum(axis=1)
)

combined.to_csv(
    OUT / "all_model_predictions.csv",
    index=False
)

combined[
    combined["error_count"] > 0
].sort_values(
    "error_count",
    ascending=False
).to_csv(
    OUT / "misclassified_samples.csv",
    index=False
)

hard_benign = combined[
    combined["hard_benign_proxy"] == True
].copy()

hard_benign.to_csv(
    OUT / "hard_benign_predictions.csv",
    index=False
)

# Prepare manual review without model predictions.

review_path = OUT / "hard_benign_manual_review.csv"

if not review_path.exists():

    review = hard_benign[
        ["test_index", "text", "label"]
    ].copy()

    review["attack_related_context"] = ""
    review["benign_label_valid"] = ""
    review["review_note"] = ""

    review.to_csv(
        review_path,
        index=False
    )

summary = []

for backbone, random_id, hard_id in [
    ("Snowflake", "E4", "E5"),
    ("MiniLM", "E8", "E9")
]:

    random_error = combined[f"{random_id}_error"]
    hard_error = combined[f"{hard_id}_error"]

    for label in [0, 1]:

        mask = combined["label"] == label

        improved = mask & random_error & ~hard_error
        degraded = mask & ~random_error & hard_error
        both_wrong = mask & random_error & hard_error

        summary.append({
            "backbone": backbone,
            "label": label,
            "improved_by_hnm": int(improved.sum()),
            "degraded_by_hnm": int(degraded.sum()),
            "both_wrong": int(both_wrong.sum())
        })

        combined[improved].to_csv(
            OUT / f"{backbone}_label{label}_improved.csv",
            index=False
        )

        combined[degraded].to_csv(
            OUT / f"{backbone}_label{label}_degraded.csv",
            index=False
        )

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    OUT / "hnm_error_transition.csv",
    index=False
)

print("\n===== ERROR TRANSITION =====")
print(summary_df.to_string(index=False))

print("\nHard benign review samples:", len(hard_benign))

print(
    "Samples misclassified by at least one model:",
    int((combined["error_count"] > 0).sum())
)

print("\nERROR ANALYSIS COMPLETE")
