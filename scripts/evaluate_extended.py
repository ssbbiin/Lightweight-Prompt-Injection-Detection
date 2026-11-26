import gc
import sys
import json
import math
import shutil
import torch
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import roc_curve
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path.cwd()))

from src.metrics import calculate_metrics

ROOT = Path("outputs/evaluation")
OUT = Path("outputs/evaluation_extended")

shutil.copytree(ROOT, OUT, dirs_exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

EXPERIMENTS = [
    "E10_snowflake_hybrid_cl",
    "E11_minilm_hybrid_cl",
    "E12_snowflake_model_aware_cl",
    "E13_minilm_model_aware_cl"
]

REFERENCE = ROOT / "E4_snowflake_random_cl"

test = pd.read_csv(
    REFERENCE / "test_predictions.csv",
    keep_default_na=False
)

val = pd.read_csv(
    REFERENCE / "validation_predictions.csv",
    keep_default_na=False
)

test_texts = test["text"].tolist()
val_texts = val["text"].tolist()

y_test = test["label"].to_numpy()
y_val = val["label"].to_numpy()

hard_mask = (
    test["hard_benign_proxy"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
    .to_numpy()
)

remaining_mask = (y_test == 0) & ~hard_mask

assert len(test) == 2101
assert len(val) == 1985
assert hard_mask.sum() == 65
assert np.all(y_test[hard_mask] == 0)


def predict(experiment, texts):

    root = Path("checkpoints") / experiment / "best"

    model = SentenceTransformer(
        str(root / "encoder"),
        device="cuda",
        trust_remote_code=False
    )

    model.float().eval()
    model.max_seq_length = 128
    model.default_prompt_name = None

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt="",
        show_progress_bar=True
    )

    classifier = joblib.load(
        root / "classifier.joblib"
    )

    assert classifier.classes_.tolist() == [0, 1]

    probs = classifier.predict_proba(
        embeddings
    )[:, 1]

    assert np.isfinite(probs).all()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return probs


def subset_fpr(probabilities, mask, threshold):

    return float(
        (probabilities[mask] >= threshold).mean()
    )


def threshold_for_recall(labels, probabilities):

    positive = np.sort(
        probabilities[labels == 1]
    )

    required = math.ceil(
        0.95 * len(positive)
    )

    return float(positive[-required])


def fpr_at_95_tpr(labels, probabilities):

    fpr, tpr, _ = roc_curve(
        labels,
        probabilities
    )

    return float(fpr[tpr >= 0.95].min())


# Verify consistency with the original evaluation.

print("\n===== REFERENCE CHECK =====")

reference_probs = predict(
    "E4_snowflake_random_cl",
    test_texts
)

error = np.abs(
    reference_probs
    - test["attack_probability"].to_numpy()
).max()

print("Reference probability max error:", error)

assert error < 1e-3, (
    "Reference evaluation mismatch"
)


test_results = []
threshold_results = []

for experiment in EXPERIMENTS:

    print("\n" + "=" * 60)
    print(experiment)
    print("=" * 60)

    with open(
        Path("outputs/metrics")
        / experiment
        / "summary.json"
    ) as f:
        training = json.load(f)

    val_probs = predict(
        experiment,
        val_texts
    )

    test_probs = predict(
        experiment,
        test_texts
    )

    threshold = threshold_for_recall(
        y_val,
        val_probs
    )

    default_metrics = calculate_metrics(
        y_test,
        test_probs,
        threshold=0.5
    )

    operating_metrics = calculate_metrics(
        y_test,
        test_probs,
        threshold=threshold
    )

    validation_metrics = calculate_metrics(
        y_val,
        val_probs,
        threshold=threshold
    )

    default_row = {
        "experiment": experiment,
        "model_id": training["model_id"],
        "total_parameters": training[
            "total_parameters"
        ],
        "training_time_seconds": training[
            "training_time_seconds"
        ],
        "peak_training_vram_gib": training[
            "peak_allocated_vram_gib"
        ],
        **default_metrics,
        "hard_benign_fpr": subset_fpr(
            test_probs, hard_mask, 0.5
        ),
        "remaining_benign_fpr": subset_fpr(
            test_probs, remaining_mask, 0.5
        ),
        "hard_benign_count": int(
            hard_mask.sum()
        ),
        "fpr_at_95_tpr_roc": fpr_at_95_tpr(
            y_test, test_probs
        )
    }

    operating_row = {
        "experiment": experiment,
        **operating_metrics,
        "validation_recall": validation_metrics[
            "recall"
        ],
        "validation_fpr": validation_metrics[
            "fpr"
        ],
        "hard_benign_fpr": subset_fpr(
            test_probs, hard_mask, threshold
        ),
        "remaining_benign_fpr": subset_fpr(
            test_probs, remaining_mask, threshold
        )
    }

    test_results.append(default_row)
    threshold_results.append(operating_row)

    exp_out = OUT / experiment
    exp_out.mkdir(parents=True, exist_ok=True)

    prediction = test[
        ["text", "label", "hard_benign_proxy"]
    ].copy()

    prediction["attack_probability"] = test_probs

    prediction["prediction_05"] = (
        test_probs >= 0.5
    ).astype(int)

    prediction["prediction_calibrated"] = (
        test_probs >= threshold
    ).astype(int)

    prediction["correct_05"] = (
        prediction["prediction_05"] == y_test
    )

    prediction.to_csv(
        exp_out / "test_predictions.csv",
        index=False
    )

    val_prediction = val[
        ["text", "label"]
    ].copy()

    val_prediction["attack_probability"] = (
        val_probs
    )

    val_prediction.to_csv(
        exp_out / "validation_predictions.csv",
        index=False
    )

    with open(exp_out / "summary.json", "w") as f:

        json.dump({
            "experiment": experiment,
            "default_metrics": default_row,
            "operating_metrics": operating_row
        }, f, indent=4, default=float)

    print(
        json.dumps(
            default_row,
            indent=4,
            default=float
        )
    )


original_test = pd.read_csv(
    ROOT / "test_results.csv"
)

original_thresholds = pd.read_csv(
    ROOT / "operating_threshold_results.csv"
)

all_test = pd.concat([
    original_test,
    pd.DataFrame(test_results)
], ignore_index=True)

all_thresholds = pd.concat([
    original_thresholds,
    pd.DataFrame(threshold_results)
], ignore_index=True)

assert len(all_test) == 13
assert all_test["experiment"].is_unique

all_test.to_csv(
    OUT / "test_results.csv",
    index=False
)

all_thresholds.to_csv(
    OUT / "operating_threshold_results.csv",
    index=False
)

cols = [
    "experiment",
    "accuracy",
    "macro_f1",
    "recall",
    "fpr",
    "roc_auc",
    "hard_benign_fpr",
    "fpr_at_95_tpr_roc"
]

display = all_test[cols].copy()

for col in [
    "accuracy",
    "macro_f1",
    "recall",
    "fpr",
    "hard_benign_fpr",
    "fpr_at_95_tpr_roc"
]:
    display[col] *= 100

print("\n===== ALL 13 EXPERIMENTS =====")

print(
    display.round(4)
    .to_string(index=False)
)

print("\nEXTENDED EVALUATION COMPLETE")
