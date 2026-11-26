import sys
import gc
import re
import json
import torch
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import roc_curve
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.metrics import calculate_metrics

OUT = Path("outputs/evaluation")
OUT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZE = 32
TARGET_RECALL = 0.95

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

EXPERIMENTS = [
    ("E1_modernbert_ce", "modernbert"),
    ("E2_snowflake_frozen", "frozen"),
    ("E3_snowflake_ce", "ce"),
    ("E4_snowflake_random_cl", "contrastive"),
    ("E5_snowflake_hard_cl", "contrastive"),
    ("E6_minilm_frozen", "frozen"),
    ("E7_minilm_ce", "ce"),
    ("E8_minilm_random_cl", "contrastive"),
    ("E9_minilm_hard_cl", "contrastive")
]

val_df = pd.read_csv(
    "data/processed/validation.csv",
    keep_default_na=False
)

test_df = pd.read_csv(
    "data/processed/test.csv",
    keep_default_na=False
)

raw_test = pd.read_csv(
    "data/raw/test.csv",
    keep_default_na=False
)

assert test_df.equals(raw_test)
assert len(test_df) == 2101

y_val = val_df["label"].to_numpy()
y_test = test_df["label"].to_numpy()

# Fixed keyword-based hard benign proxy.
# Independent of model predictions.

PATTERN = re.compile(
    r"\b("
    r"prompt(s)?|"
    r"instruction(s)?|"
    r"system|"
    r"jailbreak(s|ing)?|"
    r"inject(ion|ions|ing)?|"
    r"bypass(es|ing)?|"
    r"override(s)?|"
    r"ignore|"
    r"hidden|"
    r"developer|"
    r"reveal|"
    r"debug(ging)?|"
    r"security|"
    r"attack(s)?"
    r")\b",
    flags=re.IGNORECASE
)

hard_mask = (
    (y_test == 0)
    & test_df["text"].str.contains(
        PATTERN,
        regex=True
    ).to_numpy()
)

easy_mask = (
    (y_test == 0)
    & ~hard_mask
)

test_df["hard_benign_proxy"] = hard_mask

test_df.loc[
    hard_mask, ["text", "label"]
].to_csv(
    OUT / "hard_benign_review.csv",
    index=False
)

print("\n===== EVALUATION DATASET =====")
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Benign:", int((y_test == 0).sum()))
print("Injection:", int((y_test == 1).sum()))
print("Hard benign proxy:", int(hard_mask.sum()))
print("Remaining benign:", int(easy_mask.sum()))


def threshold_for_recall(
    labels,
    probabilities,
    target=0.95
):

    positive_scores = np.sort(
        probabilities[labels == 1]
    )

    required = int(
        np.ceil(target * len(positive_scores))
    )

    threshold = positive_scores[-required]

    return float(threshold)


def roc_fpr_at_tpr(
    labels,
    probabilities,
    target=0.95
):

    fpr, tpr, thresholds = roc_curve(
        labels,
        probabilities,
        drop_intermediate=False
    )

    valid = tpr >= target

    return float(fpr[valid].min())


def subgroup_fpr(
    probabilities,
    mask,
    threshold
):

    if mask.sum() == 0:
        return float("nan")

    return float(
        (probabilities[mask] >= threshold).mean()
    )


def modernbert_scores(experiment):

    ckpt = Path("checkpoints") / experiment / "best"

    tokenizer = AutoTokenizer.from_pretrained(
        str(ckpt)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt),
        attn_implementation="sdpa",
        reference_compile=False,
        use_safetensors=True
    )

    model = model.to("cuda").float().eval()

    def predict(texts):

        scores = []

        with torch.inference_mode():

            for start in tqdm(
                range(0, len(texts), BATCH_SIZE),
                desc="ModernBERT inference"
            ):

                batch = texts[
                    start:start + BATCH_SIZE
                ]

                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to("cuda")

                logits = model(**inputs).logits

                probs = torch.softmax(
                    logits.float(),
                    dim=-1
                )[:, 1]

                scores.extend(
                    probs.cpu().numpy()
                )

        return np.asarray(
            scores,
            dtype=np.float64
        )

    val_scores = predict(
        val_df["text"].tolist()
    )

    test_scores = predict(
        test_df["text"].tolist()
    )

    return val_scores, test_scores


def lightweight_scores(
    experiment,
    method,
    summary
):

    root = Path("checkpoints") / experiment

    if method == "frozen":

        model = SentenceTransformer(
            summary["model_id"],
            revision=summary["model_revision"],
            device="cuda",
            token=False,
            trust_remote_code=False,
            model_kwargs={
                "use_safetensors": True
            }
        )

        classifier = joblib.load(
            root / "classifier.joblib"
        )

    else:

        model = SentenceTransformer(
            str(root / "best" / "encoder"),
            device="cuda",
            trust_remote_code=False
        )

        if method == "contrastive":

            classifier = joblib.load(
                root / "best" / "classifier.joblib"
            )

        elif method == "ce":

            head = load_file(
                str(
                    root
                    / "best"
                    / "classifier.safetensors"
                )
            )

            weight = (
                head["weight"]
                .float()
                .cpu()
                .numpy()
            )

            bias = (
                head["bias"]
                .float()
                .cpu()
                .numpy()
            )

    model.float()
    model.eval()

    model.max_seq_length = MAX_LENGTH
    model.default_prompt_name = None

    def predict(texts):

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            prompt=""
        )

        if method == "ce":

            logits = (
                embeddings @ weight.T
                + bias
            )

            probabilities = softmax(
                logits,
                axis=-1
            )[:, 1]

        else:

            probabilities = (
                classifier.predict_proba(
                    embeddings
                )[:, 1]
            )

        return probabilities.astype(
            np.float64
        )

    val_scores = predict(
        val_df["text"].tolist()
    )

    test_scores = predict(
        test_df["text"].tolist()
    )

    return val_scores, test_scores


results = []
operating_results = []

for experiment, method in EXPERIMENTS:

    print("\n" + "=" * 60)
    print("EXPERIMENT:", experiment)
    print("=" * 60)

    summary_path = (
        Path("outputs/metrics")
        / experiment
        / "summary.json"
    )

    with open(summary_path) as f:
        summary = json.load(f)

    if method == "modernbert":

        val_scores, test_scores = (
            modernbert_scores(experiment)
        )

    else:

        val_scores, test_scores = (
            lightweight_scores(
                experiment,
                method,
                summary
            )
        )

    assert len(val_scores) == len(val_df)
    assert len(test_scores) == len(test_df)
    assert np.isfinite(val_scores).all()
    assert np.isfinite(test_scores).all()

    threshold = threshold_for_recall(
        y_val,
        val_scores,
        TARGET_RECALL
    )

    # Default threshold evaluation.

    metrics = calculate_metrics(
        y_test,
        test_scores,
        threshold=0.5
    )

    metrics["hard_benign_fpr"] = subgroup_fpr(
        test_scores,
        hard_mask,
        0.5
    )

    metrics["remaining_benign_fpr"] = subgroup_fpr(
        test_scores,
        easy_mask,
        0.5
    )

    metrics["hard_benign_count"] = int(
        hard_mask.sum()
    )

    metrics["fpr_at_95_tpr_roc"] = (
        roc_fpr_at_tpr(
            y_test,
            test_scores,
            TARGET_RECALL
        )
    )

    # Validation-selected operating threshold.

    op_metrics = calculate_metrics(
        y_test,
        test_scores,
        threshold=threshold
    )

    val_metrics = calculate_metrics(
        y_val,
        val_scores,
        threshold=threshold
    )

    op_metrics["validation_recall"] = (
        val_metrics["recall"]
    )

    op_metrics["validation_fpr"] = (
        val_metrics["fpr"]
    )

    op_metrics["hard_benign_fpr"] = (
        subgroup_fpr(
            test_scores,
            hard_mask,
            threshold
        )
    )

    op_metrics["remaining_benign_fpr"] = (
        subgroup_fpr(
            test_scores,
            easy_mask,
            threshold
        )
    )

    result = {
        "experiment": experiment,
        "model_id": summary["model_id"],
        "total_parameters": summary["total_parameters"],
        "training_time_seconds": summary[
            "training_time_seconds"
        ],
        "peak_training_vram_gib": summary[
            "peak_allocated_vram_gib"
        ],
        **metrics
    }

    results.append(result)

    operating_results.append({
        "experiment": experiment,
        **op_metrics
    })

    exp_dir = OUT / experiment
    exp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = test_df.copy()

    predictions["attack_probability"] = (
        test_scores
    )

    predictions["prediction_05"] = (
        test_scores >= 0.5
    ).astype(int)

    predictions["prediction_calibrated"] = (
        test_scores >= threshold
    ).astype(int)

    predictions["correct_05"] = (
        predictions["prediction_05"]
        == predictions["label"]
    )

    predictions.to_csv(
        exp_dir / "test_predictions.csv",
        index=False
    )

    val_predictions = val_df.copy()

    val_predictions["attack_probability"] = (
        val_scores
    )

    val_predictions.to_csv(
        exp_dir / "validation_predictions.csv",
        index=False
    )

    evaluation_summary = {
        "experiment": experiment,
        "inference_precision": "float32",
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "test_samples": len(test_df),
        "default_metrics": metrics,
        "validation_selected_threshold": threshold,
        "operating_metrics": op_metrics
    }

    with open(
        exp_dir / "summary.json", "w"
    ) as f:

        json.dump(
            evaluation_summary,
            f,
            indent=4
        )

    print("\nDefault threshold metrics:")
    print(json.dumps(metrics, indent=4))

    print("\nValidation-selected threshold:")
    print(json.dumps(op_metrics, indent=4))

    gc.collect()
    torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(
        OUT / "test_results.csv",
        index=False
    )

    pd.DataFrame(operating_results).to_csv(
        OUT / "operating_threshold_results.csv",
        index=False
    )

# Display final summary.

results_df = pd.DataFrame(results)

columns = [
    "experiment",
    "accuracy",
    "macro_f1",
    "recall",
    "fpr",
    "hard_benign_fpr",
    "roc_auc",
    "fpr_at_95_tpr_roc",
    "fp",
    "fn"
]

print("\n===== FINAL TEST RESULTS =====")

print(
    results_df[columns].round(6)
    .to_string(index=False)
)

print("\n===== OPERATING THRESHOLD RESULTS =====")

print(
    pd.DataFrame(operating_results)[[
        "experiment",
        "threshold",
        "validation_recall",
        "recall",
        "fpr",
        "hard_benign_fpr"
    ]].round(6).to_string(index=False)
)

print("\nALL 9 EXPERIMENTS EVALUATED")
