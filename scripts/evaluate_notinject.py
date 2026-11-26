import gc
import json
import torch
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.special import softmax
from safetensors.torch import load_file

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer

OUT = Path("outputs/external/notinject")
OUT.mkdir(parents=True, exist_ok=True)

ROOT = Path("outputs/evaluation_extended")

df = pd.read_csv(
    "data/external/notinject/notinject.csv",
    keep_default_na=False
)

experiments = pd.read_csv(
    ROOT / "test_results.csv"
)

thresholds = pd.read_csv(
    ROOT / "operating_threshold_results.csv"
).set_index("experiment")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@torch.inference_mode()
def predict(experiment, texts):

    root = Path("checkpoints") / experiment

    with open(
        Path("outputs/metrics")
        / experiment / "summary.json"
    ) as f:
        summary = json.load(f)

    if "modernbert" in experiment:

        tokenizer = AutoTokenizer.from_pretrained(
            str(root / "best")
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(root / "best"),
            attn_implementation="sdpa",
            reference_compile=False,
            use_safetensors=True
        ).cuda().float().eval()

        results = []

        for start in range(0, len(texts), 32):

            inputs = tokenizer(
                texts[start:start + 32],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to("cuda")

            logits = model(**inputs).logits

            probabilities = torch.softmax(
                logits.float(),
                dim=-1
            )[:, 1]

            results.extend(
                probabilities.cpu().numpy()
            )

        return np.asarray(results)

    if experiment.endswith("_frozen"):

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

        classifier_path = root / "classifier.joblib"

    else:

        model = SentenceTransformer(
            str(root / "best" / "encoder"),
            device="cuda",
            trust_remote_code=False
        )

        classifier_path = (
            root / "best" / "classifier.joblib"
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
        show_progress_bar=False
    )

    if experiment.endswith("_ce"):

        head = load_file(
            str(
                root / "best"
                / "classifier.safetensors"
            )
        )

        weight = head["weight"].numpy()
        bias = head["bias"].numpy()

        logits = embeddings @ weight.T + bias

        probabilities = softmax(
            logits,
            axis=1
        )[:, 1]

    else:

        classifier = joblib.load(
            classifier_path
        )

        assert classifier.classes_.tolist() == [0, 1]

        probabilities = classifier.predict_proba(
            embeddings
        )[:, 1]

    return probabilities


def summarize(mask, prediction):

    n = int(mask.sum())

    fp = int(prediction[mask].sum())

    return {
        "samples": n,
        "fp": fp,
        "fpr": fp / n if n else np.nan,
        "benign_accuracy": 1 - fp / n if n else np.nan
    }


results = []

for experiment in experiments["experiment"]:

    print("\nExperiment:", experiment)

    # Check against existing evaluation.

    reference = pd.read_csv(
        ROOT / experiment / "test_predictions.csv",
        keep_default_na=False
    ).iloc[:32]

    texts = (
        reference["text"].tolist()
        + df["text"].tolist()
    )

    scores = predict(experiment, texts)

    error = np.abs(
        scores[:32]
        - reference["attack_probability"].to_numpy()
    ).max()

    print("Reference probability max error:", error)

    assert error < 1e-3

    probs = scores[32:]

    assert len(probs) == len(df)
    assert np.isfinite(probs).all()

    threshold = float(
        thresholds.loc[experiment, "threshold"]
    )

    prediction = df.copy()
    prediction["attack_probability"] = probs

    prediction["prediction_05"] = (
        probs >= 0.5
    ).astype(int)

    prediction["prediction_calibrated"] = (
        probs >= threshold
    ).astype(int)

    prediction.to_csv(
        OUT / f"{experiment}_predictions.csv",
        index=False
    )

    masks = {
        "overall": np.ones(len(df), dtype=bool),
        "trigger_1": (
            df["trigger_count"] == 1
        ).to_numpy(),
        "trigger_2": (
            df["trigger_count"] == 2
        ).to_numpy(),
        "trigger_3": (
            df["trigger_count"] == 3
        ).to_numpy(),
        "no_original_overlap": (
            ~df["overlap_original"]
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
        ).to_numpy()
    }

    for category in df["category"].unique():

        masks[f"category_{category}"] = (
            df["category"] == category
        ).to_numpy()

    for operating_point, th in [
        ("default_05", 0.5),
        ("validation_tpr95", threshold)
    ]:

        pred = (probs >= th).astype(int)

        for subset, mask in masks.items():

            row = {
                "experiment": experiment,
                "operating_point": operating_point,
                "threshold": th,
                "subset": subset,
                **summarize(mask, pred)
            }

            results.append(row)

            if subset == "overall":
                print(row)

    gc.collect()
    torch.cuda.empty_cache()

out = pd.DataFrame(results)

out.to_csv(
    OUT / "notinject_results.csv",
    index=False
)

overall = out[
    out["subset"] == "overall"
].copy()

overall.to_csv(
    OUT / "notinject_overall.csv",
    index=False
)

print("\n===== NOTINJECT OVERALL RESULTS =====")

print(
    overall.round(5).to_string(index=False)
)

print("\nNOTINJECT EVALUATION COMPLETE")
