import gc
import time
import json
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

OUT = Path("outputs/latency")
OUT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZES = [1, 16, 32]
WARMUP = 10
REPEATS = 3
N_SAMPLES = 512
SEED = 42

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

df = pd.read_csv(
    "data/processed/test.csv",
    keep_default_na=False
)

sample_indices = np.random.default_rng(SEED).choice(
    len(df),
    size=N_SAMPLES,
    replace=False
)

texts = df.iloc[sample_indices]["text"].tolist()

np.save(
    OUT / "benchmark_sample_indices.npy",
    sample_indices
)

metadata = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "precision": "fp32",
    "tf32": False,
    "max_length": MAX_LENGTH,
    "batch_sizes": BATCH_SIZES,
    "warmup": WARMUP,
    "repeats": REPEATS,
    "samples": N_SAMPLES,
    "seed": SEED,
    "timing_scope": (
        "tokenization, transfer, encoder, "
        "classifier, CPU probability output"
    )
}

with open(OUT / "benchmark_config.json", "w") as f:
    json.dump(metadata, f, indent=4)


def load_predictor(experiment, method):

    root = Path("checkpoints") / experiment

    with open(
        Path("outputs/metrics")
        / experiment
        / "summary.json"
    ) as f:
        summary = json.load(f)

    if method == "modernbert":

        ckpt = root / "best"

        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt)
        )

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                str(ckpt),
                attn_implementation="sdpa",
                reference_compile=False,
                use_safetensors=True
            )
            .to("cuda")
            .float()
            .eval()
        )

        @torch.inference_mode()
        def predict(batch):

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to("cuda")

            logits = model(**inputs).logits

            return torch.softmax(
                logits.float(),
                dim=-1
            )[:, 1].cpu().numpy()

        return predict

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

        classifier_path = (
            root / "classifier.joblib"
        )

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
    model.max_seq_length = MAX_LENGTH
    model.default_prompt_name = None

    if method == "ce":

        head = load_file(
            str(
                root / "best"
                / "classifier.safetensors"
            )
        )

        weight = head["weight"].float().cuda()
        bias = head["bias"].float().cuda()

    else:

        classifier = joblib.load(
            classifier_path
        )

        assert classifier.classes_.tolist() == [0, 1]

        weight = torch.tensor(
            classifier.coef_,
            dtype=torch.float32,
            device="cuda"
        )

        bias = torch.tensor(
            classifier.intercept_,
            dtype=torch.float32,
            device="cuda"
        )

    @torch.inference_mode()
    def predict(batch):

        features = model.tokenize(batch)

        features = {
            k: v.to("cuda")
            for k, v in features.items()
        }

        embeddings = model(
            features
        )["sentence_embedding"]

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=-1
        )

        logits = F.linear(
            embeddings,
            weight,
            bias
        )

        if method == "ce":

            probabilities = torch.softmax(
                logits,
                dim=-1
            )[:, 1]

        else:

            probabilities = torch.sigmoid(
                logits
            ).reshape(-1)

        return probabilities.cpu().numpy()

    return predict


results = []
raw_results = []

for experiment, method in EXPERIMENTS:

    print("\n" + "=" * 60)
    print("EXPERIMENT:", experiment)
    print("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()

    predict = load_predictor(
        experiment,
        method
    )

    # Verify that the benchmark reproduces
    # the original evaluation probabilities.

    reference = pd.read_csv(
        Path("outputs/evaluation")
        / experiment
        / "test_predictions.csv",
        keep_default_na=False
    )

    assert reference["text"].tolist() == df["text"].tolist()

    expected = reference.iloc[
        sample_indices[:32]
    ]["attack_probability"].to_numpy()

    actual = predict(texts[:32])

    assert np.isfinite(actual).all()

    max_error = float(
        np.abs(expected - actual).max()
    )

    print("Probability max error:", max_error)

    if max_error > 1e-3:
        print(
            "WARNING: Probability mismatch. "
            "Inspect before interpreting latency."
        )

    for batch_size in BATCH_SIZES:

        batches = [
            texts[i:i + batch_size]
            for i in range(
                0,
                len(texts),
                batch_size
            )
        ]

        # Warm-up
        for i in range(WARMUP):

            predict(
                batches[i % len(batches)]
            )

        torch.cuda.synchronize()

        baseline_memory = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        torch.cuda.reset_peak_memory_stats()

        latencies = []

        for repeat in range(REPEATS):

            progress = tqdm(
                batches,
                desc=(
                    f"BS={batch_size} "
                    f"Repeat={repeat + 1}"
                ),
                leave=False
            )

            for batch in progress:

                torch.cuda.synchronize()

                start = time.perf_counter()

                probabilities = predict(batch)

                torch.cuda.synchronize()

                elapsed = (
                    time.perf_counter() - start
                )

                assert len(probabilities) == len(batch)

                latency_ms = elapsed * 1000

                latencies.append(latency_ms)

                raw_results.append({
                    "experiment": experiment,
                    "batch_size": batch_size,
                    "repeat": repeat + 1,
                    "samples": len(batch),
                    "latency_ms": latency_ms
                })

        peak_allocated = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        peak_reserved = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        latencies = np.asarray(latencies)

        total_samples = (
            N_SAMPLES * REPEATS
        )

        total_seconds = (
            latencies.sum() / 1000
        )

        result = {
            "experiment": experiment,
            "batch_size": batch_size,
            "samples": total_samples,
            "measured_batches": len(latencies),
            "mean_batch_latency_ms": float(
                latencies.mean()
            ),
            "p50_batch_latency_ms": float(
                np.percentile(latencies, 50)
            ),
            "p95_batch_latency_ms": float(
                np.percentile(latencies, 95)
            ),
            "std_batch_latency_ms": float(
                latencies.std()
            ),
            "amortized_ms_per_sample": float(
                latencies.sum() / total_samples
            ),
            "throughput_samples_per_second": (
                total_samples / total_seconds
            ),
            "baseline_vram_gib": baseline_memory,
            "peak_allocated_vram_gib": peak_allocated,
            "peak_reserved_vram_gib": peak_reserved,
            "incremental_peak_vram_gib": (
                peak_allocated - baseline_memory
            ),
            "probability_max_error": max_error
        }

        results.append(result)

        print(f"\n===== BATCH SIZE {batch_size} =====")
        print(json.dumps(result, indent=4))

        pd.DataFrame(results).to_csv(
            OUT / "inference_results.csv",
            index=False
        )

        pd.DataFrame(raw_results).to_csv(
            OUT / "raw_latency_measurements.csv",
            index=False
        )

    del predict

    gc.collect()
    torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

cols = [
    "experiment",
    "batch_size",
    "mean_batch_latency_ms",
    "p95_batch_latency_ms",
    "amortized_ms_per_sample",
    "throughput_samples_per_second",
    "peak_allocated_vram_gib"
]

print("\n===== FINAL BENCHMARK RESULTS =====")

print(
    results_df[cols]
    .round(4)
    .to_string(index=False)
)

print("\nINFERENCE BENCHMARK COMPLETE")
