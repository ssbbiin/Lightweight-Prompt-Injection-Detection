import sys
import time
import json
import yaml
import torch
import joblib
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from transformers import (
    set_seed,
    get_linear_schedule_with_warmup
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.metrics import calculate_metrics

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=["snowflake", "minilm"]
)

parser.add_argument(
    "--method",
    required=True,
    choices=["random", "hard", "hybrid", "model_aware"]
)

args = parser.parse_args()

MODEL = args.model
METHOD = args.method

EXPERIMENTS = {
    ("snowflake", "random"): "E4_snowflake_random_cl",
    ("snowflake", "hard"): "E5_snowflake_hard_cl",
    ("minilm", "random"): "E8_minilm_random_cl",
    ("minilm", "hard"): "E9_minilm_hard_cl",
    ("snowflake", "hybrid"): "E10_snowflake_hybrid_cl",
    ("minilm", "hybrid"): "E11_minilm_hybrid_cl",
    ("snowflake", "model_aware"): "E12_snowflake_model_aware_cl",
    ("minilm", "model_aware"): "E13_minilm_model_aware_cl"
}

EXPERIMENT = EXPERIMENTS[(MODEL, METHOD)]

with open("configs/models.yaml") as f:
    config = yaml.safe_load(f)

cfg = config["training"]

SEED = cfg["seed"]
EPOCHS = cfg["epochs"]
MAX_LENGTH = cfg["max_length"]
LR = cfg["learning_rate"]

MARGIN = config["contrastive"]["margin"]
BATCH_SIZE = 32

set_seed(SEED)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

torch.backends.cuda.matmul.allow_tf32 = True

OUT = Path("outputs/metrics") / EXPERIMENT

CKPT = Path("checkpoints") / EXPERIMENT / "best"

OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

with open(
    Path("data/embeddings") / MODEL / "metadata.json"
) as f:
    metadata = json.load(f)

MODEL_ID = metadata["model_id"]
REVISION = metadata["revision"]

assert metadata["max_length"] == MAX_LENGTH

print("\n===== EXPERIMENT CONFIG =====")
print("Experiment:", EXPERIMENT)
print("Model:", MODEL_ID)
print("Revision:", REVISION)
print("Method:", METHOD)
print("Batch size:", BATCH_SIZE)
print("Margin:", MARGIN)
print("Epochs:", EPOCHS)

# Always start from the original pretrained encoder.

model = SentenceTransformer(
    MODEL_ID,
    revision=REVISION,
    device="cuda",
    token=False,
    trust_remote_code=False,
    model_kwargs={
        "use_safetensors": True
    }
)

model.max_seq_length = MAX_LENGTH
model.default_prompt_name = None
model.requires_grad_(True)

encoder_params = sum(
    p.numel()
    for p in model.parameters()
)

train_df = pd.read_csv(
    "data/processed/train.csv",
    keep_default_na=False
)

val_df = pd.read_csv(
    "data/processed/validation.csv",
    keep_default_na=False
)

texts = train_df["text"].tolist()

y_train = train_df["label"].to_numpy()
y_val = val_df["label"].to_numpy()

triplets = pd.read_csv(
    Path("data/triplets")
    / MODEL
    / f"{METHOD}_triplets.csv"
)

triplet_indices = triplets[
    ["anchor_idx", "positive_idx", "negative_idx"]
].to_numpy(dtype=np.int64)

assert triplet_indices.shape == (len(train_df), 3)
assert triplet_indices.min() >= 0
assert triplet_indices.max() < len(train_df)

a, p, n = triplet_indices.T

assert np.all(a != p)
assert np.all(y_train[a] == y_train[p])
assert np.all(y_train[a] != y_train[n])

dataset = TensorDataset(
    torch.from_numpy(triplet_indices)
)

generator = torch.Generator()
generator.manual_seed(SEED)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
    num_workers=0
)

# AdamW optimizer.

decay = []
no_decay = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    if param.ndim < 2 or name.endswith(".bias"):
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": cfg["weight_decay"]
        },
        {
            "params": no_decay,
            "weight_decay": 0.0
        }
    ],
    lr=LR
)

total_steps = len(loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(
        total_steps * cfg["warmup_ratio"]
    ),
    num_training_steps=total_steps
)

print("\n===== TRAINING INFORMATION =====")
print("Encoder parameters:", encoder_params)
print("Triplets:", len(dataset))
print("Steps per epoch:", len(loader))
print("Total steps:", total_steps)

history = []
step_history = []

best_f1 = -1.0
best_epoch = None
best_metrics = None

total_optimization_time = 0.0

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start_total = time.perf_counter()

for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_loss = 0.0
    epoch_active = 0
    seen = 0

    torch.cuda.synchronize()
    start_epoch = time.perf_counter()

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for step, (batch_indices,) in enumerate(
        progress,
        start=1
    ):

        batch_indices = batch_indices.numpy()

        batch_texts = [
            texts[i]
            for i in batch_indices.reshape(-1)
        ]

        features = model.tokenize(batch_texts)

        features = {
            key: value.to("cuda")
            for key, value in features.items()
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):

            embeddings = model(
                features
            )["sentence_embedding"]

        embeddings = F.normalize(
            embeddings.float(),
            p=2,
            dim=-1
        )

        embeddings = embeddings.reshape(
            -1,
            3,
            embeddings.shape[-1]
        )

        anchor = embeddings[:, 0]
        positive = embeddings[:, 1]
        negative = embeddings[:, 2]

        positive_sim = (
            anchor * positive
        ).sum(dim=-1)

        negative_sim = (
            anchor * negative
        ).sum(dim=-1)

        losses = F.relu(
            negative_sim
            - positive_sim
            + MARGIN
        )

        loss = losses.mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()
        scheduler.step()

        batch_n = len(batch_indices)

        epoch_loss += loss.item() * batch_n
        epoch_active += int(
            (losses > 0).sum().item()
        )

        seen += batch_n

        if step % 25 == 0 or step == 1:

            step_history.append({
                "epoch": epoch,
                "step": step,
                "loss": float(loss.item()),
                "positive_similarity": float(
                    positive_sim.mean().item()
                ),
                "negative_similarity": float(
                    negative_sim.mean().item()
                ),
                "active_triplet_ratio": float(
                    (losses > 0).float().mean().item()
                ),
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                )
            })

        progress.set_postfix({
            "loss": f"{epoch_loss / seen:.4f}",
            "active": f"{epoch_active / seen:.3f}"
        })

    torch.cuda.synchronize()

    epoch_time = (
        time.perf_counter() - start_epoch
    )

    total_optimization_time += epoch_time

    mean_loss = epoch_loss / seen
    active_ratio = epoch_active / seen

    # Evaluate the learned embeddings after each epoch.
    # Train the classifier using training data only.

    model.eval()

    X_train = model.encode(
        train_df["text"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt="",
        show_progress_bar=False
    )

    X_val = model.encode(
        val_df["text"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt="",
        show_progress_bar=False
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=SEED
    )

    classifier.fit(
        X_train,
        y_train
    )

    probabilities = classifier.predict_proba(
        X_val
    )[:, 1]

    metrics = calculate_metrics(
        y_val,
        probabilities
    )

    epoch_summary = {
        "epoch": epoch,
        "triplet_loss": float(mean_loss),
        "active_triplet_ratio": float(active_ratio),
        "optimization_time_seconds": epoch_time,
        **metrics
    }

    history.append(epoch_summary)

    print(f"\n===== EPOCH {epoch} RESULTS =====")
    print(json.dumps(epoch_summary, indent=4))

    # Select the encoder and classifier as one model.

    if metrics["macro_f1"] > best_f1:

        best_f1 = metrics["macro_f1"]
        best_epoch = epoch
        best_metrics = metrics.copy()

        model.save(
            str(CKPT / "encoder"),
            safe_serialization=True
        )

        joblib.dump(
            classifier,
            CKPT / "classifier.joblib"
        )

        predictions = val_df.copy()

        predictions["attack_probability"] = (
            probabilities
        )

        predictions["prediction"] = (
            probabilities >= 0.5
        ).astype(int)

        predictions["correct"] = (
            predictions["prediction"]
            == predictions["label"]
        )

        predictions.to_csv(
            OUT / "validation_predictions.csv",
            index=False
        )

        print("Best model updated:", epoch)

    pd.DataFrame(history).to_csv(
        OUT / "training_history.csv",
        index=False
    )

    pd.DataFrame(step_history).to_csv(
        OUT / "step_history.csv",
        index=False
    )

torch.cuda.synchronize()

training_time = (
    time.perf_counter() - start_total
)

peak_allocated = (
    torch.cuda.max_memory_allocated() / 1024**3
)

peak_reserved = (
    torch.cuda.max_memory_reserved() / 1024**3
)

classifier_params = (
    classifier.coef_.size
    + classifier.intercept_.size
)

summary = {
    "experiment": EXPERIMENT,
    "model_id": MODEL_ID,
    "model_revision": REVISION,
    "method": METHOD,
    "encoder_parameters": encoder_params,
    "classifier_parameters": int(classifier_params),
    "total_parameters": int(
        encoder_params + classifier_params
    ),
    "trainable_parameters": int(
        encoder_params + classifier_params
    ),
    "triplets": len(dataset),
    "batch_size": BATCH_SIZE,
    "margin": MARGIN,
    "epochs": EPOCHS,
    "best_epoch": best_epoch,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "training_time_seconds": training_time,
    "optimization_time_seconds": (
        total_optimization_time
    ),
    "peak_allocated_vram_gib": peak_allocated,
    "peak_reserved_vram_gib": peak_reserved,
    "training_config": cfg,
    "validation_metrics": best_metrics
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n===== FINAL SUMMARY =====")
print(json.dumps(summary, indent=4))

print("\nBest model:", CKPT)
print(f"\n{EXPERIMENT} COMPLETE")
