import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(
    "outputs/evaluation/test_results.csv"
)

latency = pd.read_csv(
    "outputs/latency/inference_results.csv"
)

test["id"] = test["experiment"].str.split("_").str[0]
latency["id"] = latency["experiment"].str.split("_").str[0]

names = {
    "E1": "ModernBERT CE",
    "E2": "Snowflake Frozen",
    "E3": "Snowflake CE",
    "E4": "Snowflake Random CL",
    "E5": "Snowflake Hard CL",
    "E6": "MiniLM Frozen",
    "E7": "MiniLM CE",
    "E8": "MiniLM Random CL",
    "E9": "MiniLM Hard CL"
}

colors = {
    "E1": "#64748b",
    "E2": "#3b82f6",
    "E3": "#3b82f6",
    "E4": "#3b82f6",
    "E5": "#3b82f6",
    "E6": "#0d9488",
    "E7": "#0d9488",
    "E8": "#0d9488",
    "E9": "#0d9488"
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False
})


def save(fig, name):

    fig.tight_layout()

    for ext in ["png", "pdf"]:

        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )

    plt.close(fig)


def barplot(
    data,
    column,
    title,
    xlabel,
    filename,
    scale=1,
    unit=""
):

    fig, ax = plt.subplots(
        figsize=(10, 5.5)
    )

    values = data[column].to_numpy() * scale

    bars = ax.barh(
        [names[i] for i in data["id"]],
        values,
        color=[colors[i] for i in data["id"]]
    )

    ax.invert_yaxis()

    ax.bar_label(
        bars,
        labels=[f"{v:.2f}{unit}" for v in values],
        padding=4,
        fontsize=10
    )

    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.2)

    save(fig, filename)


barplot(
    test, "accuracy",
    "Test Accuracy",
    "Accuracy (%)",
    "01_test_accuracy",
    100, "%"
)

barplot(
    test, "fpr",
    "False Positive Rate",
    "FPR (%)",
    "02_test_fpr",
    100, "%"
)

# Hard benign vs remaining benign

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(test))

ax.bar(
    x - 0.18,
    test["hard_benign_fpr"] * 100,
    width=0.36,
    label="Keyword-containing benign (n=65)",
    color="#dc2626"
)

ax.bar(
    x + 0.18,
    test["remaining_benign_fpr"] * 100,
    width=0.36,
    label="Remaining benign (n=985)",
    color="#94a3b8"
)

ax.set_xticks(x, test["id"])
ax.set_ylabel("False Positive Rate (%)")
ax.set_title("Hard Benign Proxy Evaluation")
ax.legend()
ax.grid(axis="y", alpha=0.2)

save(fig, "03_hard_benign_fpr")

# ROC curves

fig, ax = plt.subplots(figsize=(8, 7))

for _, row in test.iterrows():

    pred = pd.read_csv(
        Path("outputs/evaluation")
        / row["experiment"]
        / "test_predictions.csv"
    )

    fpr, tpr, _ = roc_curve(
        pred["label"],
        pred["attack_probability"]
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=1.6,
        label=(
            f'{row["id"]} {names[row["id"]]} '
            f'({row["roc_auc"]:.4f})'
        )
    )

ax.plot(
    [0, 1], [0, 1],
    "--", color="gray", alpha=0.5
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.2)

save(fig, "04_roc_curve")

barplot(
    test, "fpr_at_95_tpr_roc",
    "FPR at 95% Test TPR",
    "False Positive Rate (%)",
    "05_fpr_at_95_tpr",
    100, "%"
)

bs1 = latency[
    latency["batch_size"] == 1
].copy()

barplot(
    bs1, "mean_batch_latency_ms",
    "Single-Prompt Inference Latency",
    "Latency (ms)",
    "06_inference_latency",
    1, " ms"
)

# Throughput

fig, ax = plt.subplots(figsize=(10, 6))

for model_id in names:

    sub = latency[
        latency["id"] == model_id
    ].sort_values("batch_size")

    ax.plot(
        sub["batch_size"].astype(str),
        sub["throughput_samples_per_second"],
        marker="o",
        label=f"{model_id} {names[model_id]}"
    )

ax.set_xlabel("Batch Size")
ax.set_ylabel("Prompts per Second")
ax.set_title("Inference Throughput Comparison")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

save(fig, "07_throughput")

# Accuracy-latency tradeoff

merged = test.merge(
    bs1[["id", "mean_batch_latency_ms"]],
    on="id",
    validate="one_to_one"
)

fig, ax = plt.subplots(figsize=(9, 6))

for _, row in merged.iterrows():

    x = row["mean_batch_latency_ms"]
    y = row["accuracy"] * 100

    ax.scatter(
        x, y,
        s=110,
        color=colors[row["id"]]
    )

    ax.annotate(
        row["id"],
        (x, y),
        xytext=(7, 5),
        textcoords="offset points"
    )

ax.set_xlabel("Single-Prompt Latency (ms)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Accuracy-Latency Trade-off")
ax.grid(alpha=0.25)

save(fig, "08_accuracy_latency_tradeoff")

barplot(
    bs1, "peak_allocated_vram_gib",
    "Peak Inference GPU Memory",
    "Allocated VRAM (MiB)",
    "09_inference_vram",
    1024, " MiB"
)

print("\nALL FIGURES GENERATED")
