import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("outputs/final")
OUT.mkdir(parents=True, exist_ok=True)

test = pd.read_csv(
    "outputs/evaluation_extended/test_results.csv"
)

latency = pd.read_csv(
    "outputs/latency_extended/inference_results.csv"
)

external = pd.read_csv(
    "outputs/external/notinject/notinject_overall.csv"
)

embedding = pd.read_csv(
    "outputs/analysis_extended/embedding_separation.csv"
)

trigger = pd.read_csv(
    "outputs/external/notinject/trigger_comparison.csv"
)

bs1 = latency[latency["batch_size"] == 1]
bs32 = latency[latency["batch_size"] == 32]

external_05 = external[
    external["operating_point"] == "default_05"
][["experiment", "fpr"]].rename(
    columns={"fpr": "notinject_fpr"}
)

external_val = external[
    external["operating_point"] == "validation_tpr95"
][["experiment", "fpr"]].rename(
    columns={"fpr": "notinject_calibrated_fpr"}
)

final = test.merge(
    bs1[[
        "experiment",
        "mean_batch_latency_ms",
        "p95_batch_latency_ms",
        "peak_allocated_vram_gib"
    ]],
    on="experiment",
    validate="one_to_one"
).merge(
    bs32[[
        "experiment",
        "throughput_samples_per_second"
    ]],
    on="experiment",
    validate="one_to_one"
).merge(
    external_05,
    on="experiment",
    validate="one_to_one"
).merge(
    external_val,
    on="experiment",
    validate="one_to_one"
).merge(
    embedding[["experiment", "silhouette_cosine"]],
    on="experiment",
    how="left",
    validate="one_to_one"
).merge(
    trigger,
    on="experiment",
    validate="one_to_one"
)

assert len(final) == 13

final["id"] = final["experiment"].str.split("_").str[0]
final["number"] = final["id"].str[1:].astype(int)
final = final.sort_values("number")

final.to_csv(
    OUT / "final_results.csv",
    index=False
)

names = {
    "E1": "ModernBERT CE",
    "E2": "Snowflake Frozen",
    "E3": "Snowflake CE",
    "E4": "Snowflake Random CL",
    "E5": "Snowflake Hard CL",
    "E6": "MiniLM Frozen",
    "E7": "MiniLM CE",
    "E8": "MiniLM Random CL",
    "E9": "MiniLM Hard CL",
    "E10": "Snowflake Hybrid",
    "E11": "MiniLM Hybrid",
    "E12": "Snowflake Model-Aware",
    "E13": "MiniLM Model-Aware"
}

methods = [
    "Frozen", "CE", "Random CL",
    "Hard CL", "Hybrid", "Model-Aware"
]

snowflake_ids = ["E2", "E3", "E4", "E5", "E10", "E12"]
minilm_ids = ["E6", "E7", "E8", "E9", "E11", "E13"]

indexed = final.set_index("id")

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


def compare(column, title, ylabel, filename, scale=1):

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(methods))

    a = indexed.loc[snowflake_ids, column] * scale
    b = indexed.loc[minilm_ids, column] * scale

    ax.bar(
        x - 0.18, a,
        width=0.36,
        label="Snowflake",
        color="#2563eb"
    )

    ax.bar(
        x + 0.18, b,
        width=0.36,
        label="MiniLM",
        color="#0d9488"
    )

    if column != "silhouette_cosine":
        baseline = indexed.loc["E1", column] * scale
        ax.axhline(
            baseline,
            linestyle="--",
            color="gray",
            label="ModernBERT CE"
        )

    ax.set_xticks(x, methods)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    save(fig, filename)


compare(
    "accuracy",
    "Test Accuracy",
    "Accuracy (%)",
    "01_accuracy", 100
)

compare(
    "fpr",
    "Internal Test False Positive Rate",
    "FPR (%)",
    "02_internal_fpr", 100
)

compare(
    "fpr_at_95_tpr_roc",
    "FPR at 95% Test TPR",
    "FPR (%)",
    "03_fpr_at_95_tpr", 100
)

compare(
    "notinject_fpr",
    "External Overdefense Evaluation",
    "NotInject FPR (%)",
    "04_notinject_fpr", 100
)

compare(
    "silhouette_cosine",
    "Embedding Class Separation",
    "Silhouette Score",
    "05_silhouette"
)

compare(
    "mean_batch_latency_ms",
    "Single-Prompt Inference Latency",
    "Latency (ms)",
    "06_latency"
)

compare(
    "throughput_samples_per_second",
    "Inference Throughput at Batch Size 32",
    "Prompts/sec",
    "07_throughput"
)

compare(
    "peak_allocated_vram_gib",
    "Peak GPU Memory at Batch Size 1",
    "VRAM (MiB)",
    "08_vram", 1024
)

# Trigger count analysis

fig, ax = plt.subplots(figsize=(10, 6))

for model_id in ["E1", "E4", "E5", "E10", "E12"]:

    row = indexed.loc[model_id]

    ax.plot(
        [1, 2, 3],
        row[["trigger_1", "trigger_2", "trigger_3"]]
        .astype(float).to_numpy(),
        marker="o",
        linewidth=2,
        label=names[model_id]
    )

ax.set_xticks([1, 2, 3])
ax.set_ylim(0, 100)
ax.set_xlabel("Number of Trigger Words")
ax.set_ylabel("False Positive Rate (%)")
ax.set_title("Trigger Word Count and Overdefense")
ax.grid(alpha=0.2)
ax.legend()

save(fig, "09_trigger_analysis")

# Internal versus external

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(final))

ax.bar(
    x - 0.18,
    final["fpr"] * 100,
    width=0.36,
    label="Internal Test",
    color="#64748b"
)

ax.bar(
    x + 0.18,
    final["notinject_fpr"] * 100,
    width=0.36,
    label="NotInject",
    color="#dc2626"
)

ax.set_xticks(x, final["id"])
ax.set_ylabel("FPR (%)")
ax.set_title("Internal vs External False Positive Rate")
ax.legend()
ax.grid(axis="y", alpha=0.2)

save(fig, "10_generalization_gap")

# Accuracy-latency trade-off

fig, ax = plt.subplots(figsize=(10, 6))

for model_id in ["E1", "E4", "E5", "E10", "E7", "E8", "E11"]:

    row = indexed.loc[model_id]

    ax.scatter(
        row["mean_batch_latency_ms"],
        row["accuracy"] * 100,
        s=100,
        label=names[model_id]
    )

ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy-Latency Trade-off")
ax.grid(alpha=0.2)
ax.legend(fontsize=9)

save(fig, "11_accuracy_latency")

print("\nFINAL RESULTS GENERATED")
print(final[[
    "id",
    "accuracy",
    "recall",
    "fpr",
    "fpr_at_95_tpr_roc",
    "notinject_fpr",
    "mean_batch_latency_ms",
    "silhouette_cosine"
]].round(5).to_string(index=False))
