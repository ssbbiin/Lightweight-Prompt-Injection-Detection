import gc
import json
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

OUT = Path("outputs/figures")
ANALYSIS = Path("outputs/analysis")

OUT.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

SEED = 42

EXPERIMENTS = [
    ("E2_snowflake_frozen", "Frozen"),
    ("E3_snowflake_ce", "CE"),
    ("E4_snowflake_random_cl", "Random CL"),
    ("E5_snowflake_hard_cl", "Hard CL"),
    ("E6_minilm_frozen", "Frozen"),
    ("E7_minilm_ce", "CE"),
    ("E8_minilm_random_cl", "Random CL"),
    ("E9_minilm_hard_cl", "Hard CL")
]

df = pd.read_csv(
    "data/processed/test.csv",
    keep_default_na=False
)

sample = pd.concat([
    df[df["label"] == 0].sample(
        n=300, random_state=SEED
    ),
    df[df["label"] == 1].sample(
        n=300, random_state=SEED
    )
]).sample(frac=1, random_state=SEED)

labels = sample["label"].to_numpy()
texts = sample["text"].tolist()

sample.to_csv(
    ANALYSIS / "tsne_samples.csv",
    index=True,
    index_label="test_index"
)

fig, axes = plt.subplots(
    2, 4,
    figsize=(17, 8)
)

results = []

for i, (experiment, method) in enumerate(EXPERIMENTS):

    print("\nExperiment:", experiment)

    with open(
        Path("outputs/metrics")
        / experiment
        / "summary.json"
    ) as f:
        summary = json.load(f)

    if method == "Frozen":

        model = SentenceTransformer(
            summary["model_id"],
            revision=summary["model_revision"],
            device="cuda",
            token=False,
            trust_remote_code=False
        )

    else:

        model = SentenceTransformer(
            str(
                Path("checkpoints")
                / experiment
                / "best"
                / "encoder"
            ),
            device="cuda",
            trust_remote_code=False
        )

    model.float().eval()
    model.max_seq_length = 128
    model.default_prompt_name = None

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        prompt="",
        show_progress_bar=True
    )

    assert np.isfinite(embeddings).all()

    # Measure class separation in the original
    # 384-dimensional embedding space.

    score = silhouette_score(
        embeddings,
        labels,
        metric="cosine"
    )

    coords = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=SEED
    ).fit_transform(embeddings)

    coords_df = pd.DataFrame({
        "test_index": sample.index,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels
    })

    coords_df.to_csv(
        ANALYSIS / f"{experiment}_tsne.csv",
        index=False
    )

    ax = axes.flat[i]

    for label, color, name in [
        (0, "#2563eb", "Benign"),
        (1, "#dc2626", "Injection")
    ]:

        mask = labels == label

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.65,
            color=color,
            label=name,
            edgecolors="none"
        )

    backbone = (
        "Snowflake" if i < 4 else "MiniLM"
    )

    ax.set_title(
        f"{backbone} - {method}\n"
        f"Silhouette: {score:.4f}",
        fontsize=11
    )

    ax.set_xticks([])
    ax.set_yticks([])

    results.append({
        "experiment": experiment,
        "samples": len(labels),
        "silhouette_cosine": float(score)
    })

    print("Silhouette score:", score)

    del model
    gc.collect()
    torch.cuda.empty_cache()

handles, legend_labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    ncol=2,
    fontsize=12
)

fig.suptitle(
    "Embedding Space Comparison by Training Method",
    fontsize=17
)

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

plt.savefig(
    OUT / "embedding_tsne_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUT / "embedding_tsne_comparison.pdf",
    bbox_inches="tight"
)

plt.close()

pd.DataFrame(results).to_csv(
    ANALYSIS / "embedding_separation.csv",
    index=False
)

print("\n===== EMBEDDING SEPARATION =====")
print(pd.DataFrame(results).to_string(index=False))

print("\nEMBEDDING VISUALIZATION COMPLETE")
