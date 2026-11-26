import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

SEED = 42
HARD_RATIO = 0.5
CANDIDATE_K = 100
RISK_TOP_K = 10
MARGIN = 0.2

for model in ["snowflake", "minilm"]:

    print(f"\n===== {model.upper()} =====")

    root = Path("data/triplets") / model
    emb = Path("data/embeddings") / model

    X = np.load(emb / "train_embeddings.npy")
    y = np.load(emb / "train_labels.npy")

    random = pd.read_csv(root / "random_triplets.csv")
    hard = pd.read_csv(root / "hard_triplets.csv")

    cols = ["anchor_idx", "positive_idx"]

    assert random[cols].equals(hard[cols])

    a = random["anchor_idx"].to_numpy()
    p = random["positive_idx"].to_numpy()

    assert np.array_equal(a, np.arange(len(y)))
    assert np.all(y[a] == y[p])

    rng = np.random.default_rng(SEED)

    # Hybrid: select 50% of anchors in each class.

    hard_mask = np.zeros(len(y), dtype=bool)

    for label in [0, 1]:

        pool = np.where(y[a] == label)[0]

        selected = rng.choice(
            pool,
            size=int(len(pool) * HARD_RATIO),
            replace=False
        )

        hard_mask[selected] = True

    hybrid = np.where(
        hard_mask,
        hard["negative_idx"].to_numpy(),
        random["negative_idx"].to_numpy()
    )

    # Estimate attack probability using
    # out-of-fold training predictions.

    oof = np.zeros(len(y))

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y), start=1
    ):

        clf = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            random_state=SEED
        )

        clf.fit(X[train_idx], y[train_idx])

        oof[valid_idx] = clf.predict_proba(
            X[valid_idx]
        )[:, 1]

        print(f"OOF fold {fold} complete")

    np.save(root / "oof_attack_scores.npy", oof)

    # Model-Aware HNM:
    # Keep hybrid samples, but replace hard benign
    # negatives with high false-positive-risk samples.

    aware = hybrid.copy()

    benign_pool = np.where(y == 0)[0]

    attack_anchors = np.where(
        (y[a] == 1) & hard_mask
    )[0]

    risk_rng = np.random.default_rng(SEED + 3)

    for start in range(0, len(attack_anchors), 256):

        batch = attack_anchors[start:start + 256]

        sim = X[a[batch]] @ X[benign_pool].T

        k = min(CANDIDATE_K, len(benign_pool))

        top = np.argpartition(
            -sim, kth=k - 1, axis=1
        )[:, :k]

        for row, idx in enumerate(batch):

            candidates = benign_pool[top[row]]

            ranked = candidates[
                np.argsort(-oof[candidates])
            ]

            aware[idx] = risk_rng.choice(
                ranked[:RISK_TOP_K]
            )

    summary = {}

    for method, negatives in [
        ("hybrid", hybrid),
        ("model_aware", aware)
    ]:

        assert np.all(y[a] != y[negatives])

        pos_sim = np.sum(X[a] * X[p], axis=1)

        neg_sim = np.sum(
            X[a] * X[negatives],
            axis=1
        )

        out = random.copy()

        out["negative_idx"] = negatives
        out["negative_label"] = y[negatives]
        out["positive_similarity"] = pos_sim
        out["negative_similarity"] = neg_sim

        out.to_csv(
            root / f"{method}_triplets.csv",
            index=False
        )

        stats = {
            "triplets": len(out),
            "hard_ratio": float(hard_mask.mean()),
            "mean_positive_similarity": float(
                pos_sim.mean()
            ),
            "mean_negative_similarity": float(
                neg_sim.mean()
            ),
            "unique_negative_samples": int(
                len(np.unique(negatives))
            ),
            "active_triplet_ratio": float(
                (
                    neg_sim - pos_sim + MARGIN > 0
                ).mean()
            ),
            "targeted_attack_anchors": int(
                len(attack_anchors)
            ),
            "selected_benign_mean_attack_score": float(
                oof[negatives[attack_anchors]].mean()
            )
        }

        summary[method] = stats

        print(f"\n{method.upper()}")
        print(json.dumps(stats, indent=4))

    with open(
        root / "improved_mining_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=4)

print("\nIMPROVED MINING COMPLETE")
