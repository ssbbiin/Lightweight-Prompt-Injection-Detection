import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import binomtest

ROOT = Path("outputs/external/notinject")

df = pd.read_csv(ROOT / "notinject_results.csv")

# Trigger count analysis

trigger = df[
    (df["operating_point"] == "default_05")
    & df["subset"].isin([
        "trigger_1", "trigger_2", "trigger_3"
    ])
]

table = trigger.pivot(
    index="experiment",
    columns="subset",
    values="fpr"
) * 100

table.to_csv(ROOT / "trigger_comparison.csv")

print("\n===== TRIGGER COUNT FPR (%) =====")
print(table.round(2).to_string())

# Paired exact McNemar test

experiments = {
    "Snowflake": {
        "Random": "E4_snowflake_random_cl",
        "Hard": "E5_snowflake_hard_cl",
        "Hybrid": "E10_snowflake_hybrid_cl",
        "ModelAware": "E12_snowflake_model_aware_cl"
    },
    "MiniLM": {
        "Random": "E8_minilm_random_cl",
        "Hard": "E9_minilm_hard_cl",
        "Hybrid": "E11_minilm_hybrid_cl",
        "ModelAware": "E13_minilm_model_aware_cl"
    }
}

pairs = [
    ("Random", "Hard"),
    ("Random", "Hybrid"),
    ("Random", "ModelAware"),
    ("Hard", "Hybrid"),
    ("Hard", "ModelAware"),
    ("Hybrid", "ModelAware")
]

results = []

for backbone, methods in experiments.items():

    for baseline, improved_method in pairs:

        a = pd.read_csv(
            ROOT / f"{methods[baseline]}_predictions.csv"
        )

        b = pd.read_csv(
            ROOT / f"{methods[improved_method]}_predictions.csv"
        )

        assert a["text"].equals(b["text"])

        x = a["prediction_05"].to_numpy()
        y = b["prediction_05"].to_numpy()

        improved = int(((x == 1) & (y == 0)).sum())
        degraded = int(((x == 0) & (y == 1)).sum())

        n = improved + degraded

        p = binomtest(
            improved, n, p=0.5
        ).pvalue if n else 1.0

        results.append({
            "backbone": backbone,
            "baseline": baseline,
            "method": improved_method,
            "improved": improved,
            "degraded": degraded,
            "net_fp_change": degraded - improved,
            "p_value": p
        })

out = pd.DataFrame(results)

# Holm multiple comparison correction

p = out["p_value"].to_numpy()
order = np.argsort(p)

adjusted = np.empty(len(p))

adjusted[order] = np.minimum(
    1.0,
    np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(p)))
    )
)

out["holm_p_value"] = adjusted
out["significant"] = adjusted < 0.05

out.to_csv(
    ROOT / "mcnemar_results.csv",
    index=False
)

print("\n===== NOTINJECT MCNEMAR TEST =====")
print(out.to_string(index=False))
