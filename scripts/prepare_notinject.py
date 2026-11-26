import json
import unicodedata
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

OUT = Path("data/external/notinject")

dataset_id = "leolee99/NotInject"

revision = HfApi().dataset_info(
    dataset_id,
    token=False
).sha

dataset = load_dataset(
    dataset_id,
    revision=revision,
    token=False
)

frames = []

for split, data in dataset.items():

    df = data.to_pandas()

    df["source_split"] = split
    df["trigger_count"] = df["word_list"].apply(len)

    frames.append(df)

df = pd.concat(
    frames,
    ignore_index=True
)

df = df.rename(columns={"prompt": "text"})
df["label"] = 0

assert len(df) == 339
assert df["trigger_count"].value_counts().to_dict() == {
    1: 113, 2: 113, 3: 113
}

def normalize(text):

    return " ".join(
        unicodedata.normalize(
            "NFKC", str(text)
        ).casefold().split()
    )

known = set()

for split in ["train", "validation", "test"]:

    original = pd.read_csv(
        f"data/processed/{split}.csv",
        keep_default_na=False
    )

    known.update(
        original["text"].map(normalize)
    )

df["overlap_original"] = (
    df["text"].map(normalize).isin(known)
)

df.to_csv(
    OUT / "notinject.csv",
    index=False
)

metadata = {
    "dataset_id": dataset_id,
    "revision": revision,
    "samples": len(df),
    "benign": len(df),
    "overlap_original": int(
        df["overlap_original"].sum()
    ),
    "internal_duplicates": int(
        df["text"].map(normalize).duplicated().sum()
    ),
    "trigger_counts": df[
        "trigger_count"
    ].value_counts().to_dict()
}

with open(OUT / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print(json.dumps(metadata, indent=4))
print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nNOTINJECT PREPARATION COMPLETE")
