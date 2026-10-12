"""Load the compact FathomNet 2025 crops and split them into train/test.

The dataset folder is produced by ../prepare_fathomnet2025.py and contains
annotations.csv (one row per crop), taxonomy.csv (lineage of the 79 labels) and images/.
"""
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
TARGETS = ["label", "phylum", "class", "order"]


def find_data(root=None):
    """Return the dataset folder: `root`, $FATHOMNET_DATA, or the usual places next to this course."""
    here = Path(__file__).resolve().parent.parent
    candidates = [root, os.environ.get("FATHOMNET_DATA"), "fathomnet2025_compact",
                  here.parent / "fathomnet2025_compact", here / "fathomnet2025_compact"]
    for c in candidates:
        if c and (Path(c) / "annotations.csv").exists():
            return Path(c)
    raise FileNotFoundError("No dataset found. Run `python prepare_fathomnet2025.py` in the repository root "
                            "(or set FATHOMNET_DATA to the folder that contains annotations.csv).")


def ensure_dataset(per_category=None, out=None, workers=8):
    """Find the dataset, or build it with ../prepare_fathomnet2025.py if it is missing.

    per_category=None downloads all 23,698 crops (~15 min, 250 MB); per_category=100 gives a balanced
    subset of ~7,900 crops in ~5 min. Images are streamed and only the 224x224 crops are stored.
    """
    try:
        return find_data(out)
    except FileNotFoundError:
        pass
    import sys
    repo = Path(__file__).resolve().parent.parent.parent
    out = Path(out or repo / "fathomnet2025_compact")
    sys.path.insert(0, str(repo))
    import prepare_fathomnet2025 as prep
    args = ["--out", str(out), "--repo-dir", str(repo / "fgvc-comp-2025"), "--workers", str(workers)]
    if per_category:
        args += ["--per-category", str(per_category)]
    prep.main(args)  # returns 1 because of the one known broken box (annotation 21685) - that is expected
    return find_data(out)


def dive_of(url):
    """ROV dive / camera deployment = the folder of the image URL (e.g. 'Doc Ricketts/images/0621')."""
    p = urlparse(url)
    return p.netloc + unquote(os.path.dirname(p.path)).replace("/static/m3/framegrabs", "")


def load_dataset(root=None, clean=True):
    """One row per crop with columns label, path, image_id, dive and the lineage kingdom ... species.

    clean=True removes exact duplicate boxes (same image, same box) and drops boxes that appear
    twice with *different* labels - see KNOWN_ISSUES.md.
    """
    root = find_data(root)
    ann = pd.read_csv(root / "annotations.csv")
    tax = pd.read_csv(root / "taxonomy.csv", dtype=str).fillna("")
    df = ann.rename(columns={"concept_name": "label"})
    df = df.merge(tax.rename(columns={"concept_name": "label"}), on="label", how="left")
    df["path"] = [str(root / p) for p in df["crop_path"]]
    df["dive"] = df["source_url"].map(dive_of)
    df["box_size"] = df[["bbox_width", "bbox_height"]].min(axis=1)
    if clean:
        key = ["image_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        n_labels = df.groupby(key)["label"].transform("nunique")
        df = df[n_labels == 1].drop_duplicates(key)
    return df.sort_values("annotation_id").reset_index(drop=True)


def load_taxonomy(root=None):
    """taxonomy.csv indexed by label (the 79 concept names)."""
    tax = pd.read_csv(find_data(root) / "taxonomy.csv", dtype=str).fillna("")
    return tax.rename(columns={"concept_name": "label"}).set_index("label")


def set_target(df, target="label"):
    """Add column `y` = what the classifier must predict.

    'label' = the 79 FathomNet labels; 'phylum' / 'class' / 'order' = that rank of the label's lineage.
    Rows whose label sits above the requested rank (e.g. the label 'Porifera' when target='class')
    are dropped, because they have no answer at that rank.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    df = df.copy()
    df["y"] = df[target]
    return df[df["y"] != ""].reset_index(drop=True)


def subsample(df, per_class=None, seed=0, by="label"):
    """At most `per_class` rows per label (None = everything). Makes experiments fast."""
    if per_class is None:
        return df
    parts = [g.sample(min(len(g), per_class), random_state=seed) for _, g in df.groupby(by)]
    return pd.concat(parts).sort_values("annotation_id").reset_index(drop=True)


def split(df, how="random", test_size=0.25, seed=0):
    """Return (train, test).

    how='random' - every crop goes to train or test at random (stratified by y). Crops from the same
                   image or the same dive can end up on both sides -> optimistic.
    how='image'  - all crops of one image stay together.
    how='dive'   - all crops of one ROV dive stay together: the honest "new expedition" test.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    n_splits = max(2, round(1 / test_size))
    y = df["y"].to_numpy()
    if how == "random":
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, y)
    elif how in ("image", "dive"):
        groups = df["image_id" if how == "image" else "dive"].to_numpy()
        folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(df, y, groups)
    else:
        raise ValueError("how must be 'random', 'image' or 'dive'")
    tr, te = next(iter(folds))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def overlap_report(train, test):
    """How many test crops share an image / a dive with the training set?"""
    return {"test_crops": len(test),
            "share_image_with_train": int(test["image_id"].isin(set(train["image_id"])).sum()),
            "share_dive_with_train": int(test["dive"].isin(set(train["dive"])).sum()),
            "test_labels_missing_in_train": sorted(set(test["y"]) - set(train["y"]))}


def show_examples(df, labels=None, n=6, seed=0, size=1.6):
    """Grid of example crops: one row per label."""
    import matplotlib.pyplot as plt
    from PIL import Image

    labels = labels if labels is not None else sorted(df["y"].unique())[:6]
    fig, axes = plt.subplots(len(labels), n, figsize=(n * size, len(labels) * size), squeeze=False)
    for r, lab in enumerate(labels):
        rows = df[df["y"] == lab]
        rows = rows.sample(min(n, len(rows)), random_state=seed)
        for c, ax in enumerate(axes[r]):
            ax.axis("off")
            if c < len(rows):
                ax.imshow(Image.open(rows.iloc[c]["path"]))
        axes[r][0].set_title(lab, loc="left", fontsize=9)
    fig.tight_layout()
    return fig
