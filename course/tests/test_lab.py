"""Offline self-test of classifier_lab on a tiny synthetic dataset (no internet, no torch needed).

    python tests/test_lab.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier_lab import data, evaluate, features, models  # noqa: E402

TAX = [  # label, rank, kingdom, phylum, class, order, family, genus, species
    ("Sebastes diploproa", "species", "Animalia", "Chordata", "Teleostei", "Perciformes", "Sebastidae", "Sebastes", "Sebastes diploproa"),
    ("Sebastes", "genus", "Animalia", "Chordata", "Teleostei", "Perciformes", "Sebastidae", "Sebastes", ""),
    ("Asteroidea", "class", "Animalia", "Echinodermata", "Asteroidea", "", "", "", ""),
    ("Ophiuroidea", "class", "Animalia", "Echinodermata", "Ophiuroidea", "", "", "", ""),
    ("Actiniaria", "order", "Animalia", "Cnidaria", "Hexacorallia", "Actiniaria", "", "", ""),
]
COLOURS = {"Sebastes diploproa": (200, 60, 50), "Sebastes": (180, 80, 60), "Asteroidea": (230, 140, 40),
           "Ophiuroidea": (120, 120, 120), "Actiniaria": (240, 240, 230)}


def make_dataset(root):
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, aid = [], 0
    for li, (lab, *_r) in enumerate(TAX):
        for i in range(40):
            aid += 1
            img_id = 1000 * li + i // 2            # two crops per image
            dive = f"Ventana/images/{li}{(i // 2) % 4}"  # 4 dives per label, both crops of an image in one dive
            arr = np.clip(np.array(COLOURS[lab]) + rng.normal(0, 25, (24, 24, 3)), 0, 255).astype(np.uint8)
            p = f"images/{img_id}_{aid}.jpg"
            Image.fromarray(arr).resize((224, 224)).save(root / p)
            rows.append({"annotation_id": aid, "image_id": img_id, "category_id": li + 1, "concept_name": lab,
                         "crop_path": p, "bbox_x": i, "bbox_y": 0, "bbox_width": 30, "bbox_height": 40,
                         "source_url": f"https://database.fathomnet.org/static/m3/framegrabs/{dive}/{img_id}.png",
                         "bbox_clipped": 0, "image_width": 1920, "image_height": 1080})
    # one exact duplicate box and one conflicting duplicate -> removed by clean=True
    rows.append({**rows[0], "annotation_id": 9001})
    rows.append({**rows[1], "annotation_id": 9002, "concept_name": "Asteroidea"})
    pd.DataFrame(rows).to_csv(root / "annotations.csv", index=False)
    pd.DataFrame(TAX, columns=["concept_name", "label_rank", "kingdom", "phylum", "class", "order", "family",
                               "genus", "species"]).to_csv(root / "taxonomy.csv", index=False)


def main():
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "fathomnet2025_compact"
    make_dataset(root)
    os.environ["FATHOMNET_DATA"] = str(root)
    features.CACHE = tmp / "cache"

    df = data.load_dataset()
    assert len(df) == 199, len(df)                    # 200 + 2 duplicates - 1 exact dup - 2 conflicting
    assert df["dive"].iloc[0] == "database.fathomnet.org/Ventana/images/00", df["dive"].iloc[0]
    assert len(data.set_target(df, "order")) == 199 - 80   # the two class-level labels (2 x 40 crops) have no order
    df = data.set_target(df, "label")

    for how in ["random", "image", "dive"]:
        tr, te = data.split(df, how, seed=0)
        assert len(tr) + len(te) == len(df) and not set(tr.annotation_id) & set(te.annotation_id)
        ov = data.overlap_report(tr, te)
        if how == "dive":
            assert ov["share_dive_with_train"] == 0 and ov["share_image_with_train"] == 0, ov
        if how == "image":
            assert ov["share_image_with_train"] == 0, ov

    X = features.colour_stats(df)
    assert X.shape == (len(df), len(features.COLOUR_STAT_NAMES)), X.shape
    assert np.array_equal(X, features.colour_stats(df))     # cache round trip
    P = features.pixels(df)
    assert P.shape == (len(df), 768)

    tr, te = data.split(df, "random", seed=0)
    m = df["annotation_id"].isin(tr["annotation_id"]).to_numpy()
    y = df["y"].to_numpy()
    for name in models.NAMES:
        res, pred = evaluate.run(models.make(name), X[m], y[m], X[~m], y[~m])
        assert 0 <= res["accuracy"] <= 1 and len(pred) == (~m).sum()
        if name != "majority":
            assert res["accuracy"] > 0.5, (name, res)   # colours are clearly separable except the two Sebastes

    tax = data.load_taxonomy()
    lin = tax.to_dict("index")
    assert evaluate.shared_rank(lin["Sebastes diploproa"], lin["Sebastes"]) == "genus"
    assert evaluate.shared_rank(lin["Asteroidea"], lin["Ophiuroidea"]) == "phylum"
    assert evaluate.shared_rank(lin["Asteroidea"], lin["Actiniaria"]) == "kingdom"
    rep = evaluate.taxonomic_report(["Sebastes", "Asteroidea", "Asteroidea"],
                                    ["Sebastes diploproa", "Ophiuroidea", "Asteroidea"], tax)
    assert rep["n_mistakes"] == 2
    assert rep["mistakes_agree_up_to"]["genus"] == 1 and rep["mistakes_agree_up_to"]["phylum"] == 1
    assert rep["accuracy_at_rank"]["phylum"] == 1.0 and abs(rep["accuracy_at_rank"]["class"] - 2 / 3) < 1e-9
    s = evaluate.scores(["a", "a", "a", "b"], ["a", "a", "a", "a"])
    assert s["accuracy"] == 0.75 and s["balanced_accuracy"] == 0.5
    print("all classifier_lab tests passed")


if __name__ == "__main__":
    main()
