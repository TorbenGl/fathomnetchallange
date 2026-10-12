#!/usr/bin/env python3
"""Run classifier experiments from the command line (the notebook does the same step by step).

Examples
    python run_lab.py --target phylum --features colour_stats --classifiers knn decision_tree
    python run_lab.py --features pixels colour_stats cnn --classifiers all --split random dive
    python run_lab.py --features cnn --classifiers logistic_regression --per-class 300 --out results/my_runs.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier_lab import data, evaluate, features, models  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="dataset folder (default: ../fathomnet2025_compact)")
    ap.add_argument("--target", default="label", choices=data.TARGETS)
    ap.add_argument("--features", nargs="+", default=["cnn"],
                    help="pixels colour_stats cnn cnn_resnet50 cnn_efficientnet_b0")
    ap.add_argument("--classifiers", nargs="+", default=["all"], help=f"all or some of {models.NAMES}")
    ap.add_argument("--split", nargs="+", default=["random", "dive"], choices=["random", "image", "dive"])
    ap.add_argument("--per-class", type=int, default=None, help="at most N crops per label (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="append results to this CSV")
    a = ap.parse_args(argv)

    df = data.subsample(data.set_target(data.load_dataset(a.data), a.target), a.per_class, a.seed)
    print(f"{len(df)} crops, {df['y'].nunique()} classes (target = {a.target})")
    clfs = models.NAMES if a.classifiers == ["all"] else a.classifiers
    feats = {f: features.get(f, df) for f in a.features}
    rows = []
    for how in a.split:
        train, test = data.split(df, how, seed=a.seed)
        ov = data.overlap_report(train, test)
        print(f"\nsplit={how}: train {len(train)}, test {len(test)}; test crops whose dive is also in train: "
              f"{ov['share_dive_with_train']}")
        tr_idx = df.index[df["annotation_id"].isin(train["annotation_id"])]
        te_idx = df.index[df["annotation_id"].isin(test["annotation_id"])]
        for fname, X in feats.items():
            for cname in clfs:
                res, _ = evaluate.run(models.make(cname, seed=a.seed), X[tr_idx], df.loc[tr_idx, "y"],
                                      X[te_idx], df.loc[te_idx, "y"])
                res.update(split=how, features=fname, classifier=cname, target=a.target,
                           n_train=len(tr_idx), n_test=len(te_idx), per_class=a.per_class, seed=a.seed,
                           when=time.strftime("%Y-%m-%d %H:%M"))
                rows.append(res)
                print(f"  {fname:14s} {cname:20s} test acc {res['accuracy']:.3f}  "
                      f"train acc {res['train_accuracy']:.3f}  macro-F1 {res['macro_f1']:.3f}  "
                      f"({res['fit_seconds']:.1f}s)", flush=True)
    out = pd.DataFrame(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(a.out, mode="a", header=not Path(a.out).exists(), index=False)
        print(f"\nappended {len(out)} rows to {a.out}")
    return out


if __name__ == "__main__":
    main()
