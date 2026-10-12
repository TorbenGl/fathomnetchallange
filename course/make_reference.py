#!/usr/bin/env python3
"""Reference run for tutors: computes every number and figure used in the slides and the exam.

    python make_reference.py            # full dataset; ~25 min on 4 CPU cores (CNN features are cached)

Writes results/reference_*.csv, results/reference_results.json and slides/figures/*.pdf
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier_lab import data, evaluate, features, models  # noqa: E402
from classifier_lab.evaluate import AQUA, BLUE, INK, MUTED, ORANGE  # noqa: E402

FIG = HERE / "slides" / "figures"
RES = HERE / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 12.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})
BLUES = LinearSegmentedColormap.from_list("b", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
OUT = {}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("  figure", name)


def idx_of(df, part):
    return df["annotation_id"].isin(part["annotation_id"]).to_numpy()


t_start = time.time()
df = data.set_target(data.load_dataset(), "label")
tax = data.load_taxonomy()
raw = data.load_dataset(clean=False)
OUT["dataset"] = {"crops_clean": len(df), "crops_raw": len(raw), "labels": int(df["label"].nunique()),
                  "images": int(df["image_id"].nunique()), "dives": int(df["dive"].nunique()),
                  "share_box_lt32": float((df["box_size"] < 32).mean())}
per_dive = df.groupby("dive").size().sort_values(ascending=False)
OUT["dataset"]["top10_dives_share"] = float(per_dive.head(10).sum() / len(df))
print(OUT["dataset"])

# ------------------------------------------------------------------ deck 0: animals + phyla
show = ["Ptychogastria polaris", "Pyrosoma atlanticum", "Sebastes", "Asteroidea", "Hippasteria",
        "Paragorgia arborea", "Scotoplanes globosa", "Chionoecetes tanneri", "Octopus rubescens",
        "Funiculina", "Metridium farcimen", "Strongylocentrotus fragilis", "Heterochone calyx",
        "Florometra serratissima", "Pandalus platyceros", "Merluccius productus"]
fig, axes = plt.subplots(2, 8, figsize=(14, 4.4))
for ax, lab in zip(axes.ravel(), show):
    r = df[df["label"] == lab].nlargest(30, "box_size").sample(1, random_state=3).iloc[0]
    ax.imshow(Image.open(r["path"]))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{lab}\n{tax.loc[lab, 'phylum']}", fontsize=10, loc="left", fontweight="normal", color=INK)
fig.tight_layout()
save(fig, "animals_grid")

ph = df.groupby("phylum").agg(crops=("y", "size"), labels=("label", "nunique")).sort_values("crops")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.barh(ph.index, ph["crops"], color=BLUE, height=0.62)
for i, (c, n) in enumerate(zip(ph["crops"], ph["labels"])):
    ax.text(c + 80, i, f"{c:,}  ({n} labels)", va="center", fontsize=10.5, color=INK)
ax.set_xlim(0, ph["crops"].max() * 1.45)
ax.set_xlabel("crops")
ax.set_title("Crops per phylum", loc="left")
save(fig, "phylum_counts")

# ------------------------------------------------------------------ deck 1: two labels, two features
acc = lambda p, t: float((np.asarray(p) == np.asarray(t)).mean())  # noqa: E731
names = features.COLOUR_STAT_NAMES


def pair(a, b):
    """Crops of two labels described by (redness, brightness); random 75/25 split."""
    two = df[df["label"].isin([a, b])].reset_index(drop=True)
    F = features.colour_stats(two)
    X = F[:, [names.index("redness"), names.index("brightness")]]
    tr = idx_of(two, data.split(two, "random", seed=0)[0])
    return {"A": a, "B": b, "X": X, "y": two["y"].to_numpy(), "tr": tr, "te": ~tr}


def scatter(ax, P, mask, title, legend=False):
    for lab, col, mk in [(P["A"], ORANGE, "o"), (P["B"], BLUE, "s")]:
        m = mask & (P["y"] == lab)
        ax.scatter(P["X"][m, 0], P["X"][m, 1], s=13, c=col, marker=mk, label=lab, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("redness  (mean red − blue)")
    ax.set_ylabel("brightness")
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(fontsize=10.5, frameon=False, loc="lower right")


def regions(ax, P, clf):
    X = P["X"]
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .02, X[:, 0].max() + .02, 300),
                         np.linspace(X[:, 1].min() - .02, X[:, 1].max() + .02, 300))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) == P["A"]
    ax.contourf(xx, yy, z.reshape(xx.shape), levels=[-.5, .5, 1.5], colors=["#e3eefc", "#fde3d8"])


def scores2(P, clf):
    return (acc(clf.predict(P["X"][P["tr"]]), P["y"][P["tr"]]), acc(clf.predict(P["X"][P["te"]]), P["y"][P["te"]]))


EASY = pair("Ptychogastria polaris", "Pyrosoma atlanticum")      # red jellyfish vs bluish tunicate colony
HARD = pair("Ophiuroidea", "Asteroidea")                          # brittle stars vs sea stars: colours overlap
A, B, X2, y2, tr2, te2 = EASY["A"], EASY["B"], EASY["X"], EASY["y"], EASY["tr"], EASY["te"]

fig, ax = plt.subplots(figsize=(5.2, 3.9))
scatter(ax, EASY, tr2, "Training crops", legend=True)
save(fig, "two_features")

thr = 0.05
OUT["two_features"] = {"n_train": int(tr2.sum()), "n_test": int(te2.sum()), "threshold": thr,
                       "thr_train_acc": acc(np.where(X2[tr2, 0] > thr, A, B), y2[tr2]),
                       "thr_test_acc": acc(np.where(X2[te2, 0] > thr, A, B), y2[te2])}
fig, ax = plt.subplots(figsize=(5.2, 3.9))
scatter(ax, EASY, tr2, f"A rule by hand: redness > {thr}", legend=True)
ax.axvline(thr, color=INK, lw=1.5, ls="--")
ax.text(thr + 0.01, X2[:, 1].max(), f"→ {A.split()[0]}", fontsize=10.5, va="top")
save(fig, "two_features_threshold")

for P, name, kw, fname, ttl in [
        (EASY, "nearest_centroid", {}, "boundary_centroid", "nearest centroid"),
        (EASY, "knn", {"k": 5}, "boundary_knn5", "5-nearest neighbours"),
        (EASY, "decision_tree", {"max_depth": 2}, "boundary_tree2", "decision tree, depth 2"),
        (HARD, "knn", {"k": 1}, "hard_knn1", "k = 1"),
        (HARD, "knn", {"k": 101}, "hard_knn101", "k = 101")]:
    clf = models.make(name, **kw).fit(P["X"][P["tr"]], P["y"][P["tr"]])
    a_tr, a_te = scores2(P, clf)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    regions(ax, P, clf)
    scatter(ax, P, P["tr"], f"{ttl}: train {a_tr:.0%} · test {a_te:.0%}", legend=P is HARD)
    if name == "nearest_centroid":
        for lab, col in [(A, ORANGE), (B, BLUE)]:
            ax.scatter(*X2[tr2 & (y2 == lab)].mean(0), s=160, marker="*", c=col, edgecolor=INK, linewidth=1, zorder=5)
    save(fig, fname)
    OUT.setdefault("boundaries", {})[fname] = {"train": a_tr, "test": a_te}

from sklearn.tree import export_text, plot_tree  # noqa: E402

tree2 = models.make("decision_tree", max_depth=2).fit(X2[tr2], y2[tr2])
OUT["tree2_text"] = export_text(tree2, feature_names=["redness", "brightness"])
fig, ax = plt.subplots(figsize=(11, 4.4))
plot_tree(tree2, feature_names=["redness", "brightness"], class_names=[c.split()[0] for c in tree2.classes_],
          impurity=False, ax=ax, fontsize=11, filled=False)
save(fig, "tree2")

Xh, yh, trh, teh = HARD["X"], HARD["y"], HARD["tr"], HARD["te"]
OUT["hard_pair"] = {"n_train": int(trh.sum()), "n_test": int(teh.sum())}
rows = []
for k in [1, 3, 5, 11, 25, 51, 101]:
    clf = models.make("knn", k=k).fit(Xh[trh], yh[trh])
    rows.append({"param": k, "train": acc(clf.predict(Xh[trh]), yh[trh]), "test": acc(clf.predict(Xh[teh]), yh[teh])})
knn_sweep = pd.DataFrame(rows)
rows = []
for d in [1, 2, 3, 5, 8, 12, None]:
    t = models.make("decision_tree", max_depth=d).fit(Xh[trh], yh[trh])
    rows.append({"param": "none" if d is None else d, "leaves": int(t.get_n_leaves()),
                 "train": acc(t.predict(Xh[trh]), yh[trh]), "test": acc(t.predict(Xh[teh]), yh[teh])})
tree_sweep = pd.DataFrame(rows)
OUT["knn_sweep"], OUT["tree_sweep"] = knn_sweep.to_dict("records"), tree_sweep.to_dict("records")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
for ax, d, xl in [(axes[0], knn_sweep, "k (neighbours)   ← more flexible"),
                  (axes[1], tree_sweep, "maximum tree depth   more flexible →")]:
    x = np.arange(len(d))
    ax.plot(x, d["train"], "-o", color=MUTED, lw=2, ms=6, label="training crops")
    ax.plot(x, d["test"], "-o", color=ORANGE, lw=2, ms=6, label="test crops")
    ax.set_xticks(x, d["param"].astype(str))
    ax.set_xlabel(xl)
    ax.set_ylim(0.5, 1.02)
axes[0].set_ylabel("accuracy")
axes[0].legend(frameon=False, fontsize=10.5)
axes[0].set_title("k-nearest neighbours", loc="left")
axes[1].set_title("decision tree", loc="left")
save(fig, "overfitting")

# ------------------------------------------------------------------ deck 2: features + zoo
print("features ...")
X = {"pixels": features.pixels(df), "colour_stats": features.colour_stats(df), "cnn": features.cnn_embedding(df)}
y = df["y"].to_numpy()

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

samp = df.sample(3000, random_state=0).index
phyla = ["Cnidaria", "Echinodermata", "Arthropoda"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, (name, F) in zip(axes, X.items()):
    Z = PCA(min(30, F.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(F[samp]))
    P = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    p_ = df.loc[samp, "phylum"].to_numpy()
    for p, col in zip(phyla, [BLUE, ORANGE, AQUA]):
        ax.scatter(*P[p_ == p].T, s=3, c=col, label=p, alpha=0.55, rasterized=True)
    ax.set_title(name, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
axes[0].legend(markerscale=4, fontsize=10.5, frameon=False)
save(fig, "tsne_maps")

tr_df, te_df = data.split(df, "random", seed=0)
tr = idx_of(df, tr_df)
rows = []
for fname, F in X.items():
    for cname in models.NAMES:
        res, _ = evaluate.run(models.make(cname), F[tr], y[tr], F[~tr], y[~tr])
        rows.append({"features": fname, "classifier": cname, **res})
        print(f"  {fname:13s} {cname:20s} test {res['accuracy']:.3f} train {res['train_accuracy']:.3f} "
              f"{res['fit_seconds']:.0f}s", flush=True)
zoo = pd.DataFrame(rows)
zoo.to_csv(RES / "reference_zoo.csv", index=False)
tab = zoo.pivot(index="classifier", columns="features", values="accuracy").loc[models.NAMES, list(X)]
fig, ax = plt.subplots(figsize=(5.4, 4))
ax.imshow(tab.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3), ["pixels\n(768)", "colour stats\n(30)", "CNN\n(512)"])
ax.set_yticks(range(len(tab)), [n.replace("_", " ") for n in tab.index])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for (i, j), v in np.ndenumerate(tab.to_numpy()):
    ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > .55 else INK, fontsize=10)
ax.set_title("Test accuracy, 79 labels, random split", loc="left")
save(fig, "zoo_heatmap")

# phylum-target with colour stats (deck 1 / lab 1 last table)
rows = []
for target in ["phylum", "label"]:
    yt = df[target].to_numpy()
    for cname in ["majority", "nearest_centroid", "knn", "decision_tree"]:
        res, _ = evaluate.run(models.make(cname, max_depth=12), X["colour_stats"][tr], yt[tr], X["colour_stats"][~tr], yt[~tr])
        rows.append({"target": target, "classifier": cname, "test": res["accuracy"], "train": res["train_accuracy"]})
OUT["lab1_colourstats"] = rows

# ------------------------------------------------------------------ deck 3: honest evaluation
Xc = X["cnn"]
clf = models.make("logistic_regression").fit(Xc[tr], y[tr])
pred = clf.predict(Xc[~tr])
truth = y[~tr]
OUT["random_logreg"] = evaluate.scores(truth, pred)
pl = evaluate.per_label(truth, pred).join(tax[["label_rank", "phylum"]])
OUT["recall_by_label_rank"] = pl.groupby("label_rank")["recall"].agg(["mean", "size"]).reset_index().to_dict("records")
OUT["hardest_labels_random"] = pl.head(8)[["recall"]].reset_index().to_dict("records")

rows = []
for how in ["random", "image", "dive"]:
    for seed in [0, 1, 2]:
        a_, b_ = data.split(df, how, seed=seed)
        m = idx_of(df, a_)
        for cname in ["knn", "logistic_regression"]:
            c = models.make(cname).fit(Xc[m], y[m])
            p = c.predict(Xc[~m])
            ov = data.overlap_report(df[m], df[~m])
            rows.append({"split": how, "seed": seed, "classifier": cname, **evaluate.scores(y[~m], p),
                         "train_accuracy": float((c.predict(Xc[m]) == y[m]).mean()),
                         "share_dive_seen": ov["share_dive_with_train"] / ov["test_crops"],
                         "share_image_seen": ov["share_image_with_train"] / ov["test_crops"]})
        print(f"  split {how} seed {seed} done", flush=True)
sp = pd.DataFrame(rows)
sp.to_csv(RES / "reference_splits.csv", index=False)
spm = sp.groupby(["split", "classifier"], sort=False).mean(numeric_only=True).drop(columns="seed")
OUT["splits"] = spm.reset_index().to_dict("records")
print(spm)

fig, ax = plt.subplots(figsize=(6.4, 3.0))
order = ["random", "image", "dive"]
lr = spm.xs("logistic_regression", level="classifier").loc[order]
kn = spm.xs("knn", level="classifier").loc[order]
yy = np.arange(3)
ax.barh(yy + 0.19, lr["accuracy"], height=0.36, color=BLUE, label="logistic regression")
ax.barh(yy - 0.19, kn["accuracy"], height=0.36, color=ORANGE, label="5-nearest neighbours")
for i in range(3):
    ax.text(lr["accuracy"].iloc[i] + 0.01, i + 0.19, f"{lr['accuracy'].iloc[i]:.0%}", va="center", fontsize=11)
    ax.text(kn["accuracy"].iloc[i] + 0.01, i - 0.19, f"{kn['accuracy'].iloc[i]:.0%}", va="center", fontsize=11)
ax.set_yticks(yy, ["random crops", "whole images", "whole dives\n(new expedition)"])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("test accuracy (79 labels, CNN features, mean of 3 splits)")
ax.legend(frameon=False, fontsize=10.5, loc="lower right")
ax.set_title("Same model — three ways to split train/test", loc="left")
save(fig, "split_comparison")

a_, b_ = data.split(df, "dive", seed=0)
m = idx_of(df, a_)
c = models.make("logistic_regression").fit(Xc[m], y[m])
pd_ = c.predict(Xc[~m])
td_ = y[~m]
to_ph = tax["phylum"].to_dict()
ph_order = ["Cnidaria", "Porifera", "Echinodermata", "Arthropoda", "Mollusca", "Annelida", "Chordata"]
fig, ax = plt.subplots(figsize=(5.2, 4.6))
evaluate.plot_confusion([to_ph[t] for t in td_], [to_ph[p] for p in pd_], labels=ph_order, ax=ax)
ax.set_title("Phylum confusion, dive split", loc="left")
save(fig, "confusion_phylum")
rep_d = evaluate.taxonomic_report(td_, pd_, tax)
rep_r = evaluate.taxonomic_report(truth, pred, tax)
OUT["dive_logreg"] = evaluate.scores(td_, pd_)
OUT["accuracy_at_rank"] = {"random": rep_r["accuracy_at_rank"].to_dict(), "dive": rep_d["accuracy_at_rank"].to_dict()}
OUT["severity_dive"] = (rep_d["mistakes_agree_up_to"] / rep_d["n_mistakes"]).to_dict()
OUT["severity_random"] = (rep_r["mistakes_agree_up_to"] / rep_r["n_mistakes"]).to_dict()
OUT["n_mistakes_dive"] = rep_d["n_mistakes"]
from sklearn.metrics import confusion_matrix  # noqa: E402

cm = confusion_matrix([to_ph[t] for t in td_], [to_ph[p] for p in pd_], labels=ph_order)
OUT["phylum_confusion_dive"] = {"labels": ph_order, "matrix": cm.tolist()}

sev = rep_d["mistakes_agree_up_to"] / rep_d["n_mistakes"]
sev = sev[sev > 0]
sev.index = [{"none": "nothing", "kingdom": "kingdom only\n(different phylum)"}.get(i, i) for i in sev.index]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.barh(sev.index, sev.values, color=BLUE, height=0.6)
for i, v in enumerate(sev.values):
    ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=11)
ax.set_xlabel("share of mistakes")
ax.set_title("Mistakes agree up to … (dive split)", loc="left")
ax.set_xlim(0, max(sev.values) * 1.2)
save(fig, "mistake_severity")

rk = pd.DataFrame(OUT["accuracy_at_rank"]).loc[["species", "genus", "family", "order", "class", "phylum"]]
fig, ax = plt.subplots(figsize=(5.6, 3.1))
x = np.arange(len(rk))
ax.plot(x, rk["random"], "-o", color=MUTED, lw=2, label="random split")
ax.plot(x, rk["dive"], "-o", color=ORANGE, lw=2, label="dive split")
ax.set_xticks(x, rk.index)
ax.set_ylim(0.4, 1)
ax.set_ylabel("accuracy at this rank")
ax.legend(frameon=False, fontsize=10.5)
ax.set_title("Asking for a coarser rank is easier", loc="left")
save(fig, "accuracy_by_rank")

OUT["runtime_minutes"] = (time.time() - t_start) / 60
(RES / "reference_results.json").write_text(json.dumps(OUT, indent=1, default=float))
print(f"done in {OUT['runtime_minutes']:.1f} min")
