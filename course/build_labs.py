"""Generates the five lab notebooks of the course (run: python build_labs.py).

Edit the text here, not in the .ipynb files, then rebuild and re-execute:
    python build_labs.py [lab2 ...] && for f in lab*.ipynb; do jupyter nbconvert --to notebook --execute --inplace $f; done
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

BOOT = '''
# Run this cell first. On Google Colab it fetches the course; on your own computer it only loads the data.
import os, subprocess, sys
from pathlib import Path
if "google.colab" in sys.modules:
    if not Path("/content/fathomnetchallange").exists():
        if Path("/content/fathomnetchallange.zip").exists():      # zip from your tutor (drag it into Files)
            subprocess.run(["unzip", "-q", "/content/fathomnetchallange.zip", "-d", "/content"], check=True)
        else:
            subprocess.run(["git", "clone", "-q", "https://github.com/TorbenGl/fathomnetchallange",
                            "/content/fathomnetchallange"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "fathomnet"], check=True)
    os.chdir("/content/fathomnetchallange/course")
sys.path.insert(0, str(Path.cwd()))

from setup_env import setup
ROOT = setup()          # first time: downloads the crops (~15 min). setup(per_category=100) is a quick subset.

import warnings; warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from classifier_lab import data, features, models, evaluate
plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False})
'''


class NB:
    def __init__(self):
        self.cells, self.q = [], 0

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def question(self, s):
        self.q += 1
        self.md(f"> ### ✍️ Q{self.q}\n> {s.strip()}")
        self.md("*Your answer:* ")

    def save(self, name):
        nb = nbf.v4.new_notebook()
        nb.cells = self.cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["colab"] = {"provenance": []}
        nbf.write(nb, HERE / name)
        print("wrote", name, len(self.cells), "cells")


# =============================================================================== Lab 0
def lab0():
    n = NB()
    n.md("""
# Lab 0 – Meet the data
*Session 1 · about 2 hours · work in pairs*

Before we let a computer identify animals, we look at what it will learn from. By the end of this lab you can
say how the dataset was made, what a *label* is, how the labels sit in the tree of life, and why the **dive** an
image comes from will matter later.

**How these notebooks work:** run cells from top to bottom with *Shift + Enter*. Cells marked ✍️ are questions for
you – write your answer in the cell below them (double-click to edit). You never have to write code from scratch;
at most you change a word or a number.
""")
    n.code(BOOT)
    n.md("""
## 1. One table, one row per animal
The official FathomNet 2025 training set has 8,981 deep-sea images. Experts drew a **box** around each animal
and named it. We cut each box out and shrank it to a 224 × 224 picture – a **crop**. Every crop is one row:
""")
    n.code("""
df = data.load_dataset()
print(len(df), "crops,", df["label"].nunique(), "labels,", df["image_id"].nunique(), "images,", df["dive"].nunique(), "dives")
df[["annotation_id", "image_id", "label", "phylum", "class", "order", "family", "genus", "species", "dive", "box_size"]].head(8)
""")
    n.md("""
| column | meaning |
|---|---|
| `label` | the name the expert gave – one of **79 labels** |
| `phylum` … `species` | the label's lineage in the tree of life (from WoRMS, the World Register of Marine Species) |
| `image_id` | which video frame the crop was cut from (one frame can contain several animals) |
| `dive` | which ROV dive (or camera deployment) the frame comes from |
| `box_size` | the shorter side of the box in the original image, in pixels |
""")
    n.md("## 2. Look at the animals\nEach row of the figure is one label, each picture one crop. Run the cell several times with a different `seed`.")
    n.code("""
df["y"] = df["label"]
fig = data.show_examples(df, labels=["Asteroidea", "Sebastes", "Ptychogastria polaris", "Pyrosoma atlanticum",
                                     "Paragorgia arborea", "Scotoplanes globosa"], n=7, seed=0)
""")
    n.md("Pick labels you find interesting: the full list is printed below. Copy names into the list above and re-run.")
    n.code("""print(", ".join(sorted(df["label"].unique())))""")
    n.question("Pick one label and describe its crops in two sentences, like a field-guide entry. "
               "Which *characters* (colour, shape, texture, background) would **you** use to recognise it? "
               "Which crops would even an expert find hard, and why?")
    n.md("## 3. How many examples per label, per phylum?")
    n.code("""
counts = df.groupby(["phylum", "label"]).size().rename("crops").reset_index()
print(counts["crops"].describe()[["min", "max"]])
ax = df["phylum"].value_counts().sort_values().plot.barh(color=evaluate.BLUE, figsize=(6, 3))
ax.set_xlabel("crops"); ax.set_title("Crops per phylum", loc="left");
""")
    n.question("Every label has about 300 crops, but the phyla are very unequal. Why? (Hint: count the labels per phylum "
               "with `counts.groupby('phylum').size()`.) What could this mean for a classifier that predicts *phyla*?")
    n.md("""
## 4. Labels sit at different ranks
Not every label is a species. An expert names only what can be seen on the video: a blurry sea star stays
*Asteroidea* (a class). `label_rank` says at which rank each label sits.
""")
    n.code("""
tax = data.load_taxonomy()
display(tax["label_rank"].value_counts().to_frame("labels"))
tax.loc[["Sebastes diploproa", "Sebastes", "Asteroidea", "Porifera", "Tunicata"], ["label_rank", "phylum", "class", "order", "family", "genus"]]
""")
    n.question("Why is it a problem for a classifier that both *Sebastes* (a genus) and *Sebastes diploproa* (a species in that genus) "
               "are labels? What does a crop labelled *Sebastes* tell you about the species?")
    n.md("""
## 5. Dives: why the source of an image matters
A dive is a few hours of video from one place. Frames of one dive share the site, the light, the camera –
and often **the same individual animal**, filmed again a few seconds later.
""")
    n.code("""
per_dive = df.groupby("dive").size().sort_values(ascending=False)
print(f"{len(per_dive)} dives; the 10 largest contribute {per_dive.head(10).sum() / len(df):.0%} of all crops")
ax = per_dive.reset_index(drop=True).plot(logy=True, color=evaluate.BLUE, figsize=(6, 2.6))
ax.set_xlabel("dives, largest first"); ax.set_ylabel("crops"); ax.set_title("Crops per dive", loc="left");
""")
    n.md("Crops of one label from **one** dive – how different are they really?")
    n.code("""
lab = "Scotoplanes globosa"
d = df[df["label"] == lab]
top_dive = d["dive"].value_counts().index[0]
one = d[d["dive"] == top_dive].copy(); one["y"] = "same dive: " + top_dive[-25:]
other = d[d["dive"] != top_dive].copy(); other["y"] = "other dives"
data.show_examples(pd.concat([one, other]), labels=[one["y"].iloc[0], "other dives"], n=8, seed=1);
""")
    n.question("Imagine we train a classifier on half of the frames of a dive and test it on the other half. Why could the "
               "result look better than it really is? Keep your answer – we test it in Lab 3.")
    n.md("## 6. Tiny animals\nMany boxes are small in the original image; the crop was blown up to 224 × 224 and is blurry.")
    n.code("""
print((df["box_size"] < 32).mean().round(3), "of all crops come from a box smaller than 32 px")
small = df[df["box_size"] < 24].copy(); small["y"] = "box < 24 px"
big = df[df["box_size"] > 300].copy(); big["y"] = "box > 300 px"
data.show_examples(pd.concat([small, big]), labels=["box < 24 px", "box > 300 px"], n=8, seed=2);
""")
    n.md("""
## 7. Your group
From now on, each pair follows **one group** through all labs (e.g. sea stars = Asteroidea, soft corals = Octocorallia,
fish = Teleostei / Actinopterygii, sea cucumbers = Holothuroidea, sponges = Porifera, crabs and shrimps = Malacostraca).
""")
    n.code("""
MY_RANK, MY_GROUP = "class", "Asteroidea"        # <- change to your group, e.g. ("phylum", "Porifera")
mine = df[df[MY_RANK] == MY_GROUP]
print(len(mine), "crops,", mine["label"].nunique(), "labels:", ", ".join(sorted(mine["label"].unique())))
mine = mine.assign(y=mine["label"])
data.show_examples(mine, labels=sorted(mine["label"].unique())[:6], n=7);
""")
    n.question("Hand-in for Lab 0: name your group, list its labels, and save the figure above "
               "(right-click → save image). Which two labels of your group do you expect a computer to confuse most? Why?")
    n.save("lab0_data.ipynb")


# =============================================================================== Lab 1
def lab1():
    n = NB()
    n.md("""
# Lab 1 – Your first classifier
*Session 2 · about 2 hours*

A **classifier** takes a few numbers that describe an animal (its *features*, like the characters in an
identification key) and returns a label. It *learns* its rule from labelled examples – the **training set** –
and we check it on examples it has never seen – the **test set**.

We start as small as possible: **two labels, two features**, so that everything fits in one picture.
""")
    n.code(BOOT)
    n.md("""
## 1. Two animals, two characters
*Ptychogastria polaris* is a small, reddish deep-sea jellyfish (Cnidaria). *Pyrosoma atlanticum* is a colonial
tunicate (Chordata) that looks bluish on the video. We describe every crop by two numbers:
* **redness** = mean of (red − blue) over all pixels,
* **brightness** = mean grey value.
""")
    n.code("""
A, B = "Ptychogastria polaris", "Pyrosoma atlanticum"
df = data.set_target(data.load_dataset())
two = df[df["label"].isin([A, B])].reset_index(drop=True)
data.show_examples(two, labels=[A, B], n=8);

F = features.colour_stats(two)                         # 30 numbers per crop; we use two of them
names = features.COLOUR_STAT_NAMES
X2 = F[:, [names.index("redness"), names.index("brightness")]]
y2 = two["y"].to_numpy()
train, test = data.split(two, "random", seed=0)        # 75 % to learn from, 25 % to test on
tr = two["annotation_id"].isin(train["annotation_id"]).to_numpy(); te = ~tr
print(f"train: {tr.sum()} crops, test: {te.sum()} crops")
""")
    n.code("""
def scatter(ax, X, y, mask, title=""):
    for lab, col, mk in [(A, evaluate.ORANGE, "o"), (B, evaluate.BLUE, "s")]:
        m = mask & (y == lab)
        ax.scatter(X[m, 0], X[m, 1], s=14, c=col, marker=mk, label=lab, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("redness"); ax.set_ylabel("brightness"); ax.set_title(title, loc="left"); ax.legend(fontsize=8)

fig, ax = plt.subplots(figsize=(5.5, 4))
scatter(ax, X2, y2, tr, "Training crops")
""")
    n.md("""
## 2. Be the classifier
Find a rule by eye: a **threshold on redness** that separates the two animals. Change `THRESHOLD` and re-run until
the training accuracy is as high as you can get. Then – only once – look at the test accuracy.
""")
    n.code("""
THRESHOLD = 0.0          # <- your choice

def my_rule(X):
    return np.where(X[:, 0] > THRESHOLD, A, B)      # redder than the threshold -> the jellyfish

acc = lambda pred, truth: (pred == truth).mean()
print(f"training accuracy: {acc(my_rule(X2[tr]), y2[tr]):.1%}")
""")
    n.code("""
print(f"test accuracy:     {acc(my_rule(X2[te]), y2[te]):.1%}")
""")
    n.question("Which threshold did you choose and why? Is the test accuracy higher or lower than the training accuracy? "
               "Would a sloping line (using brightness as well) be better? Sketch it on paper.")
    n.md("""
## 3. Let the computer find the rule
Three classic ways to learn a rule from examples:

| classifier | idea | biology analogy |
|---|---|---|
| **nearest centroid** | compute the *average* crop of each label; assign a new crop to the closest average | compare with the type description |
| **k-nearest neighbours (kNN)** | find the *k* most similar training crops; majority vote | compare with the reference collection |
| **decision tree** | a sequence of yes/no questions on single features | a dichotomous identification key |

The coloured background shows which label the classifier would give to every possible point – its **decision regions**.
The line between regions is the **decision boundary**.
""")
    n.code("""
def regions(ax, clf, X):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .02, X[:, 0].max() + .02, 300),
                         np.linspace(X[:, 1].min() - .02, X[:, 1].max() + .02, 300))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) == A
    ax.contourf(xx, yy, z.reshape(xx.shape), levels=[-.5, .5, 1.5], colors=["#cde2fb", "#fbd9cb"], alpha=.8)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name, kw in zip(axes, ["nearest_centroid", "knn", "decision_tree"], [{}, {"k": 5}, {"max_depth": 2}]):
    clf = models.make(name, **kw).fit(X2[tr], y2[tr])
    regions(ax, clf, X2)
    scatter(ax, X2, y2, te, f"{name}: test accuracy {acc(clf.predict(X2[te]), y2[te]):.1%}")
""")
    n.md("""
## 4. Overfitting: learning the collection by heart
The jellyfish and the tunicate were easy. Now a pair that colour alone separates badly: **sea stars (Asteroidea)
and brittle stars (Ophiuroidea)**. kNN with **k = 1** copies the label of the single most similar training crop.
On the training set it is always right (every crop is its own nearest neighbour). Is it also the best on new crops?
""")
    n.code("""
A, B = "Ophiuroidea", "Asteroidea"
hard = df[df["label"].isin([A, B])].reset_index(drop=True)
Fh = features.colour_stats(hard)
Xh = Fh[:, [names.index("redness"), names.index("brightness")]]
yh = hard["y"].to_numpy()
trh = hard["annotation_id"].isin(data.split(hard, "random", seed=0)[0]["annotation_id"]).to_numpy(); teh = ~trh

rows = []
for k in [1, 3, 5, 11, 25, 51, 101]:
    clf = models.make("knn", k=k).fit(Xh[trh], yh[trh])
    rows.append({"k": k, "train accuracy": acc(clf.predict(Xh[trh]), yh[trh]), "test accuracy": acc(clf.predict(Xh[teh]), yh[teh])})
display(pd.DataFrame(rows).set_index("k").style.format("{:.1%}"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, k in zip(axes, [1, 101]):
    clf = models.make("knn", k=k).fit(Xh[trh], yh[trh])
    regions(ax, clf, Xh); scatter(ax, Xh, yh, trh, f"k = {k} (training crops shown)")
""")
    n.question("Describe the decision regions for k = 1 and k = 101. Which one has *learned the collection by heart*? "
               "This is called **overfitting**. Why is the training accuracy useless for judging a classifier?")
    n.md("""
## 5. A decision tree is an identification key
A decision tree asks one yes/no question per step. Here is the tree learned from the two features:
""")
    n.code("""
from sklearn.tree import plot_tree
A, B = "Ptychogastria polaris", "Pyrosoma atlanticum"     # back to the easy pair
tree = models.make("decision_tree", max_depth=2).fit(X2[tr], y2[tr])
fig, ax = plt.subplots(figsize=(11, 4.5))
plot_tree(tree, feature_names=["redness", "brightness"], class_names=list(tree.classes_), filled=False,
          impurity=False, proportion=False, ax=ax, fontsize=9);
""")
    n.md("Now trees of growing depth on the **hard pair** (sea stars vs brittle stars):")
    n.code("""
rows = []
for depth in [1, 2, 3, 5, 8, 12, None]:
    t = models.make("decision_tree", max_depth=depth).fit(Xh[trh], yh[trh])     # the hard pair
    rows.append({"max_depth": str(depth), "leaves": t.get_n_leaves(),
                 "train accuracy": acc(t.predict(Xh[trh]), yh[trh]), "test accuracy": acc(t.predict(Xh[teh]), yh[teh])})
pd.DataFrame(rows).set_index("max_depth").style.format({"train accuracy": "{:.1%}", "test accuracy": "{:.1%}"})
""")
    n.question("Read the depth-2 tree above like a key: write it as a dichotomous key (1a/1b, 2a/2b …). "
               "What happens to train and test accuracy when the tree for the hard pair gets deeper? Would a taxonomist write a key with "
               "hundreds of couplets that each fit one single specimen?")
    n.md("""
## 6. From 2 to 79 labels, from 2 to 30 features
Real life is harder. We now use **all 30 colour/texture features** and ask for the **phylum** (7 answers) and then
for the **label** (79 answers). The **majority** classifier always says the most common answer – the baseline
that every real classifier must beat.
""")
    n.code("""
full = data.set_target(data.load_dataset(), "phylum")
Xf = features.colour_stats(full)
train, test = data.split(full, "random", seed=0)
tr = full["annotation_id"].isin(train["annotation_id"]).to_numpy()
rows = []
for target in ["phylum", "label"]:
    y = full[target].to_numpy()
    for name in ["majority", "nearest_centroid", "knn", "decision_tree"]:
        res, _ = evaluate.run(models.make(name, max_depth=12), Xf[tr], y[tr], Xf[~tr], y[~tr])
        rows.append({"target": target, "name": name, "test accuracy": res["accuracy"], "train accuracy": res["train_accuracy"]})
table = pd.DataFrame(rows).pivot(index="name", columns="target", values=["test accuracy", "train accuracy"])
table.style.format("{:.1%}")
""")
    n.question("How much better than the majority baseline is the best classifier, for phyla and for labels? "
               "Is 7-phylum classification really 'easier' – or is the baseline just higher? "
               "Why do you think colour features alone are not enough for 79 labels?")
    n.md("""
## Summary
* A classifier = features + a rule learned from a training set.
* Judge it **only on a test set** it has not seen. Training accuracy measures memory, not skill.
* Flexible rules (k = 1, deep trees) can **overfit**.
* Always compare with a **baseline**.

Next lab: better features – including what a neural network "sees".
""")
    n.save("lab1_first_classifier.ipynb")


# =============================================================================== Lab 2
def lab2():
    n = NB()
    n.md("""
# Lab 2 – How a computer sees an animal, and the classifier zoo
*Session 3 · about 2 hours (computing the CNN features takes ~15 min on a 4-core laptop, a few minutes on a Colab GPU)*

The same classifier can be excellent or useless depending on the numbers we give it. Today we compare three
ways to describe a crop, and seven classifiers on each.

| feature set | numbers per crop | idea |
|---|---|---|
| `pixels` | 768 | the crop shrunk to 16 × 16 pixels × 3 colours |
| `colour_stats` | 30 | hand-made characters: colour, brightness, texture, hue histogram (Lab 1) |
| `cnn` | 512 | what a **convolutional neural network** pre-trained on 1.2 million everyday photos (ImageNet) "sees" |
""")
    n.code(BOOT)
    n.md("## 1. A picture is a table of numbers")
    n.code("""
df = data.set_target(data.load_dataset(), "label")
row = df[df["label"] == "Hippasteria"].iloc[3]
im = Image.open(row["path"])
small = im.resize((16, 16), Image.BILINEAR)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im); axes[0].set_title("crop, 224 × 224", loc="left")
axes[1].imshow(small); axes[1].set_title("16 × 16: what 'pixels' sees", loc="left")
red = np.asarray(small)[:, :, 0]
axes[2].imshow(red, cmap="Reds")
for (i, j), v in np.ndenumerate(red[::2, ::2]):
    axes[2].text(2 * j, 2 * i, v, ha="center", va="center", fontsize=6)
axes[2].set_title("red channel values (every 2nd pixel)", loc="left")
for a in axes: a.axis("off")
""")
    n.md("""
## 2. What a CNN does (in one paragraph)
A CNN slides thousands of small **filters** over the image. The first layers respond to edges and colour blobs,
middle layers to textures and repeated patterns (spines, tentacles, polyps), the last layers to whole object parts.
Trained on ImageNet (dogs, cars, mushrooms …), these filters are general enough to describe deep-sea animals too.
We cut off the network's last layer and keep the **512 numbers** before it: the *embedding*. We do **not** train
the CNN here – we only use it as a very good character extractor (*transfer learning*).
""")
    n.code("""
X = {"pixels": features.pixels(df),
     "colour_stats": features.colour_stats(df),
     "cnn": features.cnn_embedding(df)}          # cached in features_cache/ after the first run
{k: v.shape for k, v in X.items()}
""")
    n.md("""
## 3. A map of the animals
Each feature set places every crop as a point in a space with 30–768 dimensions. **t-SNE** draws a 2-D map of that
space in which crops that are close in the feature space stay close on the map (distances between far-apart groups
mean little). Points are coloured by phylum. This takes about a minute per map.
""")
    n.code("""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
sample = df.sample(3000, random_state=0).index
phyla = df["phylum"].value_counts().index[:4]
cols = dict(zip(phyla, [evaluate.BLUE, evaluate.ORANGE, evaluate.AQUA, "#6250d6"]))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, F) in zip(axes, X.items()):
    Z = PCA(min(30, F.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(F[sample]))
    P = TSNE(2, init="pca", random_state=0).fit_transform(Z)
    ph = df.loc[sample, "phylum"].to_numpy()
    for p in phyla:
        ax.scatter(*P[ph == p].T, s=4, c=cols[p], label=p, alpha=.6)
    ax.set_title(name, loc="left"); ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(markerscale=3, fontsize=8);
""")
    n.question("In which map do the phyla form the clearest groups? What does that suggest about which features will "
               "work best? (A 2-D map loses information – a mixed-up map does not prove the features are useless.)")
    n.md("""
## 4. Nearest neighbours: 'which crops look most similar?'
For one query crop, show the 5 most similar crops in each feature space. Correct neighbours (same label) get a ✓.
""")
    n.code("""
from sklearn.neighbors import NearestNeighbors
query = df[df["label"] == "Sebastolobus"].index[5]
fig, axes = plt.subplots(3, 6, figsize=(12, 6.5))
for r, (name, F) in enumerate(X.items()):
    Fs = StandardScaler().fit_transform(F)
    nn = NearestNeighbors(n_neighbors=6).fit(Fs)
    idx = nn.kneighbors(Fs[[query]], return_distance=False)[0]
    for c, i in enumerate(idx):
        ax = axes[r, c]; ax.imshow(Image.open(df.loc[i, "path"])); ax.set_xticks([]); ax.set_yticks([])
        ok = "query" if c == 0 else ("✓ " if df.loc[i, "label"] == df.loc[query, "label"] else "✗ ") + df.loc[i, "label"][:18]
        ax.set_title(ok, fontsize=8, loc="left")
    axes[r, 0].set_ylabel(name)
fig.tight_layout()
""")
    n.md("""
## 5. The classifier zoo
Seven classifiers × three feature sets, 79 labels, a random 75 / 25 split. This takes a few minutes (the random
forest and the neural network are the slowest). Each classifier is one line in `classifier_lab/models.py` – open it!

| name | idea |
|---|---|
| `majority` | always the most common label (baseline) |
| `nearest_centroid` | closest average crop |
| `knn` | vote of the 5 most similar training crops |
| `decision_tree` | learned identification key |
| `random_forest` | 300 randomised keys, majority vote |
| `logistic_regression` | a weighted checklist: score per label = sum of (weight × feature) |
| `mlp` | a small neural network: checklists of checklists |
""")
    n.code("""
train, test = data.split(df, "random", seed=0)
tr = df["annotation_id"].isin(train["annotation_id"]).to_numpy()
y = df["y"].to_numpy()
rows = []
for fname, F in X.items():
    for cname in models.NAMES:
        res, _ = evaluate.run(models.make(cname), F[tr], y[tr], F[~tr], y[~tr])
        rows.append({"features": fname, "classifier": cname, **res})
        print(f"{fname:13s} {cname:20s} test {res['accuracy']:.1%}   train {res['train_accuracy']:.1%}   {res['fit_seconds']:.0f}s")
zoo = pd.DataFrame(rows)
zoo.to_csv("results/lab2_zoo.csv", index=False) if Path("results").exists() else None
""")
    n.code("""
tab = zoo.pivot(index="classifier", columns="features", values="accuracy").loc[models.NAMES, ["pixels", "colour_stats", "cnn"]]
fig, ax = plt.subplots(figsize=(6, 4))
from matplotlib.colors import LinearSegmentedColormap
ax.imshow(tab.to_numpy(), cmap=LinearSegmentedColormap.from_list("b", ["#ffffff", "#86b6ef", "#104281"]), vmin=0, vmax=1,
          aspect="auto")
ax.set_xticks(range(3), tab.columns); ax.set_yticks(range(len(tab)), tab.index)
for (i, j), v in np.ndenumerate(tab.to_numpy()):
    ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v > .6 else "black")
ax.set_title("Test accuracy, 79 labels", loc="left");
""")
    n.question("Which matters more for accuracy: the choice of **features** or the choice of **classifier**? "
               "Support your answer with two numbers from the table.")
    n.question("Compare train and test accuracy for `knn`, `decision_tree` and `random_forest` on CNN features "
               "(`zoo[zoo.features == 'cnn']`). Which ones overfit? Is overfitting always bad for the test accuracy?")
    n.question("Logistic regression is a 'weighted checklist'. Why might such a simple rule work so well on CNN features but "
               "badly on raw pixels?")
    n.md("""
## 6. Your group
How well does the best combination do on the labels of *your* group (from Lab 0)?
""")
    n.code("""
MY_RANK, MY_GROUP = "class", "Asteroidea"        # <- your group
clf = models.make("logistic_regression").fit(X["cnn"][tr], y[tr])
pred = pd.Series(clf.predict(X["cnn"][~tr]), index=df.index[~tr])
mine = df.index[~tr][df.loc[~tr, MY_RANK] == MY_GROUP]
evaluate.per_label(df.loc[mine, "y"], pred[mine])
""")
    n.question("Report the recall of each label of your group. Which label is hardest? Look at a few of its crops "
               "(Lab 0) – can you see why?")
    n.save("lab2_features_and_zoo.ipynb")


# =============================================================================== Lab 3
def lab3():
    n = NB()
    n.md("""
# Lab 3 – Testing honestly
*Session 4 · about 2 hours*

A single accuracy number hides a lot. Today we ask: **which** labels are confused, how **biologically** wrong are
the mistakes, and – most important – would the classifier work on a **new expedition**?

We use the best combination from Lab 2: CNN features + logistic regression.
""")
    n.code(BOOT)
    n.code("""
df = data.set_target(data.load_dataset(), "label")
X = features.cnn_embedding(df)
tax = data.load_taxonomy()
y = df["y"].to_numpy()

def fit_predict(how, seed=0, clf_name="logistic_regression"):
    train, test = data.split(df, how, seed=seed)
    tr = df["annotation_id"].isin(train["annotation_id"]).to_numpy()
    clf = models.make(clf_name).fit(X[tr], y[tr])
    return tr, clf.predict(X[~tr]), clf

tr, pred, clf = fit_predict("random")
truth = y[~tr]
print(evaluate.scores(truth, pred))
""")
    n.md("""
## 1. Three scores
* **accuracy** – fraction of test crops labelled correctly.
* **balanced accuracy** – the accuracy of each label separately (its *recall*), then averaged: every label counts equally.
* **macro-F1** – like balanced accuracy, but also punishes labels that are *predicted too often* (low *precision*).

With ~300 crops per label all three are similar here. They differ a lot when some labels are rare – for example in a
real survey, where 90 % of the animals may be one species.
""")
    n.question("A survey video contains 900 brittle stars and 100 sea pens. A classifier that always answers 'brittle star' "
               "has what accuracy? What balanced accuracy? Which number would you report, and why?")
    n.md("## 2. Which labels are easy, which are hard?")
    n.code("""
pl = evaluate.per_label(truth, pred)
fig, ax = plt.subplots(figsize=(7, 5))
show = pd.concat([pl.head(10), pl.tail(10)])
ax.barh(show.index, show["recall"], color=[evaluate.ORANGE] * 10 + [evaluate.BLUE] * 10, height=.6)
ax.set_xlim(0, 1); ax.set_xlabel("recall (fraction found)"); ax.set_title("10 hardest and 10 easiest labels", loc="left")
ax.invert_yaxis();
""")
    n.code("""
pl.join(tax[["label_rank", "phylum"]]).groupby("label_rank")["recall"].agg(["mean", "size"]).sort_values("mean")
""")
    n.question("Are species-level labels easier or harder than labels at higher ranks (family, order, class)? "
               "Explain with biology: what does a crop labelled with a *class* name look like compared with one labelled "
               "with a *species* name?")
    n.md("""
## 3. The confusion matrix
Rows = true label, columns = predicted label. The diagonal holds the correct answers; everything else is a mistake.
With 79 labels the matrix is big, so we first look at it **by phylum** (we translate both truth and prediction to
their phylum), then at the labels of one phylum.
""")
    n.code("""
to_phylum = tax["phylum"].to_dict()
fig, ax = plt.subplots(figsize=(6, 5))
evaluate.plot_confusion([to_phylum[t] for t in truth], [to_phylum[p] for p in pred], ax=ax)
ax.set_title("Confusion by phylum (rows sum to 1)", loc="left");
""")
    n.code("""
PHYLUM = "Echinodermata"          # <- try Cnidaria, Porifera, Arthropoda, Chordata ...
labs = [l for l in tax.sort_values(["class", "order", "family", "genus", "label_rank"]).index if to_phylum[l] == PHYLUM]
m = np.isin(truth, labs)
pred_m = [p if p in labs else "(other phylum)" for p in pred[m]]
evaluate.plot_confusion(truth[m], pred_m, labels=labs + ["(other phylum)"], annotate=False);
""")
    n.question("Find the biggest off-diagonal cell in the matrix for your phylum. Which two labels are confused? "
               "Show crops of both (Lab 0, `data.show_examples`). Would a human expert confuse them too?")
    n.md("""
## 4. How *biologically* wrong are the mistakes?
For every mistake we ask: up to which rank do the true and the predicted label still agree?
Confusing two *Sebastes* species agrees up to the genus (a near miss). Confusing a sea star with a sea anemone
agrees only in the kingdom Animalia (a far miss). This is the idea behind the FathomNet 2025 score.
""")
    n.code("""
rep = evaluate.taxonomic_report(truth, pred, tax)
display(rep["accuracy_at_rank"].to_frame("accuracy if we only ask for this rank").style.format("{:.1%}"))
sev = rep["mistakes_agree_up_to"]
ax = (sev / sev.sum()).plot.barh(color=evaluate.BLUE, figsize=(6, 3))
ax.set_xlabel("share of mistakes"); ax.set_title(f"{rep['n_mistakes']} mistakes: deepest shared rank", loc="left");
""")
    n.question("Are most mistakes near misses (same genus/family) or far misses (different phylum)? "
               "For a biodiversity survey, which kind of mistake is worse, and why?")
    n.md("""
## 5. The big one: random split vs. new dives
So far crops were assigned to train and test **at random**. But frames of the same dive – often of the same
individual animal – end up on both sides. That is like an exam that reuses the practice questions.

Now we split three ways:
* `random` – crop by crop,
* `image` – all crops of one video frame stay together,
* `dive` – all crops of one dive stay together: **the classifier is tested on expeditions it has never seen**.
""")
    n.code("""
rows = []
for how in ["random", "image", "dive"]:
    for seed in [0, 1, 2]:
        tr_, pred_, _ = fit_predict(how, seed)
        ov = data.overlap_report(df[tr_], df[~tr_])
        rows.append({"split": how, "seed": seed, **evaluate.scores(y[~tr_], pred_),
                     "test crops from a dive seen in training": ov["share_dive_with_train"] / ov["test_crops"]})
splits = pd.DataFrame(rows)
summary = splits.groupby("split", sort=False)[["accuracy", "macro_f1", "test crops from a dive seen in training"]].mean()
summary.style.format("{:.1%}")
""")
    n.code("""
ax = evaluate.plot_comparison(summary.reset_index().rename(columns={"split": "name"}), "accuracy",
                              title="Same features, same classifier - different test sets")
""")
    n.question("How large is the drop from `random` to `dive`? Explain *why* it happens, using what you saw in Lab 0 (§5). "
               "Which number would you put in a paper or a survey report – and why?")
    n.question("The prize-winning FathomNet models scored ~98 % on crops they had trained on, ~63–66 % on new images and "
               "~57–61 % on images from new dives. Compare with your numbers. What does this tell you about how far "
               "'99 % accuracy' claims in the news can be trusted?")
    n.md("""
## 6. Look at the mistakes
Confident and wrong is the most dangerous combination for a survey. Below: test crops from the **dive** split that
were labelled wrongly with high confidence.
""")
    n.code("""
tr_d, pred_d, clf_d = fit_predict("dive", 0)
proba = clf_d.predict_proba(X[~tr_d]).max(axis=1)
wrong = df[~tr_d].assign(pred=pred_d, conf=proba)
wrong = wrong[wrong["y"] != wrong["pred"]].sort_values("conf", ascending=False).head(12)
fig, axes = plt.subplots(2, 6, figsize=(13, 5.5))
for ax, (_, r) in zip(axes.ravel(), wrong.iterrows()):
    ax.imshow(Image.open(r["path"])); ax.axis("off")
    ax.set_title(f"true: {r['y'][:16]}\\npred: {r['pred'][:16]} ({r['conf']:.0%})", fontsize=7, loc="left")
fig.tight_layout()
""")
    n.question("Pick three of these mistakes. For each: is it a mistake of the classifier, a hard/blurry crop, or might "
               "the expert label itself be questionable?")
    n.md("""
**Hand-in for Lab 3:** the phylum confusion matrix, the split comparison chart, and half a page answering:
*Which mistakes does the classifier make, how biologically severe are they, and how well would it do on a new expedition?*
""")
    n.save("lab3_honest_evaluation.ipynb")


# =============================================================================== Lab 4
def lab4():
    n = NB()
    n.md("""
# Lab 4 – Mini-challenge: the best classifier for a *new* expedition
*Session 5 · about 2 hours + poster*

**The task:** build the classifier with the highest **macro-F1 on the dive-split test set** (79 labels).
You may change the features, the classifier and its settings – anything in this notebook.

**The rules (they are what makes the result honest):**
1. The test set is fixed: `data.split(df, "dive", seed=0)`. Do not change it.
2. **Never tune on the test set.** Try ideas on the *validation* set (a dive split *inside* the training set).
3. When you are done, evaluate **once** on the test set and submit that number – even if it is lower than you hoped.
""")
    n.code(BOOT)
    n.code("""
df = data.set_target(data.load_dataset(), "label")
train_full, test = data.split(df, "dive", seed=0)          # fixed: the 'new expedition'
train, val = data.split(train_full, "dive", seed=1)        # your playground
idx = lambda part: df.index[df["annotation_id"].isin(part["annotation_id"])]
i_train, i_val, i_full, i_test = idx(train), idx(val), idx(train_full), idx(test)
print(len(i_train), "train /", len(i_val), "validation /", len(i_test), "test crops")
""")
    n.md("""
## 1. Ideas to try
* other feature sets: `features.get("cnn_resnet50", df)` (2,048 numbers, slower) or `"cnn_efficientnet_b0"`;
  or **combine** feature sets with `np.hstack([X1, X2])`
* other classifier settings: `models.make("logistic_regression", C=1.0)`, `models.make("knn", k=15)`,
  `models.make("mlp", hidden=512)`, `models.make("random_forest", n_trees=500)`
* fewer labels with very few, blurry crops? (no – the test set keeps them; but you can *drop tiny boxes from training*:
  `df.loc[i_train, "box_size"] > 16`)
* **Level 5 (GPU recommended):** fine-tune the whole CNN, see section 3.
""")
    n.code("""
X = features.cnn_embedding(df)                              # <- change me
y = df["y"].to_numpy()

def try_on_validation(clf, X):
    res, _ = evaluate.run(clf, X[i_train], y[i_train], X[i_val], y[i_val])
    print(f"validation: accuracy {res['accuracy']:.1%}  macro-F1 {res['macro_f1']:.3f}  ({res['fit_seconds']:.0f}s)")
    return res

for C in [0.01, 0.1, 1.0]:                                  # <- an example of tuning one setting
    print("C =", C, end=": "); try_on_validation(models.make("logistic_regression", C=C), X)
""")
    n.md("Keep a log of what you tried – it goes on your poster.")
    n.code("""
log = []   # e.g. log.append({"idea": "logreg C=0.1 on resnet18", "val_macro_f1": 0.52})
""")
    n.md("""
## 2. Final evaluation – run once
Train your chosen classifier on **all** training data (train + validation) and evaluate on the test set.
""")
    n.code("""
TEAM = "team-name"                                          # <- your team name
FINAL = models.make("logistic_regression", C=0.1)           # <- your final choice
FEATURES = "cnn (resnet18)"                                 # <- describe your features

res, pred = evaluate.run(FINAL, X[i_full], y[i_full], X[i_test], y[i_test])
print(f"TEST: accuracy {res['accuracy']:.1%}   macro-F1 {res['macro_f1']:.3f}")
rep = evaluate.taxonomic_report(y[i_test], pred, data.load_taxonomy())
print("accuracy at phylum level:", f"{rep['accuracy_at_rank']['phylum']:.1%}")

Path("results").mkdir(exist_ok=True)
entry = pd.DataFrame([{"team": TEAM, "features": FEATURES, "classifier": str(FINAL)[:80],
                       "test_accuracy": res["accuracy"], "test_macro_f1": res["macro_f1"],
                       "phylum_accuracy": rep["accuracy_at_rank"]["phylum"]}])
f = Path("results/leaderboard.csv"); entry.to_csv(f, mode="a", header=not f.exists(), index=False)
pd.read_csv(f).sort_values("test_macro_f1", ascending=False)
""")
    n.md("""
## 3. Level 5 (optional): fine-tune a whole CNN
Up to now the CNN was frozen: we only trained a classifier on its 512 numbers. The competition teams instead
**adjust the CNN itself** to deep-sea animals. On a Colab T4 GPU three epochs should take a few minutes; on a laptop
CPU it takes hours – set `RUN_FINETUNE = True` only on a GPU (*Runtime → Change runtime type → T4 GPU*).
""")
    n.code("""
RUN_FINETUNE = False
if RUN_FINETUNE:
    from classifier_lab.finetune import finetune
    pred_ft = finetune(df.loc[i_full], df.loc[i_test], epochs=3)
    print("fine-tuned ResNet-18, TEST:", evaluate.scores(y[i_test], pred_ft))
""")
    n.md("""
## 4. Your poster (A3, or 4 slides)
1. **Question & data** – one sentence each; acknowledge FathomNet / MBARI.
2. **Method** – features + classifier in words a biologist understands (no code).
3. **Results** – your test macro-F1 and accuracy, the random-vs-dive comparison (Lab 3), one confusion matrix.
4. **Mistakes** – three example mistakes with crops and a biological explanation.
5. **Verdict** – would you use this classifier for a survey on a new expedition? What would have to be checked by hand?
6. **What you tried** – your log from section 1, including ideas that did not work.
""")
    n.save("lab4_challenge.ipynb")


if __name__ == "__main__":
    import sys
    labs = {"lab0": lab0, "lab1": lab1, "lab2": lab2, "lab3": lab3, "lab4": lab4}
    for name in sys.argv[1:] or labs:          # python build_labs.py lab2  -> only that notebook
        labs[name]()
