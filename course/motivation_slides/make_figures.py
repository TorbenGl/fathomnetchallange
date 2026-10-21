"""Figures for the two motivation slides (run from course/: python motivation_slides/make_figures.py).

Slide 1: one held-out crop of Mediaster aequalis, the real ResNet-18 feature maps after each block, and the
         real top-3 probabilities of a logistic regression trained on ResNet-18 features (dive split, so the
         crop's dive was never seen in training).
Slide 2: one crop per example label + the taxonomic distance of each answer to Mediaster aequalis.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from classifier_lab import data, features, models  # noqa: E402

IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)
TRUTH = "Mediaster aequalis"
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

df = data.set_target(data.load_dataset(), "label")
X = features.get("cnn", df)
df["row"] = np.arange(len(df))
train, test = data.split(df, "dive", seed=0)
clf = models.make("logistic_regression").fit(X[train.row], train.y)
classes = clf.classes_

# --- slide 1: pick a clear, correctly classified held-out crop of the truth label -------------------------
cand = test[test.label == TRUTH].copy()
P = clf.predict_proba(X[cand.row])
cand["p_true"] = P[:, list(classes).index(TRUTH)]
cand["correct"] = classes[P.argmax(1)] == TRUTH
cand = cand[cand.correct & (cand.box_size >= 150)].sort_values("p_true", ascending=False)
pick = cand.iloc[min(3, len(cand) - 1)]          # a confident, but not the single most extreme, example
p = clf.predict_proba(X[[pick.row]])[0]
top = np.argsort(p)[::-1][:3]
print("picked", pick.annotation_id, pick.path, "box", pick.box_size, "dive", pick.dive)
probs = [(classes[i], float(p[i])) for i in top]
print(probs)
Image.open(os.path.join(os.path.dirname(HERE), "..", pick.path)).convert("RGB").save(
    os.path.join(IMG, "cnn_input.jpg"), quality=92)

# real feature maps after each ResNet-18 block
import torch  # noqa: E402
import torchvision  # noqa: E402
net = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
x = features.imagenet_transform()(Image.open(os.path.join(IMG, "cnn_input.jpg"))).unsqueeze(0)
maps = {}
with torch.no_grad():
    h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        h = getattr(net, name)(h)
        maps[name] = h[0].numpy()
for name, m in maps.items():
    # show the 3 channels that respond most to the animal (crop centre) relative to the seafloor (border)
    n = m.shape[1]
    ctr = np.zeros((n, n), bool)
    ctr[n // 4:n - n // 4, n // 4:n - n // 4] = True
    order = np.argsort(m[:, ctr].mean(1) - m[:, ~ctr].mean(1))[::-1]
    for k, c in enumerate(order[:3]):
        a = m[c]
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        rgb = (plt.get_cmap("magma")(a)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).resize((224, 224), Image.NEAREST).save(os.path.join(IMG, f"fmap_{name}_{k}.png"))
    print(name, m.shape)

# --- slide 2: example answers and their distance to the truth --------------------------------------------
tax = data.load_taxonomy()


def path_of(label):
    r = tax.loc[label]
    return [(rk, r[rk]) for rk in RANKS if isinstance(r[rk], str) and r[rk]]


def distance(a, b):
    pa, pb = path_of(a), path_of(b)
    common = 0
    for x_, y_ in zip(pa, pb):
        if x_ != y_:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


examples = [TRUTH, "Hippasteria", "Asteroidea", "Rathbunaster californicus", "Liponema brevicorne"]
out = {}
for i, lab in enumerate(examples):
    r = df[df.label == lab].nlargest(5, "box_size").iloc[0 if lab != TRUTH else 1]
    Image.open(os.path.join(os.path.dirname(HERE), "..", r.path)).convert("RGB").save(
        os.path.join(IMG, f"ex{i}.jpg"), quality=92)
    out[lab] = distance(TRUTH, lab)
    print(lab, tax.loc[lab, "label_rank"], "distance", out[lab], r.path)
assert distance("Mediaster aequalis", "Mediaster aequalis") == 0
json.dump({"probs": probs, "distances": out, "input": os.path.basename(pick.path)}, open(os.path.join(IMG, "figures.json"), "w"),
          indent=1)
