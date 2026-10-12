"""How good is a classifier - and how *biologically* wrong are its mistakes?"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .data import RANKS

# colours (validated categorical palette, light mode) - used by every plot of the course
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def scores(y_true, y_pred):
    """accuracy, balanced accuracy (mean of per-label recalls) and macro-F1."""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def run(clf, X_train, y_train, X_test, y_test):
    """Fit, predict, score. Returns (dict of scores incl. train accuracy and seconds, test predictions)."""
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_s = time.time() - t0
    pred = clf.predict(X_test)
    res = scores(y_test, pred)
    res["train_accuracy"] = accuracy_score(y_train, clf.predict(X_train))
    res["fit_seconds"] = fit_s
    return res, pred


def per_label(y_true, y_pred):
    """Recall (fraction of each true label that was found) and support, worst first."""
    d = pd.DataFrame({"true": y_true, "ok": np.asarray(y_true) == np.asarray(y_pred)})
    return d.groupby("true")["ok"].agg(recall="mean", n="size").sort_values("recall")


# ----------------------------------------------------------------------------- taxonomy
def shared_rank(lineage_a, lineage_b):
    """Deepest rank at which two lineages (dicts rank -> name) still agree ('none' if not even kingdom)."""
    deepest = "none"
    for r in RANKS:
        a, b = lineage_a.get(r, ""), lineage_b.get(r, "")
        if a and a == b:
            deepest = r
        else:
            break
    return deepest


def taxonomic_report(true_labels, pred_labels, taxonomy):
    """For 79-label predictions: accuracy if we only ask for the phylum, class, ... and how severe
    the mistakes are. `taxonomy` = data.load_taxonomy() (indexed by label)."""
    tax = taxonomy[RANKS].to_dict("index")
    t, p = list(true_labels), list(pred_labels)
    rank_acc = {}
    for r in RANKS[1:]:
        ok = [tax[a][r] == tax[b][r] for a, b in zip(t, p) if tax[a][r]]
        rank_acc[r] = float(np.mean(ok)) if ok else np.nan
    wrong = [(a, b) for a, b in zip(t, p) if a != b]
    sev = pd.Series([shared_rank(tax[a], tax[b]) for a, b in wrong], dtype="object")
    order = ["none"] + RANKS
    severity = sev.value_counts().reindex(order[::-1], fill_value=0)
    return {"accuracy_at_rank": pd.Series(rank_acc), "mistakes_agree_up_to": severity,
            "n_mistakes": len(wrong)}


# ----------------------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, labels=None, normalize=True, ax=None, annotate=None):
    """Confusion matrix. Rows = truth, columns = prediction. normalize=True -> each row sums to 1."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if ax is None:
        n = len(labels)
        _, ax = plt.subplots(figsize=(min(3 + 0.35 * n, 16), min(2.5 + 0.35 * n, 15)))
    cmap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=10 if len(labels) < 30 else 4)
    ax.set_yticks(range(len(labels)), labels, fontsize=10 if len(labels) < 30 else 4)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    if annotate if annotate is not None else len(labels) <= 12:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]), ha="center", va="center",
                            fontsize=9, color="white" if cm[i, j] > 0.5 else INK)
    return ax


def plot_comparison(results, metric="accuracy", ax=None, title=None):
    """Horizontal bar chart from a list/DataFrame of results with columns 'name' and `metric`."""
    import matplotlib.pyplot as plt

    d = pd.DataFrame(results).sort_values(metric)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.4 * len(d) + 1))
    ax.barh(d["name"], d[metric], color=BLUE, height=0.6)
    for i, v in enumerate(d[metric]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.08), ax.set_xlabel(metric)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title, loc="left")
    return ax
