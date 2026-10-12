"""The classifier zoo. Every entry is a scikit-learn model with .fit(X, y) and .predict(X).

Biology analogies (used in the slides):
    majority             "always say the most common species"  - the baseline every model must beat
    nearest_centroid     compare with the *average* specimen of each species (a type description)
    knn                  compare with the k most similar specimens in the collection, take a vote
    decision_tree        a dichotomous identification key, learned from the data
    random_forest        300 different keys, majority vote
    logistic_regression  a weighted checklist of characters -> a score per species
    mlp                  a small neural network: checklists of checklists
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DESCRIPTIONS = {
    "majority": "always predicts the most frequent label",
    "nearest_centroid": "closest average specimen",
    "knn": "vote of the 5 most similar training crops",
    "decision_tree": "learned identification key",
    "random_forest": "300 randomised keys, majority vote",
    "logistic_regression": "weighted checklist per label",
    "mlp": "small neural network (1 hidden layer)",
}


def make(name, seed=0, **kw):
    """A fresh, untrained classifier. Keyword arguments override the defaults, e.g. make('knn', k=1)."""
    scaled = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731 - put features on one scale
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "nearest_centroid":
        return scaled(NearestCentroid())
    if name == "knn":
        return scaled(KNeighborsClassifier(n_neighbors=kw.get("k", 5), n_jobs=-1))
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=kw.get("max_depth"), min_samples_leaf=kw.get("min_leaf", 1),
                                      random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=kw.get("n_trees", 300), max_depth=kw.get("max_depth"),
                                      n_jobs=-1, random_state=seed)
    if name == "logistic_regression":
        return scaled(LogisticRegression(C=kw.get("C", 0.1), max_iter=kw.get("max_iter", 2000)))
    if name == "mlp":
        return scaled(MLPClassifier(hidden_layer_sizes=(kw.get("hidden", 256),), alpha=kw.get("alpha", 1e-3),
                                    early_stopping=True, max_iter=300, random_state=seed))
    raise ValueError(f"unknown classifier {name!r}; choose from {list(DESCRIPTIONS)}")


NAMES = list(DESCRIPTIONS)
