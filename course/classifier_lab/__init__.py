"""classifier_lab - small helpers for the FathomNet classifier course.

Read the modules in this order; each is short:
    data.py      load crops + taxonomy, choose the target, split train/test
    features.py  turn an image into numbers (pixels, colour statistics, CNN embedding)
    models.py    the classifier zoo (all scikit-learn, one line each)
    evaluate.py  accuracy, confusion matrix, "how biologically wrong" a mistake is
    finetune.py  optional: train the last layers of a CNN (needs a GPU to be fast)
"""
from . import data, evaluate, features, models  # noqa: F401
