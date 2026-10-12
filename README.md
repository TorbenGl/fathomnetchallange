# FathomNet 2025 – compact 224×224 ROI dataset

> **Teaching:** [`course/`](course/README.md) holds a complete 6-session classifier project for biology students
> built on this dataset: slides with the roadmap, five lab notebooks, a mini-challenge and a final exam (PDFs in
> [`course/pdf/`](course/pdf)).

Builds a small local copy of the FathomNet 2025 FGVC training set (crops only, 224×224 JPEG,
full-resolution images are never written to disk) from the official
[`fathomnet/fgvc-comp-2025`](https://github.com/fathomnet/fgvc-comp-2025) `dataset_train.json`.

```bash
# Step-by-step notebook (recommended for students):
pip install -r requirements.txt -r requirements-notebook.txt
jupyter lab fathomnet2025_walkthrough.ipynb   # ships with executed outputs

# or the plain script:
pip install -r requirements.txt
python tests/test_offline.py            # optional: offline self-test (needs numpy; clones nothing)
python prepare_fathomnet2025.py --limit-images 20   # smoke test with real downloads
python prepare_fathomnet2025.py                      # full run, resumable
python prepare_fathomnet2025.py --per-category 100   # balanced subset: first 100 boxes of every label
```

Needs outbound HTTPS to `github.com`, `database.fathomnet.org` (images + WoRMS taxonomy API).
Output: `fathomnet2025_compact/{images/,annotations.csv,categories.csv,taxonomy.csv,dataset_train.json,
dataset_info.json,summary.txt,summary.json,failures.csv}`.

Known properties of the official JSON (checked by the script): 23,699 annotations (not 23,700),
16 boxes slightly leave the image (clipped, flagged in `bbox_clipped`), 1 box lies fully outside
its image and cannot be cropped (reported in `failures.csv`).

## Verified run
A full run (real downloads, 8 workers, ~15 min) produced 23,698 of 23,699 crops (252 MB), all
79 categories present, 1 failure (annotation 21685). Its summaries are in `reference_output/`
so students can compare their own results. `taxonomy.csv` has a `label_rank_inferred` column:
WoRMS returns an empty rank for a few leaf taxa (e.g. Terebellidae), so the rank is inferred from the name.

See `KNOWN_ISSUES.md` for all pitfalls found in the data and the tooling.
