# FathomNet 2025 – compact 224×224 ROI dataset

Builds a small local copy of the FathomNet 2025 FGVC training set (crops only, 224×224 JPEG,
full-resolution images are never written to disk) from the official
[`fathomnet/fgvc-comp-2025`](https://github.com/fathomnet/fgvc-comp-2025) `dataset_train.json`.

```bash
pip install -r requirements.txt
python tests/test_offline.py            # optional: offline self-test (needs numpy; clones nothing)
python prepare_fathomnet2025.py --limit-images 20   # smoke test with real downloads
python prepare_fathomnet2025.py                      # full run, resumable
```

Needs outbound HTTPS to `github.com`, `database.fathomnet.org` (images + WoRMS taxonomy API).
Output: `fathomnet2025_compact/{images/,annotations.csv,categories.csv,taxonomy.csv,dataset_train.json,
dataset_info.json,summary.txt,summary.json,failures.csv}`.

Known properties of the official JSON (checked by the script): 23,699 annotations (not 23,700),
16 boxes slightly leave the image (clipped, flagged in `bbox_clipped`), 1 box lies fully outside
its image and cannot be cropped (reported in `failures.csv`).
