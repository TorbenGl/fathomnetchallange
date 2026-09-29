# Known issues and bugfixes to watch (all found on the real data)

## Already handled in the code
| # | Issue | Effect | Handling |
|---|---|---|---|
| 1 | JSON has **23,699** annotations, not 23,700; one class has 299 | "300 per class" is false | Reported, never assumed |
| 2 | 15 boxes slightly leave the image (e.g. `y=-1`, `x+w > width`) | `PIL.crop` would pad with black | Clipped to image bounds; `bbox_clipped=1` in `annotations.csv` |
| 3 | Annotation **21685** lies fully outside its image (`y=370` on a 370 px image) | No crop possible | Listed in `failures.csv`; script exits with code 1 (expected) → 23,698 crops, Elpidia and Sebastes diploproa have 299 |
| 4 | WoRMS returns an **empty `rank`** for some leaf taxa (Terebellidae, Strongylocentrotus fragilis) | `label_rank` blank, lineage incomplete | Rank inferred from the name; `label_rank_inferred=1` |
| 5 | WoRMS ranks other than the 7 standard ones (subfamily, gigaclass, infraorder, subphylum) | Don't count as species…phylum | Counted as `other/unknown` (4 labels) |
| 6 | Labels sit at mixed ranks (33 species, 21 genus, 8 family, 6 order, 6 class, 1 phylum, 4 other) | Lower ranks empty in `taxonomy.csv`; "samples per order/family" includes `(label above this rank)` (2,700 / 4,800 ROIs) | Explicit bucket in the summary |
| 7 | Two ROIs from one image | Filename collisions | `<image_id>_<annotation_id>.jpg` |
| 8 | Network hiccups | Lost crops | 4 retries with backoff, resumable, failures logged |
| 9 | `fathomnet` needs the `[models]` extra only for `fathomnet.models` | ImportError if you import that | We only use `fathomnet.api` |

## Not fixed – be aware (data quality, decide per project)
- **Duplicated boxes:** 30 boxes appear twice on the same image (31 extra rows); **3 of them have conflicting labels**. Deduplicate or drop before training / evaluating.
- **Tiny objects:** 4,589 ROIs have a side < 32 px, 681 < 16 px; upscaling to 224×224 is very blurry.
- **Aspect ratio:** resizing to 224×224 distorts extreme boxes (100 ROIs with aspect > 4:1 or < 1:4). Padding to square first is an alternative.
- **Context is lost:** crops contain no background, so this is a classification-only dataset (matches the competition's ROI task).
- **Source images can disappear:** they are hosted at `database.fathomnet.org`; the SHA256 of the JSON is stored, but not of the images.
- **Alpha / palette PNGs:** `convert("RGB")` drops alpha without compositing. No issue seen in the full run (23,698 non-empty crops verified).
- **Bicubic vs bilinear:** torchvision `Resize` defaults to bilinear, we use PIL bicubic (recorded in `dataset_info.json`). Pick one for train **and** test.
- **JPEG re-compression** (q=95) adds a small artefact on top of the original PNGs.
- **Test set** is intentionally not prepared yet.

## Environment
- Needs outbound HTTPS to `github.com`, `database.fathomnet.org` (images + WoRMS API).
