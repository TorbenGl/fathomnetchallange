"""Generates fathomnet2025_walkthrough.ipynb (run: python tools_build_notebook.py)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# FathomNet 2025 – compact ROI dataset, step by step

We turn the official FathomNet 2025 FGVC training set (8,981 full-resolution images) into a small
dataset of **224×224 crops** of the annotated organisms, and analyse its taxonomy.

| Step | What we do |
|---|---|
| 1 | Get the official repo + `dataset_train.json` |
| 2 | Look at the COCO structure |
| 3 | Integrity checks (what can go wrong *before* downloading) |
| 4 | One image by hand: download → crop → resize |
| 5 | Batch crops with the real pipeline + verification |
| 6 | Taxonomy from WoRMS (and its bug) |
| 7 | Taxonomic analysis |
| 8 | Full run + known issues |

All heavy lifting lives in `prepare_fathomnet2025.py`; this notebook calls its functions so you can see each step.
Set `N_IMAGES` below to control how much is downloaded (the full set is `None`, ~15 min, ~250 MB).
""")
code("""
import io, json, sys, requests, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path.cwd()))
import prepare_fathomnet2025 as P

N_IMAGES = 40            # None = all 8,981 images
OUT = Path("fathomnet2025_compact_demo")
""")
md("## 1. Official repo and source of truth\nWe never query the general FathomNet database; `dataset_train.json` *defines* the dataset. We pin the commit and hash so results are reproducible.")
code("""
repo = P.get_official_repo("fgvc-comp-2025")
src = repo / "datasets" / "dataset_train.json"
print("commit:", P.git(repo, "rev-parse", "HEAD"))
print("sha256:", P.sha256_file(src))
data = json.loads(src.read_text())
""")
md("## 2. COCO structure\nThree tables: **images** (with `coco_url`), **annotations** (bbox = `[x, y, width, height]`), **categories** (79 concept names).")
code("""
print({k: len(v) if isinstance(v, list) else v for k, v in data.items()})
display(pd.DataFrame(data["images"]).head(3))
display(pd.DataFrame(data["annotations"]).head(3))
display(pd.DataFrame(data["categories"]).head(3))
""")
md("## 3. Integrity checks before downloading\nThe task description says *79 × 300 = 23,700* annotations. **Do not assume it – check.**")
code("""
report, images, cats = P.check_json(data)
pd.Series({k: v for k, v in report.items() if k != "categories_not_exactly_300"}).to_frame("value")
""")
code("""
print("Categories that are not exactly 300:", report["categories_not_exactly_300"])
ann = pd.DataFrame(data["annotations"])
ann[["x", "y", "w", "h"]] = pd.DataFrame(ann.bbox.tolist(), index=ann.index)
ann["W"] = ann.image_id.map(lambda i: images[i]["width"]); ann["H"] = ann.image_id.map(lambda i: images[i]["height"])
outside = ann[(ann.x < 0) | (ann.y < 0) | (ann.x + ann.w > ann.W) | (ann.y + ann.h > ann.H)]
print(f"{len(outside)} boxes leave the image (mostly by 1-10 px -> we clip them):")
display(outside[["id", "image_id", "bbox", "W", "H"]])
""")
code("""
# duplicated boxes on the same image (possible double labelling)
k = ["image_id", "x", "y", "w", "h"]
dups = ann[ann.duplicated(k, keep=False)]
g = dups.groupby(k).category_id.nunique()
print(f"{len(g)} duplicated boxes; {int((g > 1).sum())} of them carry CONFLICTING labels")
# box size distribution -> many objects are tiny, upscaling to 224 blurs them
print(ann[["w", "h"]].describe().round(0))
print("boxes with a side < 32 px:", int(((ann.w < 32) | (ann.h < 32)).sum()))
""")
md("## 4. One image by hand\nDownload into memory → crop `(x, y, x+w, y+h)` → RGB → resize to 224×224 (bicubic). The original is never written to disk.")
code("""
a = data["annotations"][1]
im = images[a["image_id"]]
img = P.fetch_image(im["coco_url"])
x, y, w, h = a["bbox"]
crop = img.crop((x, y, x + w, y + h)).convert("RGB")
small = crop.resize((224, 224), Image.BICUBIC)
print("source:", img.size, img.mode, "| bbox:", a["bbox"], "| crop:", crop.size, "-> ", small.size,
      "| category:", cats[a["category_id"]]["name"])
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(img); ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="red", lw=2)); ax[0].set_title("source + bbox")
ax[1].imshow(crop); ax[1].set_title(f"crop {crop.size}")
ax[2].imshow(small); ax[2].set_title("224x224")
for s in ax: s.axis("off")
plt.show()
""")
md("## 5. The real pipeline on a subset\n`build_crops` downloads each image **once**, cuts all its ROIs (`<image_id>_<annotation_id>.jpg`, so two ROIs of one image never collide) and discards the image. `verify_crops` re-opens every file.")
code("""
keep = {i["id"] for i in data["images"][:N_IMAGES]} if N_IMAGES else set(images)
anns = [a for a in data["annotations"] if a["image_id"] in keep]
rows, failures = P.build_crops(images, anns, cats, OUT, workers=8, resample=P.INTERPOLATIONS["bicubic"], force=False)
for r in rows: r["_abs"] = OUT / r["crop_path"]
bad = P.verify_crops(rows)
print(f"{len(rows)} crops, {len(failures)} failures, {len(bad)} failed verification")
df = pd.DataFrame(rows).drop(columns="_abs"); df.head()
""")
code("""
fig, axs = plt.subplots(2, 6, figsize=(15, 5))
for ax, (_, r) in zip(axs.ravel(), df.sample(12, random_state=0).iterrows()):
    ax.imshow(Image.open(OUT / r.crop_path)); ax.set_title(r.concept_name[:22], fontsize=8); ax.axis("off")
plt.show()
""")
md("## 6. Taxonomy from WoRMS\n`fathomnet.api.worms.get_ancestors` returns a nested tree root → taxon. **Bug to know:** for some leaf taxa the `rank` is an empty string – we infer it from the name (`label_rank_inferred=1`).")
code("""
from fathomnet.api import worms
n = worms.get_ancestors("Terebellidae")
while n:
    print(repr(n.rank).ljust(16), n.name); n = n.children[0] if n.children else None
""")
code("""
OUT.mkdir(exist_ok=True)
tax_rows, tax_err = P.build_taxonomy([c["name"] for c in cats.values()], OUT / "taxonomy_cache.json")   # cached: 1 query per taxon
tax = pd.DataFrame(tax_rows); print("lookup errors:", tax_err)
display(tax[tax.label_rank_inferred == 1]); tax.head(8)
""")
md("## 7. Taxonomic analysis\nSpecies-level labels are only ~40% of the classes; the others are genus/family/order/class/phylum labels – hierarchical structure is real and mixed-rank.")
code("""
s = P.analyse(report, cats, tax_rows, tax_err, rows, len(failures), len(data["annotations"]), failures)
print(P.format_report(s, P.human(P.dir_size(OUT))))
""")
code("""
# Numbers for the FULL dataset from the verified reference run (no download needed)
ref = json.load(open("reference_output/summary.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a_, rk in zip(ax, ["phylum", "class"]):
    g = pd.Series(ref["samples_per_group"][rk]).head(12)[::-1]
    a_.barh(g.index, g.values, color="#3b6ea5"); a_.set_title(f"ROIs per {rk} (full dataset)")
plt.tight_layout(); plt.show()
""")
md("""
## 8. Full run and known issues
```bash
python prepare_fathomnet2025.py            # all 8,981 images, resumable
```
Expected: 23,698 crops, 1 failure (annotation 21685), 79/79 categories present. See `KNOWN_ISSUES.md` for every pitfall
found (bbox clipping, empty WoRMS ranks, duplicated boxes, tiny objects, …).
""")
nb.cells = C
nbf.write(nb, "fathomnet2025_walkthrough.ipynb")
