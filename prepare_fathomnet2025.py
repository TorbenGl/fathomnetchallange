#!/usr/bin/env python3
"""Build a compact 224x224 ROI-crop version of the FathomNet 2025 (FGVC12) training set.

Source of truth: datasets/dataset_train.json of https://github.com/fathomnet/fgvc-comp-2025
Every image is downloaded into memory, all its annotated ROIs are cropped, resized and saved
as JPEG, and the full-resolution image is discarded (never written to disk).

Usage
-----
    pip install -r requirements.txt
    python prepare_fathomnet2025.py --limit-images 20      # quick smoke test (~1 min)
    python prepare_fathomnet2025.py                          # full run (8,981 images)

The run is resumable: crops that already exist are not downloaded again, and taxonomy
lookups are cached in <out>/taxonomy_cache.json.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image

SCRIPT_VERSION = "1.0.0"
REPO_URL = "https://github.com/fathomnet/fgvc-comp-2025"
SIZE = 224
JPEG_QUALITY = 95
INTERPOLATIONS = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
LABEL_RANK_ORDER = ["species", "genus", "family", "order", "class", "phylum"]
ANN_FIELDS = ["annotation_id", "image_id", "category_id", "concept_name", "crop_path",
              "bbox_x", "bbox_y", "bbox_width", "bbox_height", "source_url",
              "bbox_clipped", "image_width", "image_height"]


# --------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def get_official_repo(repo_dir):
    """Clone the official repo (shallow) if it is not there yet; return its path."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "datasets" / "dataset_train.json").exists():
        print(f"Cloning {REPO_URL} -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    return repo_dir


def dir_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------- integrity
def check_json(data):
    """Integrity checks on the official JSON. Returns (report dict, images-by-id, cats-by-id)."""
    images = {im["id"]: im for im in data["images"]}
    cats = {c["id"]: c for c in data["categories"]}
    anns = data["annotations"]
    ann_ids = [a["id"] for a in anns]
    per_cat = Counter(a["category_id"] for a in anns)
    bad_img = [a["id"] for a in anns if a["image_id"] not in images]
    bad_cat = [a["id"] for a in anns if a["category_id"] not in cats]
    outside = []          # bbox not fully inside its image (declared size in JSON)
    degenerate = []       # zero/negative area after clipping
    for a in anns:
        im = images.get(a["image_id"])
        if im is None:
            continue
        x, y, w, h = a["bbox"]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(im["width"], x + w), min(im["height"], y + h)
        if x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
            outside.append(a["id"])
        if x1 <= x0 or y1 <= y0:
            degenerate.append(a["id"])
    counts = sorted(per_cat.values())
    report = {
        "n_images": len(images),
        "n_categories": len(cats),
        "n_annotations": len(anns),
        "expected_annotations_if_79x300": 79 * 300,
        "unique_annotation_ids": len(set(ann_ids)) == len(ann_ids),
        "annotations_with_invalid_image": len(bad_img),
        "annotations_with_invalid_category": len(bad_cat),
        "bboxes_extending_outside_image_(will_be_clipped)": len(outside),
        "bboxes_empty_after_clipping": len(degenerate),
        "categories_without_annotations": [c["name"] for cid, c in cats.items() if per_cat[cid] == 0],
        "per_category_min": counts[0], "per_category_max": counts[-1],
        "per_category_median": counts[len(counts) // 2],
        "categories_not_exactly_300": {cats[c]["name"]: n for c, n in sorted(per_cat.items())
                                       if n != 300},
    }
    return report, images, cats


# --------------------------------------------------------------------------- crops
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "fathomnet2025-compact/" + SCRIPT_VERSION
        _local.s = s
    return _local.s


def fetch_image(url, retries=4, timeout=60):
    """Download into memory. Returns PIL image (RGB-converted lazily by the caller)."""
    err = None
    for attempt in range(retries):
        try:
            r = session().get(url, timeout=timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.load()
            return img
        except Exception as e:  # noqa: BLE001 - retry on any network/decoding problem
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{type(err).__name__}: {err}")


def process_image(image, anns, out_images, resample, force):
    """Download one image, write all its crops. Returns (rows, failures)."""
    rows, failures = [], []
    todo = []
    for a in anns:
        name = f"{image['id']}_{a['id']}.jpg"
        if (out_images / name).exists() and not force:
            todo.append((a, name, True))
        else:
            todo.append((a, name, False))
    img = None
    if any(not done for _, _, done in todo):
        try:
            img = fetch_image(image["coco_url"])
        except Exception as e:  # noqa: BLE001
            return [], [(a["id"], image["id"], image["coco_url"], str(e)) for a, _, _ in todo]
    W, H = (img.size if img is not None else (image["width"], image["height"]))
    for a, name, done in todo:
        x, y, w, h = a["bbox"]
        left, top, right, bottom = x, y, x + w, y + h
        cl, ct, cr, cb = max(0, left), max(0, top), min(W, right), min(H, bottom)
        clipped = (cl, ct, cr, cb) != (left, top, right, bottom)
        if cr <= cl or cb <= ct:
            failures.append((a["id"], image["id"], image["coco_url"], "empty bbox after clipping"))
            continue
        if not done:
            try:
                crop = img.crop((int(cl), int(ct), int(round(cr)), int(round(cb)))).convert("RGB")
                crop = crop.resize((SIZE, SIZE), resample)
                tmp = out_images / (name + ".tmp")
                crop.save(tmp, format="JPEG", quality=JPEG_QUALITY)
                tmp.rename(out_images / name)
            except Exception as e:  # noqa: BLE001
                failures.append((a["id"], image["id"], image["coco_url"], f"crop failed: {e}"))
                continue
        rows.append({"annotation_id": a["id"], "image_id": image["id"],
                     "category_id": a["category_id"], "crop_path": f"images/{name}",
                     "bbox_x": x, "bbox_y": y, "bbox_width": w, "bbox_height": h,
                     "source_url": image["coco_url"], "bbox_clipped": int(clipped),
                     "image_width": W, "image_height": H})
    return rows, failures


def build_crops(images, anns, cats, out_dir, workers, resample, force):
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    by_image = defaultdict(list)
    for a in anns:
        by_image[a["image_id"]].append(a)
    rows, failures = [], []
    total = len(by_image)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_image, images[i], al, out_images, resample, force)
                for i, al in by_image.items()]
        for n, f in enumerate(as_completed(futs), 1):
            r, fl = f.result()
            rows += r
            failures += fl
            if n % 100 == 0 or n == total:
                print(f"  images {n}/{total}  crops {len(rows)}  failed {len(failures)}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    for r in rows:
        r["concept_name"] = cats[r["category_id"]]["name"]
    rows.sort(key=lambda r: r["annotation_id"])
    return rows, failures


def verify_crops(rows):
    """Re-open every crop: must be 224x224 and non-empty."""
    bad = []
    for r in rows:
        p = r["_abs"]
        try:
            with Image.open(p) as im:
                if im.size != (SIZE, SIZE) or im.mode != "RGB":
                    bad.append((r["annotation_id"], f"size/mode {im.size} {im.mode}"))
                elif p.stat().st_size == 0 or im.getextrema() == ((0, 0),) * 3:
                    bad.append((r["annotation_id"], "empty/black"))
        except Exception as e:  # noqa: BLE001
            bad.append((r["annotation_id"], str(e)))
    return bad


# --------------------------------------------------------------------------- taxonomy
def worms_lineage(name):
    """Query FathomNet's WoRMS interface: {'label_rank':..., 'kingdom':..., ...}."""
    from fathomnet.api import worms
    node = worms.get_ancestors(name)      # nested tree root -> ... -> taxon
    lineage = {}
    label_rank = None
    while node is not None:
        rank = (node.rank or "").lower()
        if rank in RANKS:
            lineage[rank] = node.name
        label_rank = rank
        node = node.children[0] if node.children else None
    if not label_rank:
        # WoRMS returns an empty rank for some leaf taxa (e.g. Terebellidae): infer from the name
        last = name.split()
        label_rank = ("species" if len(last) == 2 else "family" if name.endswith("idae")
                      else "subfamily" if name.endswith("inae") else "")
        lineage["label_rank_inferred"] = True
        if label_rank in RANKS:
            lineage[label_rank] = name
        if label_rank == "species":
            lineage.setdefault("genus", last[0])
    lineage["label_rank"] = label_rank
    return lineage


def build_taxonomy(names, cache_path, lookup=worms_lineage):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows, errors = [], {}
    for name in names:
        if name not in cache:
            try:
                cache[name] = lookup(name)
                cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
            except Exception as e:  # noqa: BLE001
                errors[name] = f"{type(e).__name__}: {e}"
                cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
                rows.append({"concept_name": name, "label_rank": "", **{r: "" for r in RANKS}})
                continue
        lin = cache[name]
        rows.append({"concept_name": name, "label_rank": lin.get("label_rank", ""),
                     "label_rank_inferred": int(bool(lin.get("label_rank_inferred"))),
                     **{r: lin.get(r, "") or "" for r in RANKS}})
    return rows, errors


# --------------------------------------------------------------------------- analysis
def analyse(report, cats, tax_rows, tax_errors, ann_rows, n_failed, n_json_anns, failures):
    tax = {r["concept_name"]: r for r in tax_rows}
    per_cat = Counter(r["concept_name"] for r in ann_rows)
    s = {}
    s["categories"] = len(cats)
    s["annotations_in_json"] = n_json_anns
    s["crops_generated"] = len(ann_rows)
    s["failed"] = n_failed
    s["taxonomy_lookup_errors"] = tax_errors
    s["label_rank_counts"] = {rk: sum(1 for t in tax.values() if t["label_rank"] == rk)
                              for rk in LABEL_RANK_ORDER}
    s["label_rank_counts"]["other/unknown"] = len(tax) - sum(s["label_rank_counts"].values())
    s["unique_groups"] = {rk: len({t[rk] for t in tax.values() if t[rk]})
                          for rk in ["phylum", "class", "order", "family", "genus", "species"]}
    s["samples_per_group"] = {}
    for rk in ["phylum", "class", "order", "family"]:
        c = Counter()
        for r in ann_rows:
            c[tax[r["concept_name"]][rk] or "(label above this rank)"] += 1
        s["samples_per_group"][rk] = dict(c.most_common())
    s["top10_groups"] = {rk: list(s["samples_per_group"][rk].items())[:10]
                         for rk in ["phylum", "class", "order", "family"]}
    s["categories_lost"] = [c["name"] for c in cats.values() if per_cat[c["name"]] == 0]
    counts = sorted(per_cat.values())
    s["per_category_samples"] = {"min": counts[0] if counts else 0, "max": counts[-1] if counts else 0,
                                 "categories_with_exactly_300": sum(1 for v in counts if v == 300),
                                 "deviating": {k: v for k, v in sorted(per_cat.items()) if v != 300}}
    s["integrity"] = report
    return s


def format_report(s, disk):
    L = ["FathomNet 2025 compact dataset", "",
         f"Categories: {s['categories']}",
         f"Annotations in official JSON: {s['annotations_in_json']} "
         f"(nominal 79 x 300 = {s['integrity']['expected_annotations_if_79x300']})",
         f"Crops successfully generated: {s['crops_generated']}",
         f"Failed: {s['failed']}", "", "Label ranks:"]
    L += [f"  {k}: {v}" for k, v in s["label_rank_counts"].items()]
    L += ["", "Unique taxonomic groups:"]
    plural = {"phylum": "phyla", "class": "classes", "order": "orders", "family": "families",
              "genus": "genera", "species": "species"}
    L += [f"  {plural[k]}: {v}" for k, v in s["unique_groups"].items()]
    L += ["", f"Disk usage: {disk}", ""]
    for rk in ["phylum", "class", "order", "family"]:
        L.append(f"Top 10 {rk} groups by ROI samples:")
        L += [f"  {n:>6}  {g}" for g, n in s["top10_groups"][rk]]
        L.append("")
    p = s["per_category_samples"]
    L.append(f"Samples per category: min {p['min']}, max {p['max']}; "
             f"{p['categories_with_exactly_300']}/{s['categories']} categories have exactly 300")
    if p["deviating"]:
        L.append("  deviating: " + ", ".join(f"{k}={v}" for k, v in p["deviating"].items()))
    L.append("Categories lost: " + (", ".join(s["categories_lost"]) or "none"))
    if s["taxonomy_lookup_errors"]:
        L.append("Taxonomy lookup errors: " + "; ".join(f"{k}: {v}" for k, v in
                                                        s["taxonomy_lookup_errors"].items()))
    L.append("")
    L.append("JSON integrity: " + json.dumps({k: v for k, v in s["integrity"].items()
                                              if k != "categories_not_exactly_300"}))
    return "\n".join(L)


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- main
def main(argv=None, taxonomy_lookup=worms_lineage):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="fathomnet2025_compact")
    ap.add_argument("--repo-dir", default="fgvc-comp-2025",
                    help="checkout of fathomnet/fgvc-comp-2025 (cloned if missing)")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--interpolation", choices=INTERPOLATIONS, default="bicubic",
                    help="PIL resize filter (bicubic = PIL default)")
    ap.add_argument("--limit-images", type=int, help="only process the first N images (smoke test)")
    ap.add_argument("--per-category", type=int,
                    help="only the first N annotations (by id) of every category: a balanced, faster subset")
    ap.add_argument("--force", action="store_true", help="re-create crops that already exist")
    ap.add_argument("--skip-taxonomy", action="store_true")
    args = ap.parse_args(argv)

    repo = get_official_repo(args.repo_dir)
    src_json = repo / "datasets" / "dataset_train.json"
    data = json.loads(src_json.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_json, out / "dataset_train.json")

    print("Checking official JSON ...")
    report, images, cats = check_json(data)
    for k, v in report.items():
        if k != "categories_not_exactly_300":
            print(f"  {k}: {v}")
    print(f"  categories_not_exactly_300: {len(report['categories_not_exactly_300'])}")

    anns = data["annotations"]
    if args.limit_images:
        keep = {im["id"] for im in data["images"][:args.limit_images]}
        anns = [a for a in anns if a["image_id"] in keep]
        print(f"SMOKE TEST: limited to {len(keep)} images / {len(anns)} annotations")
    if args.per_category:
        seen = Counter()
        kept = []
        for a in sorted(anns, key=lambda a: a["id"]):
            if seen[a["category_id"]] < args.per_category:
                seen[a["category_id"]] += 1
                kept.append(a)
        anns = kept
        print(f"SUBSET: at most {args.per_category} annotations per category -> {len(anns)} annotations")
    anns = [a for a in anns if a["image_id"] in images and a["category_id"] in cats]

    print("Downloading + cropping ...")
    rows, failures = build_crops(images, anns, cats, out, args.workers,
                                 INTERPOLATIONS[args.interpolation], args.force)
    for r in rows:
        r["_abs"] = out / r["crop_path"]
    bad = verify_crops(rows)
    bad_ids = {i for i, _ in bad}
    rows = [r for r in rows if r["annotation_id"] not in bad_ids]
    failures += [(i, "", "", f"verification: {m}") for i, m in bad]
    write_csv(out / "annotations.csv", ANN_FIELDS, rows)
    write_csv(out / "failures.csv", ["annotation_id", "image_id", "url", "reason"],
              [dict(zip(["annotation_id", "image_id", "url", "reason"], f)) for f in failures])
    write_csv(out / "categories.csv", ["category_id", "concept_name", "n_annotations_json", "n_crops"],
              [{"category_id": c["id"], "concept_name": c["name"],
                "n_annotations_json": sum(1 for a in data["annotations"] if a["category_id"] == c["id"]),
                "n_crops": sum(1 for r in rows if r["category_id"] == c["id"])}
               for c in sorted(cats.values(), key=lambda c: c["id"])])

    tax_rows, tax_errors = [], {}
    if not args.skip_taxonomy:
        print("Querying taxonomy (cached) ...")
        tax_rows, tax_errors = build_taxonomy([c["name"] for c in cats.values()],
                                              out / "taxonomy_cache.json", taxonomy_lookup)
        write_csv(out / "taxonomy.csv", ["concept_name", "label_rank", "label_rank_inferred"] + RANKS, tax_rows)

    disk = human(dir_size(out))
    if tax_rows:
        s = analyse(report, cats, tax_rows, tax_errors, rows, len(failures), len(data["annotations"]), failures)
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        text = format_report(s, disk)
        (out / "summary.txt").write_text(text)
        print("\n" + text)

    info = {
        "official_repo": REPO_URL,
        "official_repo_commit": git(repo, "rev-parse", "HEAD"),
        "dataset_train_json_sha256": sha256_file(src_json),
        "crop_size": [SIZE, SIZE],
        "image_format": "JPEG", "jpeg_quality": JPEG_QUALITY,
        "resize_interpolation": f"PIL {args.interpolation}",
        "crop": "PIL crop (x, y, x+w, y+h), clipped to image bounds, converted to RGB",
        "script": Path(__file__).name, "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(__file__),
        "script_git_commit": git(Path(__file__).resolve().parent, "rev-parse", "HEAD"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "pillow": Image.__version__,
        "smoke_test_limit_images": args.limit_images,
        "per_category_limit": args.per_category,
        "n_annotations_in_json": len(data["annotations"]),
        "n_crops": len(rows), "n_failed": len(failures),
    }
    try:
        import importlib.metadata as md
        info["fathomnet_package"] = md.version("fathomnet")
    except Exception:  # noqa: BLE001
        pass
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
