"""Offline end-to-end test: local fake image server + fake taxonomy (no internet needed).

    python tests/test_offline.py
"""
import io, json, shutil, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_fathomnet2025 as P

OFFICIAL = Path(sys.argv[1] if len(sys.argv) > 1 else "fgvc-comp-2025")
SIZES = {}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        iid = int(self.path.strip("/").split(".")[0])
        w, h = SIZES[iid]
        arr = np.random.randint(1, 255, (h, w, 3), dtype=np.uint8)
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
        self.send_response(200); self.end_headers(); self.wfile.write(buf.getvalue())
    def log_message(self, *a): pass


def fake_lookup(name):
    return {"label_rank": "species", "kingdom": "Animalia", "phylum": "P" + name[:1],
            "class": "C" + name[:2], "species": name}


tmp = Path(tempfile.mkdtemp())
srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
repo = tmp / "repo"; (repo / "datasets").mkdir(parents=True)
d = json.loads((OFFICIAL / "datasets/dataset_train.json").read_text())
for im in d["images"]:
    SIZES[im["id"]] = (im["width"], im["height"])
    im["coco_url"] = f"http://127.0.0.1:{srv.server_port}/{im['id']}.png"
(repo / "datasets/dataset_train.json").write_text(json.dumps(d))
out = tmp / "out"
rc = P.main(["--repo-dir", str(repo), "--out", str(out), "--limit-images", "60", "--workers", "4"],
            taxonomy_lookup=fake_lookup)
assert rc == 0, "failures reported"
import csv
rows = list(csv.DictReader(open(out / "annotations.csv")))
assert rows and all(Image.open(out / r["crop_path"]).size == (224, 224) for r in rows)
assert len({r["crop_path"] for r in rows}) == len(rows)
for f in ["categories.csv", "taxonomy.csv", "dataset_train.json", "dataset_info.json", "summary.json"]:
    assert (out / f).exists(), f
assert not list(out.rglob("*.png")), "original images must not be stored"
print("OFFLINE TEST PASSED:", len(rows), "crops")
shutil.rmtree(tmp)
