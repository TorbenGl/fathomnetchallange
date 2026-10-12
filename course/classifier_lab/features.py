"""Turn crops into feature vectors (one row of numbers per animal).

A classifier never sees "an animal" - it sees a list of numbers. Which numbers we give it
matters at least as much as which classifier we choose. Three families, from naive to strong:

    pixels(df)          the raw colours of a tiny 16x16 version of the crop     (768 numbers)
    colour_stats(df)    hand-made descriptors a biologist could name           (~30 numbers)
    cnn_embedding(df)   what an ImageNet-pretrained CNN "sees" (penultimate layer, 512 numbers)

All functions return a float32 array with one row per row of `df` (same order).
Results are cached in features_cache/ so you compute each feature set only once.
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CACHE = Path(__file__).resolve().parent.parent / "features_cache"

COLOUR_STAT_NAMES = (
    ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "brightness", "saturation",
     "redness", "contrast", "edge_strength", "centre_minus_border"]
    + [f"hue_bin_{i}" for i in range(12)] + [f"brightness_bin_{i}" for i in range(6)]
)


def _cached(name, df, fn):
    ids = df["annotation_id"].to_numpy()
    key = hashlib.sha1(ids.tobytes()).hexdigest()[:10]
    f = CACHE / f"{name}_{len(ids)}_{key}.npy"
    if f.exists():
        return np.load(f)
    x = fn(list(df["path"])).astype(np.float32)
    CACHE.mkdir(exist_ok=True)
    np.save(f, x)
    return x


def _map(fn, paths, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(fn, paths)))


# ----------------------------------------------------------------------------- raw pixels
def pixels(df, size=16):
    """The crop shrunk to size x size, all pixel values in a row (0..1)."""
    def one(p):
        return np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR),
                          np.float32).ravel() / 255
    return _cached(f"pixels{size}", df, lambda paths: _map(one, paths))


# ----------------------------------------------------------------------------- hand-made
def _colour_stats_one(p):
    im = Image.open(p).convert("RGB").resize((64, 64), Image.BILINEAR)
    rgb = np.asarray(im, np.float32) / 255
    hsv = np.asarray(im.convert("HSV"), np.float32) / 255
    grey = rgb.mean(axis=2)
    gy, gx = np.gradient(grey)
    centre = grey[16:48, 16:48].mean()
    border = (grey.sum() - grey[16:48, 16:48].sum()) / (64 * 64 - 32 * 32)
    hue_hist = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=hsv[..., 1])[0]
    bright_hist = np.histogram(grey, bins=6, range=(0, 1))[0]
    return np.concatenate([
        rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
        [grey.mean(), hsv[..., 1].mean(), (rgb[..., 0] - rgb[..., 2]).mean(), grey.std(),
         np.hypot(gx, gy).mean(), centre - border],
        hue_hist / (hue_hist.sum() + 1e-6), bright_hist / bright_hist.sum(),
    ])


def colour_stats(df):
    """30 descriptors you could explain to a field biologist: mean colour, brightness, redness,
    contrast, edge strength ("texture"), is the centre brighter than the background,
    a 12-bin hue histogram and a 6-bin brightness histogram. Names: COLOUR_STAT_NAMES."""
    return _cached("colourstats", df, lambda paths: _map(_colour_stats_one, paths))


# ----------------------------------------------------------------------------- CNN embedding
BACKBONES = {
    # name: (torchvision builder, weights, embedding size)
    "resnet18": ("resnet18", "IMAGENET1K_V1", 512),
    "resnet50": ("resnet50", "IMAGENET1K_V2", 2048),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1", 1280),
}


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(name="resnet18"):
    """An ImageNet-pretrained CNN with its classification layer removed (outputs the embedding)."""
    import torch
    import torchvision

    builder, weights, _ = BACKBONES[name]
    net = getattr(torchvision.models, builder)(weights=weights)
    if hasattr(net, "fc"):
        net.fc = torch.nn.Identity()
    else:
        net.classifier = torch.nn.Identity()
    return net.eval()


def imagenet_transform(size=224):
    import torchvision.transforms as T
    return T.Compose([T.Resize((size, size)), T.ToTensor(),
                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class _Crops:
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def cnn_embedding(df, backbone="resnet18", batch_size=64, workers=4, progress=True):
    """Penultimate-layer activations of a pretrained CNN (no training involved).

    resnet18 on all 23,664 crops: ~13 min measured on 4 CPU cores; a Colab GPU should take a few minutes.
    """
    def compute(paths):
        import torch
        dev = device()
        net = load_backbone(backbone).to(dev)
        dl = torch.utils.data.DataLoader(_Crops(paths, imagenet_transform()), batch_size=batch_size,
                                         num_workers=workers)
        out = []
        with torch.no_grad():
            for i, xb in enumerate(dl):
                out.append(net(xb.to(dev)).float().cpu().numpy())
                if progress and (i % 20 == 0 or i == len(dl) - 1):
                    print(f"\r  {backbone}: {min((i + 1) * batch_size, len(paths))}/{len(paths)} crops",
                          end="", flush=True)
        if progress:
            print()
        return np.concatenate(out)
    return _cached(f"cnn_{backbone}", df, compute)


FEATURES = {"pixels": pixels, "colour_stats": colour_stats, "cnn": cnn_embedding}


def get(name, df):
    """get('pixels' | 'colour_stats' | 'cnn' | 'cnn_resnet50' | 'cnn_efficientnet_b0', df)"""
    if name.startswith("cnn_"):
        return cnn_embedding(df, backbone=name[4:])
    return FEATURES[name](df)
