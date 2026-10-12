"""One call at the top of every lab notebook: works on Google Colab and on your own computer.

    from setup_env import setup
    ROOT = setup()            # full dataset (23,698 crops)
    ROOT = setup(per_category=100)   # quick balanced subset (~7,900 crops)

On Colab the dataset is zipped to your Google Drive after the first download (if Drive is mounted),
so the next session restores it in seconds instead of downloading for ~15 minutes again.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/TorbenGl/fathomnetchallange"
COURSE = Path(__file__).resolve().parent
DRIVE_ZIP = Path("/content/drive/MyDrive/fathomnet2025_compact.zip")


def in_colab():
    return "google.colab" in sys.modules


def setup(per_category=None, mount_drive=True):
    if str(COURSE) not in sys.path:
        sys.path.insert(0, str(COURSE))
    from classifier_lab import data

    out = COURSE.parent / "fathomnet2025_compact"
    if in_colab():
        if mount_drive and not Path("/content/drive").exists():
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception as e:  # noqa: BLE001 - Drive is optional
                print("Google Drive not mounted:", e)
        if not (out / "annotations.csv").exists() and DRIVE_ZIP.exists():
            print("Restoring the dataset from Google Drive ...")
            shutil.unpack_archive(DRIVE_ZIP, out.parent)
    try:
        root = data.find_data()
    except FileNotFoundError:
        print("Dataset not found - downloading it now (images are streamed, only crops are stored).")
        root = data.ensure_dataset(per_category=per_category, out=out)
        if in_colab() and DRIVE_ZIP.parent.exists() and not DRIVE_ZIP.exists():
            print("Saving a copy to Google Drive for your next session ...")
            shutil.make_archive(str(DRIVE_ZIP.with_suffix("")), "zip", out.parent, out.name)
    n = sum(1 for _ in open(root / "annotations.csv")) - 1
    print(f"Dataset ready: {root}  ({n} crop files; exact duplicate boxes are dropped when loading)")
    return root


def colab_bootstrap():
    """Code for the very first cell on Colab (get the repository + packages). Kept here for reference."""
    repo = Path("/content/fathomnetchallange")
    if not repo.exists():
        if Path("/content/fathomnetchallange.zip").exists():
            shutil.unpack_archive("/content/fathomnetchallange.zip", "/content")
        else:
            subprocess.run(["git", "clone", "-q", REPO_URL, str(repo)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "fathomnet"], check=True)
    os.chdir(repo / "course")
