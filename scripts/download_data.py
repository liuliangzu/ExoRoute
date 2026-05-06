"""
Dataset download script for REGATE experiments.
Downloads Elliptic, BTC-Alpha, and BTC-OTC datasets.

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset elliptic
    python scripts/download_data.py --dataset btc-alpha
    python scripts/download_data.py --dataset btc-otc
"""
import os
import argparse
import subprocess
import zipfile
import tarfile
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_file(url: str, dest: Path):
    """Download a file with progress."""
    print(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%", end="", flush=True)
    print()


def download_elliptic():
    """
    Download the Elliptic dataset.
    The Elliptic dataset is available on Kaggle:
      https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

    You need Kaggle API credentials (~/.kaggle/kaggle.json).
    Alternatively, download manually and place files under data/elliptic/.
    """
    dest_dir = DATA_DIR / "elliptic"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if (dest_dir / "elliptic_txs_features.csv").exists():
        print("Elliptic dataset already exists, skipping.")
        return

    print("=" * 60)
    print("Downloading Elliptic dataset from Kaggle...")
    print("Requires: pip install kaggle && ~/.kaggle/kaggle.json")
    print("=" * 60)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "ellipticco/elliptic-data-set",
                "-p", str(dest_dir),
                "--unzip",
            ],
            check=True,
        )
        print("Elliptic dataset downloaded successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "\nFailed to download via Kaggle API. Please download manually:\n"
            "  1. Visit https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n"
            "  2. Download and extract to data/elliptic/\n"
            "  Expected files:\n"
            "    - elliptic_txs_features.csv\n"
            "    - elliptic_txs_edgelist.csv\n"
            "    - elliptic_txs_classes.csv\n"
        )


def download_btc_alpha():
    """Download BTC-Alpha trust network from Stanford SNAP."""
    dest_dir = DATA_DIR / "btc_alpha"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "soc-sign-bitcoinalpha.csv"
    if csv_path.exists():
        print("BTC-Alpha dataset already exists, skipping.")
        return

    url = "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
    gz_path = dest_dir / "soc-sign-bitcoinalpha.csv.gz"
    download_file(url, gz_path)

    import gzip
    import shutil
    with gzip.open(gz_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print("BTC-Alpha dataset downloaded successfully.")


def download_btc_otc():
    """Download BTC-OTC trust network from Stanford SNAP."""
    dest_dir = DATA_DIR / "btc_otc"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "soc-sign-bitcoinotc.csv"
    if csv_path.exists():
        print("BTC-OTC dataset already exists, skipping.")
        return

    url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
    gz_path = dest_dir / "soc-sign-bitcoinotc.csv.gz"
    download_file(url, gz_path)

    import gzip
    import shutil
    with gzip.open(gz_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    print("BTC-OTC dataset downloaded successfully.")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for REGATE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "elliptic", "btc-alpha", "btc-otc"],
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "elliptic"):
        download_elliptic()
    if args.dataset in ("all", "btc-alpha"):
        download_btc_alpha()
    if args.dataset in ("all", "btc-otc"):
        download_btc_otc()

    print("\nDone. Data directory:", DATA_DIR)


if __name__ == "__main__":
    main()
