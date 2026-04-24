"""
Project setup — create required directories and download SAM2 weights.

Usage:
  python setup.py
  python setup.py --model small    # download smaller/faster model
  python setup.py --skip-download  # only create folders
"""

import os
import argparse
import urllib.request


MODELS = {
    "large": {
        "filename": "sam2_hiera_large.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "size": "2.4 GB",
    },
    "base": {
        "filename": "sam2_hiera_base_plus.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "size": "0.9 GB",
    },
    "small": {
        "filename": "sam2_hiera_small.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "size": "0.4 GB",
    },
    "tiny": {
        "filename": "sam2_hiera_tiny.pt",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "size": "0.15 GB",
    },
}

DIRS = [
    "projects",
    "models",
    "screenshots",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Project setup")
    parser.add_argument(
        "--model",
        default="large",
        choices=MODELS.keys(),
        help="SAM2 model size to download (default: large)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only create directories, skip model download",
    )
    return parser.parse_args()


def create_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  {d}/")


def download_model(model_key):
    info     = MODELS[model_key]
    dest     = os.path.join("models", info["filename"])

    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading {info['filename']} ({info['size']})...")
    print(f"  URL: {info['url']}")

    def progress(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100, 100)
        mb  = count * block_size / 1024 / 1024
        print(f"\r  {pct:.1f}%  {mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(info["url"], dest, reporthook=progress)
    print(f"\r  Done: {dest}                    ")


def main():
    args = parse_args()

    print("Creating directories...")
    create_dirs()

    if not args.skip_download:
        print(f"\nDownloading SAM2 model ({args.model})...")
        download_model(args.model)

    print("\nSetup complete.")
    print("\nNext steps:")
    print("  1. Place your videos in  projects/{project_name}/")
    print("  2. Run:  python sam2_annotate.py --project {project_name} --classes YourClass")


if __name__ == "__main__":
    main()
