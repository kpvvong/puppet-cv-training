"""
Split train/val/test dataset and generate data.yaml
-----------------------------------------------------
Frames are grouped by source video (vid000_*, vid001_*, ...).
Each video is split individually with 70/15/15 ratio.
If a video produces fewer than MIN_SPLIT_FRAMES frames for val or test,
all frames from that video go to train instead.

Usage:
  python prepare_dataset.py --project <project_name>
  python prepare_dataset.py --input dataset_raw --output dataset
"""

import os
import re
import random
import shutil
import argparse
from collections import defaultdict


MIN_SPLIT_FRAMES = 8   # minimum frames required for val/test split


def parse_args():
    parser = argparse.ArgumentParser(description="Split train/val/test dataset and generate data.yaml")
    parser.add_argument("--project",    default=None,           help="Project name (auto-uses projects/{name}/ paths)")
    parser.add_argument("--input",      default=None,           help="Input dataset_raw folder")
    parser.add_argument("--output",     default=None,           help="Output dataset folder")
    parser.add_argument("--train-ratio",default=0.70, type=float, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio",  default=0.15, type=float, help="Val ratio (default: 0.15)")
    parser.add_argument("--seed",       default=42,   type=int,   help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.project:
        project_dir = os.path.join("projects", args.project)
        args.input  = args.input  or os.path.join(project_dir, "dataset_raw")
        args.output = args.output or os.path.join(project_dir, "dataset")
    else:
        args.input  = args.input  or "dataset_raw"
        args.output = args.output or "dataset"

    args.test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 6)
    return args


def read_classes(input_dir):
    path = os.path.join(input_dir, "classes.txt")
    if not os.path.exists(path):
        print("[WARN] classes.txt not found, fill in data.yaml names manually")
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def group_by_video(images_dir):
    """Group image filenames by video prefix (vid000, vid001, ...).

    Files without a vid prefix are grouped under key None (legacy format).
    """
    groups = defaultdict(list)
    pattern = re.compile(r'^(vid\d+)_')
    for f in sorted(os.listdir(images_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = pattern.match(f)
        key = m.group(1) if m else None
        groups[key].append(f)
    return groups


def split_video_frames(frames, train_ratio, val_ratio, seed):
    """Split a single video's frames into train/val/test.

    If val or test would have fewer than MIN_SPLIT_FRAMES,
    all frames go to train.
    """
    random.seed(seed)
    shuffled = frames[:]
    # random.shuffle(shuffled)  # Commented out to keep frames in order, which can be important for videos

    n = len(shuffled)
    n_val  = int(n * val_ratio)
    n_test = int(n * (1.0 - train_ratio - val_ratio))

    if n_val < MIN_SPLIT_FRAMES or n_test < MIN_SPLIT_FRAMES:
        return {"train": shuffled, "val": [], "test": []}

    n_train = n - n_val - n_test
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def split_dataset(input_dir, output_dir, train_ratio, val_ratio, seed):
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    groups = group_by_video(images_dir)
    totals = {"train": 0, "val": 0, "test": 0}

    for vid_key, frames in sorted(groups.items()):
        label = vid_key or "no-prefix"
        splits = split_video_frames(frames, train_ratio, val_ratio, seed)

        all_train = len(splits["val"]) == 0
        if all_train:
            print(f"  [{label}] {len(frames)} frames → train only (too few for val/test split)")
        else:
            print(f"  [{label}] {len(frames)} frames → "
                  f"train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}")

        for split, files in splits.items():
            for img_file in files:
                stem = os.path.splitext(img_file)[0]
                shutil.copy(
                    os.path.join(images_dir, img_file),
                    os.path.join(output_dir, "images", split, img_file),
                )
                label_src = os.path.join(labels_dir, f"{stem}.txt")
                label_dst = os.path.join(output_dir, "labels", split, f"{stem}.txt")
                if os.path.exists(label_src):
                    shutil.copy(label_src, label_dst)
                else:
                    open(label_dst, "w").close()
            totals[split] += len(files)

    print(f"\n  Total — train: {totals['train']}  val: {totals['val']}  test: {totals['test']}")
    return totals


def write_yaml(output_dir, classes):
    nc = len(classes)
    names_str = str(classes) if classes else "['class0']  # fill in your class names"
    yaml_content = f"""# YOLOv8 Dataset Config
path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
test:  images/test

nc: {nc}
names: {names_str}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"data.yaml written: {yaml_path}")
    if not classes:
        print("[!] Edit data.yaml and fill in the correct names")


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] {args.input} not found, run sam2_annotate.py or extract_frames.py first")
        exit(1)

    classes = read_classes(args.input)
    print(f"Classes: {classes}")
    print(f"Split ratio — train {args.train_ratio*100:.0f}% / val {args.val_ratio*100:.0f}% / test {args.test_ratio*100:.0f}%")
    print(f"Min frames for val/test split: {MIN_SPLIT_FRAMES}\n")

    split_dataset(args.input, args.output, args.train_ratio, args.val_ratio, args.seed)
    write_yaml(args.output, classes)

    print(f"\nDone! Dataset saved to {args.output}/")
