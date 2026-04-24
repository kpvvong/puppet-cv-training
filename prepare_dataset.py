"""
Split train/val dataset and generate data.yaml
-----------------------------------------------
Run extract_frames.py first.

Usage:
  # Auto-resolve paths by project name
  python prepare_dataset.py --project <project_name>

  # Specify paths manually
  python prepare_dataset.py [--input dataset_raw] [--output dataset] [--val-ratio 0.2]

Output:
  <output>/
  ├── images/train/ & images/val/
  ├── labels/train/ & labels/val/
  └── data.yaml
"""

import os
import random
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split train/val dataset and generate data.yaml")
    parser.add_argument("--project",   default=None,          help="Project name, auto-uses paths under projects/{name}/")
    parser.add_argument("--input",     default=None,          help="Input folder from extract_frames (default: projects/{name}/dataset_raw)")
    parser.add_argument("--output",    default=None,          help="Output dataset folder (default: projects/{name}/dataset)")
    parser.add_argument("--val-ratio", default=0.2, type=float, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed",      default=42,  type=int,   help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.project:
        project_dir = os.path.join("projects", args.project)
        args.input  = args.input  or os.path.join(project_dir, "dataset_raw")
        args.output = args.output or os.path.join(project_dir, "dataset")
    else:
        args.input  = args.input  or "dataset_raw"
        args.output = args.output or "dataset"

    return args


def read_classes(input_dir):
    path = os.path.join(input_dir, "classes.txt")
    if not os.path.exists(path):
        print("[WARN] classes.txt not found, fill in data.yaml names manually")
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def split_dataset(input_dir, output_dir, train_ratio, seed):
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    all_images = [f for f in os.listdir(images_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.seed(seed)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    splits = {
        "train": all_images[:split_idx],
        "val":   all_images[split_idx:]
    }

    for split, files in splits.items():
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

        for img_file in files:
            stem = os.path.splitext(img_file)[0]
            label_file = f"{stem}.txt"

            shutil.copy(f"{images_dir}/{img_file}", f"{output_dir}/images/{split}/{img_file}")

            label_src = f"{labels_dir}/{label_file}"
            if os.path.exists(label_src):
                shutil.copy(label_src, f"{output_dir}/labels/{split}/{label_file}")
            else:
                # No annotation = background image, create empty txt
                open(f"{output_dir}/labels/{split}/{label_file}", "w").close()

        print(f"  {split}: {len(files)} images")


def write_yaml(output_dir, classes):
    nc = len(classes)
    names_str = str(classes) if classes else "['class0']  # fill in your class names"

    yaml_content = f"""# YOLOv8 Dataset Config
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

nc: {nc}
names: {names_str}
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"\ndata.yaml written: {yaml_path}")
    if not classes:
        print("[!] Edit data.yaml and fill in the correct names")


if __name__ == "__main__":
    args = parse_args()
    input_dir   = args.input
    output_dir  = args.output
    train_ratio = 1.0 - args.val_ratio
    seed        = args.seed

    if not os.path.exists(input_dir):
        print(f"[ERROR] {input_dir} not found, run extract_frames.py first")
        exit(1)

    classes = read_classes(input_dir)
    print(f"Classes: {classes}")
    print(f"Splitting dataset (train {train_ratio*100:.0f}% / val {args.val_ratio*100:.0f}%)...")

    split_dataset(input_dir, output_dir, train_ratio, seed)
    write_yaml(output_dir, classes)

    print(f"\nDone! Dataset saved to {output_dir}/")
