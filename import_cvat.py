"""
Convert CVAT YOLO 1.1 export zip → dataset_raw format for prepare_dataset.py

Usage:
  python import_cvat.py --project <project_name> --zip <cvat_export.zip>
  python import_cvat.py --output dataset_raw --zip cvat_export.zip

Output:
  dataset_raw/
  ├── images/
  ├── labels/
  └── classes.txt
"""

import os
import argparse
import zipfile
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Import CVAT YOLO 1.1 zip into dataset_raw format")
    parser.add_argument("--zip",     required=True,  help="CVAT YOLO 1.1 export zip file")
    parser.add_argument("--project", default=None,   help="Project name (output to projects/{name}/dataset_raw)")
    parser.add_argument("--output",  default=None,   help="Output dataset_raw folder (manual)")
    args = parser.parse_args()

    if args.project:
        args.output = args.output or os.path.join("projects", args.project, "dataset_raw")
    elif not args.output:
        parser.error("Provide --project or --output")

    return args


def main():
    args = parse_args()
    output_dir = args.output
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    img_count   = 0
    label_count = 0

    with zipfile.ZipFile(args.zip) as zf:
        names = zf.namelist()

        # obj.names → classes.txt
        names_files = [n for n in names if n.endswith("obj.names")]
        if names_files:
            with zf.open(names_files[0]) as f:
                content = f.read().decode("utf-8")
            with open(os.path.join(output_dir, "classes.txt"), "w") as f:
                f.write(content)
            classes = [l.strip() for l in content.splitlines() if l.strip()]
            print(f"Classes: {classes}")
        else:
            print("[WARN] obj.names not found in zip")

        # obj_train_data/*.jpg / *.png → images/
        # obj_train_data/*.txt         → labels/
        for name in names:
            basename = os.path.basename(name)
            if not basename:
                continue

            if basename.lower().endswith((".jpg", ".jpeg", ".png")):
                dest = os.path.join(images_dir, basename)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                img_count += 1

            elif basename.lower().endswith(".txt") and basename not in ("train.txt", "valid.txt", "test.txt", "obj.names"):
                dest = os.path.join(labels_dir, basename)
                with zf.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                label_count += 1

    print(f"Imported {img_count} images, {label_count} labels → {output_dir}/")
    print(f"\nNext: python prepare_dataset.py --input {output_dir}")


if __name__ == "__main__":
    main()
