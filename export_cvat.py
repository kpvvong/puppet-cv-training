"""
Export dataset_raw → CVAT for images 1.1 XML zip for human review.

CVAT import steps:
  1. Create a new Task in CVAT and upload the images from dataset_raw/images/
  2. Actions → Upload annotations → CVAT 1.1 → select the exported zip

Usage:
  python export_cvat.py --project <project_name>
  python export_cvat.py --input dataset_raw --output cvat_export.zip
"""

import os
import glob as glob_module
import argparse
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Export dataset_raw to CVAT 1.1 XML zip")
    parser.add_argument("--project", default=None, help="Project name (uses projects/{name}/dataset_raw)")
    parser.add_argument("--input",   default=None, help="dataset_raw folder path (manual)")
    parser.add_argument("--output",  default=None, help="Output zip path (default: projects/{name}/cvat_export.zip)")
    args = parser.parse_args()

    if args.project:
        project_dir  = os.path.join("projects", args.project)
        args.input   = args.input  or os.path.join(project_dir, "dataset_raw")
        args.output  = args.output or os.path.join(project_dir, "cvat_export.zip")
    elif args.input:
        args.output  = args.output or os.path.join(os.path.dirname(args.input), "cvat_export.zip")
    else:
        parser.error("Provide --project or --input")

    return args


def load_boxes_normalized(label_path):
    """Return list of (class_id, cx, cy, bw, bh) normalized."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append((int(parts[0]),
                          float(parts[1]), float(parts[2]),
                          float(parts[3]), float(parts[4])))
    return boxes


def build_xml(image_files, labels_dir, classes):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    labels_el = ET.SubElement(task, "labels")
    for name in classes:
        label_el = ET.SubElement(labels_el, "label")
        ET.SubElement(label_el, "name").text = name
        ET.SubElement(label_el, "attributes")

    for img_id, img_path in enumerate(image_files):
        try:
            img_w, img_h = Image.open(img_path).size
        except Exception:
            print(f"  [WARN] Cannot read {img_path}, skipping")
            continue
        filename = os.path.basename(img_path)
        stem     = os.path.splitext(filename)[0]

        img_el = ET.SubElement(root, "image",
                               id=str(img_id),
                               name=filename,
                               width=str(img_w),
                               height=str(img_h))

        label_path = os.path.join(labels_dir, f"{stem}.txt")
        for class_id, cx, cy, bw, bh in load_boxes_normalized(label_path):
            xtl = (cx - bw / 2) * img_w
            ytl = (cy - bh / 2) * img_h
            xbr = (cx + bw / 2) * img_w
            ybr = (cy + bh / 2) * img_h
            label_name = classes[class_id] if class_id < len(classes) else str(class_id)
            ET.SubElement(img_el, "box",
                          label=label_name,
                          occluded="0",
                          xtl=f"{xtl:.2f}",
                          ytl=f"{ytl:.2f}",
                          xbr=f"{xbr:.2f}",
                          ybr=f"{ybr:.2f}")

    return ET.tostring(root, encoding="unicode", xml_declaration=False)


def main():
    args = parse_args()
    input_dir    = args.input
    images_dir   = os.path.join(input_dir, "images")
    labels_dir   = os.path.join(input_dir, "labels")
    classes_path = os.path.join(input_dir, "classes.txt")

    if not os.path.isdir(images_dir):
        print(f"[ERROR] images/ not found in {input_dir}")
        return

    image_files = sorted(
        glob_module.glob(os.path.join(images_dir, "*.jpg")) +
        glob_module.glob(os.path.join(images_dir, "*.png"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    classes = []
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            classes = [line.strip() for line in f if line.strip()]

    out_path = args.output
    print(f"Building annotations for {len(image_files)} images...")
    xml_str = '<?xml version="1.0" encoding="utf-8"?>\n' + build_xml(image_files, labels_dir, classes)

    print(f"Writing → {out_path}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.xml", xml_str)

    print(f"Done: {out_path}")
    print()
    print("CVAT import steps:")
    print("  1. Create a Task in CVAT and upload the images from dataset_raw/images/")
    print("  2. Actions → Upload annotations → CVAT 1.1 → select the zip above")


if __name__ == "__main__":
    main()
