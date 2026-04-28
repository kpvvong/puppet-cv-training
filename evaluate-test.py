"""
Evaluate trained YOLO model on test (or val) split
---------------------------------------------------
Run train.py first to produce models/{name}/weights/best.pt.

Usage:
  python evaluate-test.py --weights <path/to/best.pt> [--data <yaml>] [--split test|val]
  python evaluate-test.py --name <model_name> [--data <yaml>] [--split test|val]

Examples:
  python evaluate-test.py --weights models/puppet-v1/weights/best.pt
  python evaluate-test.py --name puppet-v1
  python evaluate-test.py --name puppet-v1 --data projects/20260421_single-puppet-test/dataset/data.yaml
  python evaluate-test.py --weights models/puppet-v1/weights/best.pt --split val   # sanity check vs training

Output: prints mAP50, mAP50-95, precision, recall.
        Full report (PR curve, confusion matrix) under runs/detect/val*/
"""

import argparse
import glob
import os
import sys

import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on a dataset split")
    parser.add_argument("--weights", help="Path to YOLO weights file (overrides --name)")
    parser.add_argument("--name",    help="Model name under models/{name}/")
    parser.add_argument("--data",    default="dataset/data.yaml",   help="data.yaml path (default: dataset/data.yaml)")
    parser.add_argument("--split",   default="test", choices=["test", "val"], help="Which split to evaluate (default: test)")
    parser.add_argument("--device",  default="cuda",                help="Device (default: cuda)")
    parser.add_argument("--batch",   default=16, type=int,          help="Batch size (default: 16)")
    return parser.parse_args()


def resolve_weights(name=None, weights=None):
    if weights:
        if os.path.exists(weights):
            return weights
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    if name:
        weights = os.path.join("models", name, "weights", "best.pt")
        if os.path.exists(weights):
            return weights

        print(f"[ERROR] Weights not found: {weights}")
        available = sorted(glob.glob(os.path.join("models", "*", "weights", "best.pt")))
        if available:
            print("Available models:")
            for w in available:
                print(f"  - {w}")
        else:
            print("No trained models found under models/*/weights/best.pt")
        sys.exit(1)

    print("[ERROR] Please specify either --weights or --name")
    sys.exit(1)


def check_split_in_yaml(data_yaml, split):
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        sys.exit(1)
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if split not in cfg or not cfg[split]:
        print(f"[ERROR] '{split}:' is missing or empty in {data_yaml}")
        print(f"        Add a line like: {split}: images/{split}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()

    weights = resolve_weights(name=args.name, weights=args.weights)
    check_split_in_yaml(args.data, args.split)

    print(f"Evaluating {weights} on split='{args.split}'...")
    model = YOLO(weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        batch=args.batch,
    )

    box = metrics.box
    print("\n=== Results ===")
    print(f"  mAP50      : {box.map50:.4f}")
    print(f"  mAP50-95   : {box.map:.4f}")
    print(f"  precision  : {box.mp:.4f}")
    print(f"  recall     : {box.mr:.4f}")
    print(f"\nFull report: {metrics.save_dir}")
