"""
Train YOLOv8n model
--------------------------
Run prepare_dataset.py first.

Usage:
  python train.py --name <model_name> [--data <yaml>] [--epochs 100] [--batch 16]

Examples:
  python train.py --name puppet-v1
  python train.py --name puppet-v2 --data projects/20260421_single-puppet-test/dataset/data.yaml --epochs 150

Output: models/{name}/weights/
  - weights/best.pt   ← use this for webcam detection
  - weights/last.pt
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n model")
    parser.add_argument("--name",   required=True,                help="Model name, output to models/{name}/")
    parser.add_argument("--data",   default="dataset/data.yaml",  help="data.yaml path (default: dataset/data.yaml)")
    parser.add_argument("--model",  default="yolov8n.pt",         help="Pretrained weights (default: yolov8n.pt)")
    parser.add_argument("--epochs", default=100, type=int,        help="Number of epochs (default: 100)")
    parser.add_argument("--batch",  default=16,  type=int,        help="Batch size (default: 16, RTX 3070 supports 16~32)")
    parser.add_argument("--device", default="cuda",               help="Device (default: cuda, use cpu for CPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        project="models",
        name=args.name,
        # Data augmentation (recommended for webcam scenarios)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        degrees=15,      # 開啟旋轉，偶有時會傾斜
        mosaic=1.0,      # 保持開啟，對多偶同框很有幫助
        scale=0.6,       # 加大縮放範圍，應對遠近距離變化
    )

    print(f"\nTraining complete! Best model: {results.save_dir}/weights/best.pt")
