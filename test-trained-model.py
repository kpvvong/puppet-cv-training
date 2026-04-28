# 需要安裝：pip install ultralytics opencv-python
# 若需用裝置名稱指定 webcam（Windows 限定）：pip install pygrabber

"""即時推論 + tracking，來源可為 webcam 或影片檔。

Usage:
    python util/test-trained-model.py --weights runs/detect/puppet/weights/best.pt
    python util/test-trained-model.py --weights best.pt --source video.mp4
    python util/test-trained-model.py --weights best.pt --source 0                  # webcam（預設第一個）
    python util/test-trained-model.py --weights best.pt --source 1                  # webcam（第二個裝置）
    python util/test-trained-model.py --weights best.pt --source "C270 HD WEBCAM"   # 用裝置名稱指定 (需安裝pygrabber)
    python util/test-trained-model.py --weights best.pt --source video.mp4 --save
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Puppet tracking")
    parser.add_argument(
        "--weights",
        required=True,
        help="模型權重路徑，例如 runs/detect/puppet/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="來源：webcam index (0) 或影片路徑",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="信心閾值 (0~1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推論影像尺寸",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="是否儲存輸出影片至 runs/track/",
    )
    parser.add_argument(
        "--device",
        default="",
        help="裝置：'' 自動, 'cpu', '0'",
    )
    return parser.parse_args()


def list_camera_devices() -> list[str]:
    """回傳 DirectShow 影像裝置名稱列表，index 與 VideoCapture(i) 一致。"""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except ImportError:
        raise RuntimeError("請先安裝：pip install pygrabber")


def resolve_camera_index(name: str) -> int:
    """回傳名稱包含 name（不分大小寫）的第一個 DirectShow 裝置 index。"""
    import re
    devices = list_camera_devices()
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    for i, dev in enumerate(devices):
        if pattern.search(dev):
            return i
    device_list = "\n".join(f"  [{i}] {d}" for i, d in enumerate(devices))
    raise RuntimeError(
        f"找不到名稱含 '{name}' 的 webcam。\n可用裝置：\n{device_list}"
    )


def open_source(source: str) -> cv2.VideoCapture:
    """將 source 字串轉為 VideoCapture。
    - 純數字 → webcam index
    - 存在的路徑 → 影片檔
    - 其他字串 → 視為裝置名稱，自動掃描對應 index
    """
    if source.isdigit():
        idx_or_path: int | str = int(source)
    elif Path(source).exists():
        idx_or_path = source
    else:
        idx_or_path = resolve_camera_index(source)
        print(f"裝置 '{source}' → camera index {idx_or_path}")
    cap = cv2.VideoCapture(idx_or_path, cv2.CAP_DSHOW if isinstance(idx_or_path, int) else 0)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟來源：{source}")
    return cap


def make_writer(cap: cv2.VideoCapture, out_path: Path) -> cv2.VideoWriter:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))


def draw_tracks(frame, result) -> None:
    """把 tracking 結果畫到 frame 上（in-place）。"""
    if result.boxes is None:
        return
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        track_id = int(box.id[0]) if box.id is not None else -1
        label = f"Puppet id={track_id} {conf:.2f}" if track_id >= 0 else f"Puppet {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"找不到權重：{weights}\n請先執行 python util/train.py 完成訓練")

    print(f"載入模型：{weights}")
    model = YOLO(str(weights))

    cap = open_source(args.source)
    source_label = f"webcam({args.source})" if args.source.isdigit() else args.source

    writer = None
    if args.save:
        out_path = Path("runs/track") / (Path(args.source).stem if not args.source.isdigit() else "webcam")
        out_path = out_path.with_suffix(".mp4")
        writer = make_writer(cap, out_path)
        print(f"輸出儲存至：{out_path}")

    print(f"開始追蹤，來源：{source_label}  按 q 離開")
    cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

    t0 = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            persist=True,   # 跨幀保持 tracker 狀態
            verbose=False,
        )

        draw_tracks(frame, results[0])

        frame_count += 1
        elapsed = time.time() - t0
        fps_display = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow("YOLOv8 Tracking", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"結束。共處理 {frame_count} 幀，平均 {frame_count / (time.time() - t0):.1f} FPS")


if __name__ == "__main__":
    main()
