"""
Webcam real-time detection
------------------------
Features:
  - YOLOv8n real-time inference
  - OSC output: bounding box coordinates + average object depth
  - Mask Texture window (black background, white boxes)
  - Depth Map window (Depth Anything V2, optionally masked to detected regions)

OSC message format (sent every frame):
  /yolo/count              i     → number of detected objects
  /yolo/clear                    → clear signal (sent at start of each frame)
  /yolo/{i}/label          s     → class name
  /yolo/{i}/conf           f     → confidence (0~1)
  /yolo/{i}/bbox           ffff  → cx cy w h (normalized 0~1)
  /yolo/{i}/bbox_px        iiii  → x1 y1 x2 y2 (pixel coordinates)
  /yolo/{i}/depth          f     → average depth in object region (0=near ~ 1=far)

Controls:
  q     → quit
  s     → screenshot
  m     → toggle Mask window
  d     → toggle Depth Map window
  o     → toggle OSC output
  +/-   → adjust confidence threshold
"""

import cv2
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from pythonosc import udp_client
from transformers import pipeline

# ═══════════════════════════════════════════════════
#  Settings
# ═══════════════════════════════════════════════════

MODEL_PATH     = "runs/train/exp/weights/best.pt"
CAMERA_ID      = 0
CONF           = 0.3
IMG_SIZE       = 640
DEVICE         = "cuda"
SCREENSHOT_DIR = "screenshots"

# OSC settings
OSC_ENABLE  = True
OSC_IP      = "127.0.0.1"
OSC_PORT    = 8000

# Mask Texture settings
MASK_ENABLE = True
MASK_FILL   = True   # True = filled, False = outline only
MASK_BLUR   = 0      # blur radius (0 = no blur, try 15~31)

# Depth Map settings
DEPTH_ENABLE        = True
DEPTH_MODEL         = "depth-anything/Depth-Anything-V2-Small-hf"  # Small=fast, Base/Large=accurate
DEPTH_MASK_TO_BOXES = True   # True = show depth only in detected regions, False = full frame
DEPTH_COLORMAP      = cv2.COLORMAP_INFERNO  # INFERNO / MAGMA / JET / TURBO

# ═══════════════════════════════════════════════════


def load_depth_model(device):
    print(f"Loading Depth Anything V2... ({DEPTH_MODEL})")
    pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=0 if device == "cuda" and torch.cuda.is_available() else -1,
    )
    print("Depth model loaded")
    return pipe


def estimate_depth(pipe, frame):
    """Returns normalized depth map (float32, 0~1, larger = farther)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    output  = pipe(pil_img)
    depth   = np.array(output["depth"], dtype=np.float32)

    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    return depth


def build_depth_visual(depth_norm, frame_shape, boxes=None, mask_to_boxes=True, colormap=cv2.COLORMAP_INFERNO):
    """
    depth_norm : float32 ndarray 0~1 (original size)
    boxes      : ultralytics Boxes (can be None)
    Returns BGR colorized depth map, same size as frame.
    """
    h, w = frame_shape[:2]
    depth_resized = cv2.resize(depth_norm, (w, h))

    depth_u8  = (depth_resized * 255).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(depth_u8, colormap)

    if mask_to_boxes and boxes is not None and len(boxes) > 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        depth_rgb = cv2.bitwise_and(depth_rgb, depth_rgb, mask=mask)

    return depth_rgb


def build_mask(frame_shape, boxes, fill=True, blur=0):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box[:4]
            if fill:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            else:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, 2)

    if blur > 0:
        blur = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def get_box_depth(depth_norm, frame_shape, box_xyxy):
    """Returns average depth (0~1) within a bounding box region."""
    h, w = frame_shape[:2]
    depth_resized = cv2.resize(depth_norm, (w, h))
    x1, y1, x2, y2 = [int(v) for v in box_xyxy[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    region = depth_resized[y1:y2, x1:x2]
    return float(region.mean()) if region.size > 0 else 0.0


def send_osc(client, results, depth_norm=None):
    boxes = results[0].boxes
    names = results[0].names
    count = len(boxes) if boxes is not None else 0

    client.send_message("/yolo/clear", [])
    client.send_message("/yolo/count", count)

    if count == 0:
        return

    for i, box in enumerate(boxes):
        label = names[int(box.cls[0])]
        conf  = float(box.conf[0])
        cx, cy, bw, bh = box.xywhn[0].cpu().numpy()
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        prefix = f"/yolo/{i}"
        client.send_message(f"{prefix}/label",   label)
        client.send_message(f"{prefix}/conf",    float(conf))
        client.send_message(f"{prefix}/bbox",    [float(cx), float(cy), float(bw), float(bh)])
        client.send_message(f"{prefix}/bbox_px", [int(x1), int(y1), int(x2), int(y2)])

        if depth_norm is not None:
            avg_depth = get_box_depth(depth_norm, results[0].orig_shape, box.xyxy[0].cpu().numpy())
            client.send_message(f"{prefix}/depth", avg_depth)


def draw_ui(frame, fps, conf, osc_on, mask_on, depth_on):
    h = frame.shape[0]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Conf: {conf:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    def status_color(on): return (0, 255, 100) if on else (80, 80, 80)
    cv2.putText(frame, f"OSC   {'ON' if osc_on   else 'OFF'}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color(osc_on), 2)
    cv2.putText(frame, f"Mask  {'ON' if mask_on  else 'OFF'}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color(mask_on), 2)
    cv2.putText(frame, f"Depth {'ON' if depth_on else 'OFF'}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color(depth_on), 2)
    cv2.putText(frame, "q:quit  s:screenshot  o:OSC  m:Mask  d:Depth  +/-:threshold",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def main():
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    Path(SCREENSHOT_DIR).mkdir(exist_ok=True)

    yolo = YOLO(MODEL_PATH)
    yolo.to(DEVICE)

    depth_pipe = load_depth_model(DEVICE) if DEPTH_ENABLE else None

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (ID={CAMERA_ID})")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    osc_client = None
    if OSC_ENABLE:
        try:
            osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
            print(f"OSC → {OSC_IP}:{OSC_PORT}")
        except Exception as e:
            print(f"[WARN] OSC init failed: {e}")

    conf       = CONF
    osc_on     = OSC_ENABLE and osc_client is not None
    mask_on    = MASK_ENABLE
    depth_on   = DEPTH_ENABLE and depth_pipe is not None
    prev_time  = time.time()
    shot_count = 0

    print("q:quit  s:screenshot  o:OSC  m:Mask  d:Depth  +/-:threshold")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results   = yolo(frame, imgsz=IMG_SIZE, conf=conf, device=DEVICE, verbose=False)
        annotated = results[0].plot()
        boxes     = results[0].boxes

        depth_norm = None
        if depth_on and depth_pipe is not None:
            depth_norm = estimate_depth(depth_pipe, frame)
            depth_vis  = build_depth_visual(
                depth_norm, frame.shape, boxes,
                mask_to_boxes=DEPTH_MASK_TO_BOXES,
                colormap=DEPTH_COLORMAP
            )
            cv2.imshow("Depth Map", depth_vis)

        if mask_on:
            mask = build_mask(frame.shape, boxes, fill=MASK_FILL, blur=MASK_BLUR)
            cv2.imshow("Mask Texture", mask)

        if osc_on and osc_client:
            try:
                send_osc(osc_client, results, depth_norm)
            except Exception as e:
                print(f"[WARN] OSC send failed: {e}")

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        draw_ui(annotated, fps, conf, osc_on, mask_on, depth_on)
        cv2.imshow("YOLOv8n Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            shot_count += 1
            path = f"{SCREENSHOT_DIR}/shot_{shot_count:04d}.jpg"
            cv2.imwrite(path, annotated)
            print(f"Screenshot: {path}")
        elif key == ord("o"):
            osc_on = not osc_on
            print(f"OSC {'ON' if osc_on else 'OFF'}")
        elif key == ord("m"):
            mask_on = not mask_on
            if not mask_on:
                cv2.destroyWindow("Mask Texture")
            print(f"Mask {'ON' if mask_on else 'OFF'}")
        elif key == ord("d"):
            depth_on = not depth_on
            if not depth_on:
                cv2.destroyWindow("Depth Map")
            print(f"Depth {'ON' if depth_on else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
            print(f"Conf: {conf:.2f}")
        elif key == ord("-"):
            conf = max(0.05, conf - 0.05)
            print(f"Conf: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
