"""
SAM2 video auto-annotation → YOLO format
----------------------------------------------
No Label Studio needed. Click on objects in the first frame,
SAM2 automatically tracks them through the entire video.
Supports multiple videos per project; frame numbers are
incremented continuously across videos.

Usage:
  python sam2_annotate.py --project <project_name> [--classes Puppet,Hand]

Controls (interactive window):
  Left click  → add foreground point (tell SAM2 "this is the object")
  Right click → add background point (tell SAM2 "this is NOT the object")
  n           → next object (use different IDs for multiple puppets)
  u           → undo last point
  Enter/Space → confirm and start tracking
  ESC         → skip this video

Output (same format as extract_frames.py):
  projects/{name}/dataset_raw/
  ├── images/      ← frame images (.jpg)
  ├── labels/      ← YOLO annotations (.txt)
  └── classes.txt
"""

import os
import cv2
import numpy as np
import torch
import glob as glob_module
import argparse
import shutil


def find_project_videos(project_name):
    project_dir = os.path.join("projects", project_name)
    if not os.path.isdir(project_dir):
        print(f"[ERROR] Project folder not found: {project_dir}")
        return None, None

    video_files = sorted(glob_module.glob(os.path.join(project_dir, "**", "*.mp4"), recursive=True))
    if not video_files:
        print(f"[ERROR] No .mp4 files found in {project_dir}")
        return None, None

    output_dir = os.path.join(project_dir, "dataset_raw")
    return video_files, output_dir


def extract_all_frames(video_path, frames_dir):
    """Extract all frames to a temp directory (SAM2 requires a JPEG sequence)."""
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0, 0, 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Extracting frames ({total} total)...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
        if idx % 100 == 0:
            print(f"    {idx}/{total}...")
    cap.release()
    return idx, w, h


class PointSelector:
    """Interactive frame annotation for SAM2.

    Point mode (default):
      Left click  → foreground point
      Right click → background point

    Box mode (press b):
      Left drag   → foreground box  (only one allowed per object)
      Right drag  → exclusion box   (converted to background point grid for SAM2)

    Navigate frames with d/a or arrow keys before placing any annotation.
    """

    COLORS     = [(0, 255, 80), (0, 120, 255), (255, 60, 120), (255, 220, 0), (0, 220, 255)]
    BG_COLOR   = (0, 0, 220)   # red-ish tint for exclusion boxes
    EXCL_GRID  = 4             # NxN background points sampled inside each exclusion box

    def __init__(self, frames_dir, total_frames, video_label, win_w=1280, win_h=720):
        self.frames_dir     = frames_dir
        self.total_frames   = total_frames
        self.frame_idx      = 0
        self.frame          = self._load_frame(0)
        # obj_id -> {"mode": "point"|"box", "points": [(x,y,lbl),...],
        #            "box": (x1,y1,x2,y2)|None, "excl_boxes": [(x1,y1,x2,y2),...]}
        self.objects        = {}
        self.current_obj_id = 1
        self.box_mode       = False
        self._drag_start    = None  # (x, y, button)  button: 1=left, 3=right
        self._drag_cur      = None
        self.win_w          = win_w
        self.win_h          = win_h
        self.window = (
            f"[{video_label}]  "
            "d/a=prev/next frame  b=box/point  n=new obj  u=undo  Enter=track  ESC=skip  |  "
            "POINT: LClick=fg  RClick=bg  |  "
            "BOX: LDrag=fg box  RDrag=exclusion box"
        )

    def _load_frame(self, idx):
        return cv2.imread(os.path.join(self.frames_dir, f"{idx:06d}.jpg"))

    def _color(self, obj_id):
        return self.COLORS[(obj_id - 1) % len(self.COLORS)]

    def _current(self):
        obj = self.objects.setdefault(
            self.current_obj_id,
            {"mode": "box" if self.box_mode else "point",
             "points": [], "box": None, "excl_boxes": []},
        )
        obj["mode"] = "box" if self.box_mode else "point"
        return obj

    def _on_mouse(self, event, x, y, flags, param):
        if not self.box_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._current()["points"].append((x, y, 1))
                self._redraw()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._current()["points"].append((x, y, 0))
                self._redraw()

        else:  # box mode
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                btn = 1 if event == cv2.EVENT_LBUTTONDOWN else 3
                self._drag_start = (x, y, btn)
                self._drag_cur   = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
                self._drag_cur = (x, y)
                self._redraw()
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP) and self._drag_start:
                sx, sy, btn = self._drag_start
                x1, y1 = min(sx, x), min(sy, y)
                x2, y2 = max(sx, x), max(sy, y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    obj = self._current()
                    if btn == 1:
                        obj["box"] = (x1, y1, x2, y2)   # foreground box
                    else:
                        obj["excl_boxes"].append((x1, y1, x2, y2))  # exclusion box
                self._drag_start = None
                self._drag_cur   = None
                self._redraw()

    def _redraw(self):
        display = self.frame.copy()
        for obj_id, obj in self.objects.items():
            color = self._color(obj_id)
            if obj["mode"] == "point" and not obj["box"]:
                for x, y, lbl in obj["points"]:
                    cv2.circle(display, (x, y), 8, color, -1 if lbl == 1 else 2)
                    cv2.circle(display, (x, y), 10, (255, 255, 255), 1)
                    cv2.putText(display, str(obj_id), (x + 12, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            else:
                if obj["box"]:
                    x1, y1, x2, y2 = obj["box"]
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, str(obj_id), (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                for x1, y1, x2, y2 in obj["excl_boxes"]:
                    cv2.rectangle(display, (x1, y1), (x2, y2), self.BG_COLOR, 2)
                    cv2.putText(display, "excl", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.BG_COLOR, 1)

        # Live drag preview
        if self._drag_start and self._drag_cur:
            sx, sy, btn = self._drag_start
            ex, ey = self._drag_cur
            drag_color = self._color(self.current_obj_id) if btn == 1 else self.BG_COLOR
            cv2.rectangle(display, (min(sx, ex), min(sy, ey)),
                          (max(sx, ex), max(sy, ey)), drag_color, 1)

        cur_color  = self._color(self.current_obj_id)
        mode_label = "BOX" if self.box_mode else "POINT"
        cv2.putText(display, f"Object {self.current_obj_id}  [{mode_label}]", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cur_color, 2)
        cv2.putText(display, f"Frame {self.frame_idx}/{self.total_frames - 1}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow(self.window, display)

    def excl_boxes_to_bg_points(self, excl_boxes):
        """Sample a grid of background points inside each exclusion box."""
        pts = []
        n = self.EXCL_GRID
        for x1, y1, x2, y2 in excl_boxes:
            for row in range(n):
                for col in range(n):
                    px = x1 + (x2 - x1) * (col + 1) / (n + 1)
                    py = y1 + (y2 - y1) * (row + 1) / (n + 1)
                    pts.append((int(px), int(py), 0))
        return pts

    def _has_annotation(self):
        for obj in self.objects.values():
            if obj["mode"] == "point" and obj["points"]:
                return True
            if obj["mode"] == "box" and (obj["box"] or obj["excl_boxes"]):
                return True
        return False

    def _go_to(self, idx):
        if self.objects:
            print("  [WARN] Cannot change frame after placing annotations. Press u to undo first.")
            return
        self.frame_idx = max(0, min(idx, self.total_frames - 1))
        self.frame = self._load_frame(self.frame_idx)
        self._redraw()

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # Enter / Space
                if not self._has_annotation():
                    print("  [WARN] Add at least one point or box first")
                    continue
                break
            elif key in (ord('d'), 83):  # d or Right arrow
                self._go_to(self.frame_idx + 1)
            elif key in (ord('a'), 81):  # a or Left arrow
                self._go_to(self.frame_idx - 1)
            elif key == ord('b'):
                self.box_mode = not self.box_mode
                mode = "BOX" if self.box_mode else "POINT"
                print(f"  Mode: {mode}")
                self._redraw()
            elif key == ord('n'):
                self.current_obj_id += 1
                print(f"  Switched to object {self.current_obj_id}")
                self._redraw()
            elif key == ord('u'):
                obj = self.objects.get(self.current_obj_id)
                if obj:
                    if obj["mode"] == "point" and obj["points"]:
                        obj["points"].pop()
                    elif obj["mode"] == "box":
                        if obj["excl_boxes"]:
                            obj["excl_boxes"].pop()    # undo last exclusion box first
                        elif obj["box"]:
                            obj["box"] = None
                    if not obj["points"] and not obj["box"] and not obj["excl_boxes"]:
                        del self.objects[self.current_obj_id]
                    self._redraw()
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyAllWindows()
        return self.objects, self.frame_idx


def mask_to_yolo_bbox(mask, frame_h, frame_w):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cx = (x1 + x2) / 2 / frame_w
    cy = (y1 + y2) / 2 / frame_h
    w  = (x2 - x1) / frame_w
    h  = (y2 - y1) / frame_h
    return cx, cy, w, h


def draw_bboxes(frame, lines, classes, frame_w, frame_h):
    """Draw YOLO bounding boxes onto a frame for preview."""
    COLORS = [(0, 255, 80), (0, 120, 255), (255, 60, 120), (255, 220, 0), (0, 220, 255)]
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - w / 2) * frame_w)
        y1 = int((cy - h / 2) * frame_h)
        x2 = int((cx + w / 2) * frame_w)
        y2 = int((cy + h / 2) * frame_h)
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = classes[class_id] if class_id < len(classes) else str(class_id)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 video auto-annotation → YOLO format")
    parser.add_argument("--project", required=True,                        help="Project name (projects/{name}/)")
    parser.add_argument("--classes", default=None,                         help="Class names, comma-separated, in object ID order (e.g. Puppet,Hand)")
    parser.add_argument("--model",   default="sam2_hiera_large.pt",        help="SAM2 checkpoint filename (place in models/)")
    parser.add_argument("--config",  default="configs/sam2/sam2_hiera_l.yaml", help="SAM2 config path (relative to sam2 package)")
    parser.add_argument("--device",  default="cuda",                       help="Device (default: cuda)")
    parser.add_argument("--preview", action="store_true",                  help="Show live preview window during tracking (ESC to abort)")
    parser.add_argument("--window-size", default="1280x720",               help="Initial window size WxH (default: 1280x720)")
    return parser.parse_args()


def main():
    args = parse_args()

    from sam2.build_sam import build_sam2_video_predictor

    try:
        win_w, win_h = [int(v) for v in args.window_size.lower().split("x")]
    except ValueError:
        print(f"[ERROR] Invalid --window-size format '{args.window_size}', expected WxH (e.g. 1280x720)")
        return

    video_files, output_dir = find_project_videos(args.project)
    if not video_files:
        return

    print(f"Found {len(video_files)} video(s):")
    for v in video_files:
        print(f"  {v}")

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    classes = args.classes.split(",") if args.classes else []

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SAM2 ({device})...")
    predictor = build_sam2_video_predictor(args.config, f"models/{args.model}", device=device)

    global_frame_count = 0

    for video_idx, video_path in enumerate(video_files):
        video_label = os.path.basename(video_path)
        print(f"\n[{video_idx + 1}/{len(video_files)}] {video_label}")

        frames_dir = os.path.join(output_dir, f"_sam2_frames_{video_idx}")
        total_frames, frame_w, frame_h = extract_all_frames(video_path, frames_dir)
        if total_frames == 0:
            shutil.rmtree(frames_dir, ignore_errors=True)
            continue

        print("  Navigate to a frame with the object visible, then click to annotate (ESC to skip)")
        selector = PointSelector(frames_dir, total_frames, video_label, win_w, win_h)
        selected, anchor_frame = selector.run()

        if selected is None:
            print(f"  Skipped {video_label}")
            shutil.rmtree(frames_dir, ignore_errors=True)
            continue

        # Fill in missing class names
        max_obj_id = max(selected.keys())
        while len(classes) < max_obj_id:
            classes.append(f"object_{len(classes) + 1}")

        saved = 0
        with torch.inference_mode():
            state = predictor.init_state(video_path=frames_dir)

            for obj_id, obj in selected.items():
                kwargs = dict(inference_state=state, frame_idx=anchor_frame, obj_id=obj_id)
                if obj["mode"] == "box" and obj["box"]:
                    x1, y1, x2, y2 = obj["box"]
                    kwargs["box"] = np.array([x1, y1, x2, y2], dtype=np.float32)
                    # Merge explicit points + background points sampled from exclusion boxes
                    all_pts = list(obj["points"]) + selector.excl_boxes_to_bg_points(obj["excl_boxes"])
                    if all_pts:
                        kwargs["points"] = np.array([[x, y] for x, y, _ in all_pts], dtype=np.float32)
                        kwargs["labels"] = np.array([lbl for _, _, lbl in all_pts], dtype=np.int32)
                else:
                    kwargs["points"] = np.array([[x, y] for x, y, _ in obj["points"]], dtype=np.float32)
                    kwargs["labels"] = np.array([lbl for _, _, lbl in obj["points"]], dtype=np.int32)
                predictor.add_new_points_or_box(**kwargs)

            print(f"  Tracking ({total_frames} frames)..." + (" Press ESC to abort." if args.preview else ""))
            aborted = False
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                lines = []
                for i, obj_id in enumerate(obj_ids):
                    mask = masks[i][0].cpu().numpy() > 0.0
                    bbox = mask_to_yolo_bbox(mask, frame_h, frame_w)
                    if bbox is None:
                        continue
                    class_id = obj_id - 1  # obj_id starts at 1, class_id at 0
                    cx, cy, w, h = bbox
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if not lines:
                    continue

                if args.preview:
                    preview_win = f"Preview: {video_label}"
                    preview = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
                    draw_bboxes(preview, lines, classes, frame_w, frame_h)
                    cv2.putText(preview, f"frame {frame_idx}  ESC=abort", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(preview_win, win_w, win_h)
                    cv2.imshow(preview_win, preview)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        print("  Tracking aborted by user")
                        aborted = True
                        break

                name = f"frame_{global_frame_count:06d}"
                shutil.copy(
                    os.path.join(frames_dir, f"{frame_idx:06d}.jpg"),
                    os.path.join(images_dir, f"{name}.jpg"),
                )
                with open(os.path.join(labels_dir, f"{name}.txt"), "w") as f:
                    f.write("\n".join(lines))

                global_frame_count += 1
                saved += 1

            if args.preview:
                cv2.destroyAllWindows()

        shutil.rmtree(frames_dir)
        print(f"  {'Aborted,' if aborted else 'Done,'} annotated {saved} frames")

    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

    print(f"\nAll done! {global_frame_count} frames total")
    print(f"Classes: {classes}")
    print(f"Output:  {output_dir}/")
    print(f"\nNext: python prepare_dataset.py --project {args.project}")


if __name__ == "__main__":
    main()
