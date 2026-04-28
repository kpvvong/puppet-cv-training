"""
Review and edit annotated dataset — select, resize, move, or delete bounding boxes.

Usage:
  python sam2_review.py --project <project_name>
  python sam2_review.py --input dataset_raw

Controls:
  d              → next frame  (auto-saves)
  a              → previous frame  (auto-saves)
  . (period)     → jump to first frame of next video group
  , (comma)      → jump to first frame of previous video group
  Click bbox      → select (shows corner handles)
  Drag corner     → resize selected bbox
  Drag center     → move selected bbox
  Delete          → delete selected bbox
  q / ESC         → quit
"""

import os
import re
import cv2
import argparse
import glob as glob_module


COLORS    = [(0, 255, 80), (0, 120, 255), (255, 60, 120), (255, 220, 0), (0, 220, 255)]
SEL_COLOR = (0, 220, 255)
HANDLE_R  = 8   # corner handle half-size in pixels


def parse_args():
    parser = argparse.ArgumentParser(description="Review and edit annotated dataset")
    parser.add_argument("--project",     default=None,       help="Project name (uses projects/{name}/dataset_raw)")
    parser.add_argument("--input",       default=None,       help="dataset_raw folder path (manual)")
    parser.add_argument("--window-size", default="1280x720", help="Initial window size WxH (default: 1280x720)")
    args = parser.parse_args()

    if args.project:
        args.input = args.input or os.path.join("projects", args.project, "dataset_raw")
    elif not args.input:
        parser.error("Provide --project or --input")

    try:
        args.win_w, args.win_h = [int(v) for v in args.window_size.lower().split("x")]
    except ValueError:
        parser.error(f"Invalid --window-size format '{args.window_size}', expected WxH (e.g. 1280x720)")

    return args


def load_classes(input_dir):
    path = os.path.join(input_dir, "classes.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_boxes(label_path, img_w, img_h):
    """YOLO label → list of [class_id, x1, y1, x2, y2] in pixels."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append([class_id, x1, y1, x2, y2])
    return boxes


def save_boxes(label_path, boxes, img_w, img_h):
    """Save pixel-space boxes back to YOLO normalized format."""
    with open(label_path, "w") as f:
        for class_id, x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _corners(box):
    """Return corner (pixel) positions: TL, TR, BL, BR."""
    _, x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]


def _hit_corner(box, mx, my):
    """Return corner index 0-3 if close enough, else -1."""
    for i, (cx, cy) in enumerate(_corners(box)):
        if abs(mx - cx) <= HANDLE_R and abs(my - cy) <= HANDLE_R:
            return i
    return -1


def _hit_box(box, mx, my):
    _, x1, y1, x2, y2 = box
    return x1 <= mx <= x2 and y1 <= my <= y2


def _vid_key(filename):
    """Extract vid prefix from filename, e.g. 'vid003_frame_000001.jpg' → 'vid003'."""
    m = re.match(r'^(vid\d+)_', os.path.basename(filename))
    return m.group(1) if m else ''


class ReviewUI:
    def __init__(self, image_files, labels_dir, classes, win_w, win_h):
        self.image_files = image_files
        self.labels_dir  = labels_dir
        self.classes     = classes
        self.win_w       = win_w
        self.win_h       = win_h
        self.total       = len(image_files)
        self.idx         = 0

        # Build sorted list of (start_idx, vid_key) for PageUp/PageDown navigation
        self.video_starts = self._build_video_starts()

        # Per-frame state
        self.boxes      = []
        self.img        = None
        self.img_h      = 0
        self.img_w      = 0
        self.label_path = ""
        self.dirty      = False

        # Interaction state
        self.selected       = -1
        self.drag_mode      = None   # None | ("corner", idx) | "move"
        self.drag_start     = None   # (mx, my) at button-down
        self.drag_box_orig  = None   # copy of box at button-down

        self.window = (
            "sam2_review  |  "
            "d/a=next/prev  ./,=next/prev video  "
            "Click=select  Drag corner=resize  Drag center=move  Del=delete  q=quit"
        )

    def _build_video_starts(self):
        """Return list of frame indices where each video group starts."""
        starts = []
        prev_key = None
        for i, f in enumerate(self.image_files):
            k = _vid_key(f)
            if k != prev_key:
                starts.append(i)
                prev_key = k
        return starts

    def _jump_to_video(self, direction):
        """Jump to first frame of next (+1) or previous (-1) video group."""
        # Find which group we're currently in
        cur_group = 0
        for i, start in enumerate(self.video_starts):
            if start <= self.idx:
                cur_group = i
        target_group = cur_group + direction
        target_group = max(0, min(target_group, len(self.video_starts) - 1))
        self._auto_save()
        self.idx = self.video_starts[target_group]
        self._load_frame()
        self._redraw()

    # ── frame loading ──────────────────────────────────────────────────────────

    def _load_frame(self):
        img_path        = self.image_files[self.idx]
        stem            = os.path.splitext(os.path.basename(img_path))[0]
        self.label_path = os.path.join(self.labels_dir, f"{stem}.txt")
        self.img        = cv2.imread(img_path)
        self.img_h, self.img_w = self.img.shape[:2]
        self.boxes      = load_boxes(self.label_path, self.img_w, self.img_h)
        self.selected   = -1
        self.dirty      = False

    def _auto_save(self):
        if self.dirty:
            save_boxes(self.label_path, self.boxes, self.img_w, self.img_h)
            self.dirty = False

    # ── rendering ─────────────────────────────────────────────────────────────

    def _redraw(self):
        display = self.img.copy()

        for i, box in enumerate(self.boxes):
            class_id, x1, y1, x2, y2 = box
            is_sel = (i == self.selected)
            color  = SEL_COLOR if is_sel else COLORS[class_id % len(COLORS)]

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            cv2.putText(display, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if is_sel:
                for cx, cy in _corners(box):
                    cv2.rectangle(display,
                                  (cx - HANDLE_R, cy - HANDLE_R),
                                  (cx + HANDLE_R, cy + HANDLE_R),
                                  SEL_COLOR, -1)

        stem       = os.path.splitext(os.path.basename(self.image_files[self.idx]))[0]
        dirty_mark = "  [unsaved]" if self.dirty else ""
        vid        = _vid_key(self.image_files[self.idx]) or "no-vid"
        grp_idx    = next((i for i, s in enumerate(self.video_starts) if s <= self.idx), 0)
        status     = (f"[{self.idx + 1}/{self.total}]  {stem}  |  "
                      f"video {grp_idx + 1}/{len(self.video_starts)} ({vid})  |  "
                      f"boxes: {len(self.boxes)}{dirty_mark}")
        cv2.putText(display, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.imshow(self.window, display)

    # ── mouse callback ────────────────────────────────────────────────────────

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Priority: corner handle of selected box → inside any box → deselect
            if self.selected >= 0:
                ci = _hit_corner(self.boxes[self.selected], x, y)
                if ci >= 0:
                    self.drag_mode     = ("corner", ci)
                    self.drag_start    = (x, y)
                    self.drag_box_orig = list(self.boxes[self.selected])
                    return

            for i, box in enumerate(self.boxes):
                if _hit_box(box, x, y):
                    self.selected      = i
                    self.drag_mode     = "move"
                    self.drag_start    = (x, y)
                    self.drag_box_orig = list(box)
                    self._redraw()
                    return

            self.selected = -1
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_mode is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            box = self.boxes[self.selected]

            if isinstance(self.drag_mode, tuple):   # corner resize
                ci = self.drag_mode[1]
                _, ox1, oy1, ox2, oy2 = self.drag_box_orig
                mx = max(0, min(x, self.img_w))
                my = max(0, min(y, self.img_h))
                if ci == 0:   ox1, oy1 = mx, my   # TL
                elif ci == 1: ox2, oy1 = mx, my   # TR
                elif ci == 2: ox1, oy2 = mx, my   # BL
                elif ci == 3: ox2, oy2 = mx, my   # BR
                if ox1 > ox2: ox1, ox2 = ox2, ox1
                if oy1 > oy2: oy1, oy2 = oy2, oy1
                box[1], box[2], box[3], box[4] = ox1, oy1, ox2, oy2

            else:   # move
                _, ox1, oy1, ox2, oy2 = self.drag_box_orig
                bw = ox2 - ox1
                bh = oy2 - oy1
                sx, sy  = self.drag_start
                nx1 = max(0, min(ox1 + (x - sx), self.img_w - bw))
                ny1 = max(0, min(oy1 + (y - sy), self.img_h - bh))
                box[1], box[2], box[3], box[4] = nx1, ny1, nx1 + bw, ny1 + bh

            self.dirty = True
            self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode     = None
            self.drag_start    = None
            self.drag_box_orig = None

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window, self._on_mouse)
        self._load_frame()
        self._redraw()

        while True:
            raw = cv2.waitKey(20)
            if raw == -1:
                continue
            key    = raw & 0xFF
            win_vk = (raw >> 16) & 0xFF   # Windows special key VK code (e.g. 0x27=Right)

            # d  → next frame
            if key == ord('d'):
                self._auto_save()
                self.idx = min(self.idx + 1, self.total - 1)
                self._load_frame()
                self._redraw()

            # a  → prev frame
            elif key == ord('a'):
                self._auto_save()
                self.idx = max(self.idx - 1, 0)
                self._load_frame()
                self._redraw()

            # .  → next video group
            elif key == ord('.'):
                self._jump_to_video(+1)

            # ,  → prev video group
            elif key == ord(','):
                self._jump_to_video(-1)

            # Delete  → remove selected bbox
            elif key in (255, 127) or win_vk == 0x2E:   # VK_DELETE=0x2E
                if 0 <= self.selected < len(self.boxes):
                    del self.boxes[self.selected]
                    self.selected = -1
                    self.dirty    = True
                    self._redraw()

            elif key in (ord('q'), 27):     # q or ESC → quit
                self._auto_save()
                break

        cv2.destroyAllWindows()


def main():
    args = parse_args()
    input_dir  = args.input
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.isdir(images_dir):
        print(f"[ERROR] images/ not found in {input_dir}")
        return

    image_files = sorted(glob_module.glob(os.path.join(images_dir, "*.jpg")) +
                         glob_module.glob(os.path.join(images_dir, "*.png")))
    if not image_files:
        print(f"[ERROR] No images found in {images_dir}")
        return

    classes = load_classes(input_dir)
    ui = ReviewUI(image_files, labels_dir, classes, args.win_w, args.win_h)
    ui.run()


if __name__ == "__main__":
    main()
