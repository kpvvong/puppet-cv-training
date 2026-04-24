"""
Label Studio JSON → frame extraction + YOLO annotations
----------------------------------------------
Supports Label Studio video keyframe format (sequence interpolation).
Classes are extracted automatically from the JSON.

Usage:
  # Auto-find files by project name
  python extract_frames.py --project <project_name> [--all-frames]

  # Specify paths manually
  python extract_frames.py --json <json_file> --video <video_file> [--output <dir>] [--all-frames]

Examples:
  python extract_frames.py --project 20260421_single-puppet-test
  python extract_frames.py \\
      --json project-1-at-2026-04-21-18-58-4ab55e31.json \\
      --video "project-1-at-2026-04-21-18-32-4ab55e31/images/6bed04b5-puppet_1.mp4" \\
      --output dataset_raw

Output:
  <output>/
  ├── images/      ← extracted frame images (.jpg)
  ├── labels/      ← YOLO format annotations (.txt)
  └── classes.txt  ← class list extracted from JSON
"""

import os
import json
import glob as glob_module
import argparse
import cv2


def find_project_files(project_name):
    """Find JSON and video under projects/{project_name}/. Returns (json_file, video_file, output_dir)."""
    project_dir = os.path.join("projects", project_name)
    if not os.path.isdir(project_dir):
        print(f"[ERROR] Project folder not found: {project_dir}")
        return None, None, None

    json_file = os.path.join(project_dir, f"{project_name}.json")
    if not os.path.exists(json_file):
        print(f"[ERROR] {project_name}.json not found in {project_dir}")
        return None, None, None

    video_files = glob_module.glob(os.path.join(project_dir, "**", "*.mp4"), recursive=True)
    if not video_files:
        print(f"[ERROR] No .mp4 files found in {project_dir}")
        return None, None, None
    video_file = video_files[0]
    if len(video_files) > 1:
        print(f"[WARN] Multiple videos found, using: {video_file}")

    output_dir = os.path.join(project_dir, "dataset_raw")
    return json_file, video_file, output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Label Studio JSON → YOLO frame extraction")
    parser.add_argument("--project", default=None, help="Project name, auto-finds files under projects/{name}/")
    parser.add_argument("--json",    default=None, help="Label Studio exported JSON path (manual)")
    parser.add_argument("--video",   default=None, help="Video file path (manual)")
    parser.add_argument("--output",  default=None, help="Output folder (default: projects/{name}/dataset_raw)")
    parser.add_argument("--all-frames", action="store_true",
                        help="Output all frames including unannotated ones (default: annotated only)")
    args = parser.parse_args()

    if args.project:
        json_file, video_file, output_dir = find_project_files(args.project)
        if not json_file:
            exit(1)
        args.json   = args.json   or json_file
        args.video  = args.video  or video_file
        args.output = args.output or output_dir
    else:
        if not args.json or not args.video:
            parser.error("Provide --project or both --json and --video")
        args.output = args.output or "dataset_raw"

    return args


def extract_classes_from_json(data):
    """Extract all classes from Label Studio JSON in order of appearance."""
    classes = []
    for item in data:
        for ann in item.get("annotations", []):
            for result in ann.get("result", []):
                val = result.get("value", {})
                for key in ("labels", "rectanglelabels", "polygonlabels", "ellipselabels"):
                    for label in val.get(key, []):
                        if label not in classes:
                            classes.append(label)
    return classes


def lerp(a, b, t):
    return a + (b - a) * t


def interpolate_sequence(sequence):
    """Linear interpolation between keyframes. Returns {frame_num: (x, y, w, h)} in percentages."""
    keyframes = [s for s in sequence if s.get("enabled", True)]
    keyframes.sort(key=lambda s: s["frame"])

    if not keyframes:
        return {}

    frame_bboxes = {}
    for i, kf in enumerate(keyframes):
        frame_bboxes[kf["frame"]] = (kf["x"], kf["y"], kf["width"], kf["height"])

        if i + 1 < len(keyframes):
            nxt   = keyframes[i + 1]
            start = kf["frame"]
            end   = nxt["frame"]
            for step in range(1, end - start):
                t = step / (end - start)
                frame_bboxes[start + step] = (
                    lerp(kf["x"],      nxt["x"],      t),
                    lerp(kf["y"],      nxt["y"],      t),
                    lerp(kf["width"],  nxt["width"],  t),
                    lerp(kf["height"], nxt["height"], t),
                )
    return frame_bboxes


def pct_to_yolo(x_pct, y_pct, w_pct, h_pct):
    """Convert Label Studio top-left percentage to YOLO center format (0~1)."""
    cx = max(0.0, min(1.0, (x_pct + w_pct / 2) / 100))
    cy = max(0.0, min(1.0, (y_pct + h_pct / 2) / 100))
    w  = max(0.0, min(1.0, w_pct / 100))
    h  = max(0.0, min(1.0, h_pct / 100))
    return cx, cy, w, h


def main():
    args = parse_args()

    json_file    = args.json
    video_file   = args.video
    output_dir   = args.output
    labeled_only = not args.all_frames

    if not os.path.exists(json_file):
        print(f"[ERROR] JSON file not found: {json_file}")
        return
    if not os.path.exists(video_file):
        print(f"[ERROR] Video file not found: {video_file}")
        return

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    classes = extract_classes_from_json(data)
    if not classes:
        print("[WARN] No labels found in JSON, class list will be built from order of appearance")

    print(f"Classes detected: {classes}")

    all_objects = []  # list of (class_id, frame_bboxes_dict)

    for item in data:
        for ann in item.get("annotations", []):
            for result in ann.get("result", []):
                val = result.get("value", {})
                if "sequence" not in val:
                    continue

                label_name = None
                for key in ("labels", "rectanglelabels", "polygonlabels"):
                    if key in val and val[key]:
                        label_name = val[key][0]
                        break

                if label_name is None:
                    if classes:
                        label_name = classes[0]
                    else:
                        label_name = "object"
                        classes.append(label_name)

                if label_name not in classes:
                    classes.append(label_name)
                    print(f"[INFO] New class: {label_name} (id={classes.index(label_name)})")

                class_id     = classes.index(label_name)
                frame_bboxes = interpolate_sequence(val["sequence"])
                all_objects.append((class_id, frame_bboxes))
                print(f"  '{label_name}': {len(val['sequence'])} keyframes → {len(frame_bboxes)} interpolated frames")

    if not all_objects:
        print("[ERROR] No valid annotation sequences found in JSON")
        return

    all_frames = set()
    for _, fb in all_objects:
        all_frames |= set(fb.keys())

    frame_annotations = {}
    for class_id, frame_bboxes in all_objects:
        for frame_num, (x, y, w, h) in frame_bboxes.items():
            cx, cy, fw, fh = pct_to_yolo(x, y, w, h)
            line = f"{class_id} {cx:.6f} {cy:.6f} {fw:.6f} {fh:.6f}"
            frame_annotations.setdefault(frame_num, []).append(line)

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_file}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = sorted(all_frames) if labeled_only else range(total_video_frames)
    frame_set     = set(target_frames)

    print(f"\nExtracting frames ({len(target_frames)} target, {total_video_frames} total in video)...")

    saved = 0
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ls_frame = current_frame + 1  # Label Studio frames are 1-indexed

        if ls_frame in frame_set:
            name = f"frame_{ls_frame:06d}"
            cv2.imwrite(f"{output_dir}/images/{name}.jpg", frame)
            with open(f"{output_dir}/labels/{name}.txt", "w") as f:
                f.write("\n".join(frame_annotations.get(ls_frame, [])))
            saved += 1
            if saved % 50 == 0:
                print(f"  Saved {saved}/{len(target_frames)} frames...")

        current_frame += 1

    cap.release()

    with open(f"{output_dir}/classes.txt", "w") as f:
        f.write("\n".join(classes))

    print(f"\nDone! Extracted {saved} frames")
    print(f"Classes: {classes}")
    print(f"Output:  {output_dir}/")
    if args.project:
        print(f"\nNext: python prepare_dataset.py --project {args.project}")
    else:
        print(f"\nNext: python prepare_dataset.py --input {output_dir}")


if __name__ == "__main__":
    main()
