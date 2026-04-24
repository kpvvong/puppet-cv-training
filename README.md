# puppet-cv-training

YOLOv8 object detection training pipeline with SAM2 auto-annotation support.

## Overview

Two annotation workflows are supported:

**Workflow A — Label Studio → YOLO training**
```
extract_frames.py → (Label Studio) → prepare_dataset.py → train.py
```

**Workflow B — SAM2 auto-annotation → YOLO training**
```
sam2_annotate.py → (sam2_review.py) → prepare_dataset.py → train.py
```

**Workflow C — SAM2 auto-annotation → CVAT review → YOLO training**
```
sam2_annotate.py → export_cvat.py → (CVAT) → import_cvat.py → prepare_dataset.py → train.py
```

---

## Installation

```bash
conda create -n cvyolo python=3.11
conda activate cvyolo

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# Other dependencies
pip install opencv-python ultralytics python-osc

# SAM2 weights
mkdir -p models
curl -L -o models/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

---

## Project structure

```
projects/
  {project_name}/
    *.mp4               # source videos
    {project_name}.json # Label Studio export (Workflow A)
    dataset_raw/        # annotated frames (images/ labels/ classes.txt)
    dataset/            # train/val split + data.yaml
    cvat_export.zip     # CVAT import/export (Workflow C)
models/
  {model_name}/
    weights/
      best.pt
      last.pt
```

---

## Scripts

### `extract_frames.py`
Extract frames and annotations from a Label Studio JSON export.

```bash
python extract_frames.py --project <project_name>
```

Output: `projects/{name}/dataset_raw/`

---

### `sam2_annotate.py`
Auto-annotate video frames using SAM2. Supports multiple videos per project.

```bash
python sam2_annotate.py --project <project_name> --classes Puppet,Hand
```

**Controls:**
| Key | Action |
|-----|--------|
| d / Right | Next frame |
| a / Left | Previous frame |
| b | Toggle point / box mode |
| Left click (point mode) | Foreground point |
| Right click (point mode) | Background point |
| Left drag (box mode) | Foreground box |
| Right drag (box mode) | Exclusion box |
| n | New object |
| u | Undo |
| Enter / Space | Confirm and start tracking |
| ESC | Skip video |

Options:
- `--device cpu` — use CPU instead of GPU
- `--preview` — show live tracking window during propagation
- `--window-size 1920x1080` — set initial window size

Output: `projects/{name}/dataset_raw/`

---

### `sam2_review.py`
Review and edit bounding boxes after annotation.

```bash
python sam2_review.py --project <project_name>
```

**Controls:**
| Key / Action | Behavior |
|---|---|
| d / Right | Next frame (auto-saves) |
| a / Left | Previous frame (auto-saves) |
| Click bbox | Select |
| Drag corner handle | Resize |
| Drag bbox center | Move |
| Delete | Delete selected bbox |
| q / ESC | Quit |

---

### `export_cvat.py`
Export `dataset_raw` to CVAT 1.1 XML zip for human review in CVAT.

```bash
python export_cvat.py --project <project_name>
```

Output: `projects/{name}/cvat_export.zip`

CVAT import: **Actions → Upload annotations → CVAT 1.1**

---

### `import_cvat.py`
Import CVAT YOLO 1.1 export back into `dataset_raw` format.

```bash
python import_cvat.py --project <project_name> --zip projects/<project_name>/cvat_export.zip
```

---

### `prepare_dataset.py`
Split `dataset_raw` into train/val sets and generate `data.yaml`.

```bash
python prepare_dataset.py --project <project_name>
```

Output: `projects/{name}/dataset/`

---

### `train.py`
Train a YOLOv8n model.

```bash
python train.py --name <model_name> --data projects/<project_name>/dataset/data.yaml
```

Options:
- `--epochs 100` (default: 100)
- `--batch 16` (default: 16)
- `--device cpu`

Output: `models/{name}/weights/best.pt`

---

### `webcam.py`
Real-time detection with OSC output and optional depth map.

```bash
python webcam.py
```

Edit `MODEL_PATH` and settings at the top of the file before running.

**Controls:** `q` quit · `s` screenshot · `o` OSC · `m` mask · `d` depth · `+/-` threshold

---

## CVAT Nuclio serverless (SAM2 interactive annotation)

SAM2 can be deployed as a Nuclio serverless function for interactive annotation directly in CVAT.

```bash
# Start CVAT with Nuclio
cd ~/cvat
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

# Deploy SAM2
./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam2
```

In CVAT: open a Job → **AI Tools** (wand icon) → **Interactors** → **Segment Anything 2.1**
