#!/bin/bash
# Pre-fetch all model weights on a login node (compute nodes may not have internet access).
# HF_HOME must match the Slurm jobs ($WORK/.cache/huggingface), which slurm/generate.py sets.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${WORK:-}" ]]; then export HF_HOME="$WORK/.cache/huggingface"; fi
mkdir -p models
python - <<'PY'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font

for name in ("yolo26n-sem.pt", "yolo26s-sem.pt", "yolo26m-sem.pt"):
    YOLO(name)
check_font("Arial.ttf")

from road_seg.models import MODELS
import transformers

for name, spec in MODELS.items():
    if spec.family != "yolo":
        transformers.AutoImageProcessor.from_pretrained(spec.weights)
        transformers.AutoModel.from_pretrained(spec.weights)
        print("cached", name)
PY
