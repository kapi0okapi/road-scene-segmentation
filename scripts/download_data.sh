#!/bin/bash
# KITTI semantics (328 MB, CC BY-NC-SA 3.0) + the Ultralytics layouts (19 and 20 classes).
#   bash scripts/download_data.sh            # into $DATA/kitti_semantics (cluster) or ./data/...
set -euo pipefail
cd "$(dirname "$0")/.."
python -m road_seg.datasets.kitti --out "${DATA:-data}/kitti_semantics"
