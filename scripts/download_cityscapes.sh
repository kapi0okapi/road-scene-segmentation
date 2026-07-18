#!/bin/bash
# Cityscapes needs a (free) account at https://www.cityscapes-dataset.com. This uses the official
# downloader with YOUR credentials: it asks for them, or reads CITYSCAPES_USERNAME and
# CITYSCAPES_PASSWORD from your environment. ~11 GB.
#   bash scripts/download_cityscapes.sh      # into $DATA/cityscapes (cluster) or ./data/cityscapes
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${DATA:-data}/cityscapes"
mkdir -p "$OUT"
pip install -q cityscapesscripts
csDownload -d "$OUT" gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
(cd "$OUT" && unzip -q -n gtFine_trainvaltest.zip && unzip -q -n leftImg8bit_trainvaltest.zip)
python -m road_seg.datasets.cityscapes --root "$OUT"
