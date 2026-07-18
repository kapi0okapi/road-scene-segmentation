"""Segment images or a video and write colour overlays (and, optionally, raw train-id PNGs).

seg-predict --model yolo26s-sem --source photos/ --out runs/predict
seg-predict --model runs/kitti20_yolo26s/weights/best.pt --source drive.mp4 --out runs/video
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from road_seg.models import Segmenter
from road_seg.visualize import overlay

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_shares(labels: np.ndarray, classes: list[str]) -> dict[str, float]:
    counts = np.bincount(labels.ravel(), minlength=len(classes))[: len(classes)]
    return {c: round(float(n) / labels.size, 4) for c, n in zip(classes, counts, strict=True) if n}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolo26s-sem")
    parser.add_argument("--source", type=Path, required=True, help="image, folder of images or video")
    parser.add_argument("--out", type=Path, default=Path("runs/predict"))
    parser.add_argument("--scale-to", type=int, default=375, help="image height fed to the model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-labels", action="store_true", help="also write train-id PNGs")
    args = parser.parse_args(argv)

    segmenter = Segmenter(args.model, args.device, scale_to=args.scale_to)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        capture = cv2.VideoCapture(str(args.source))
        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        writer, frames, started = None, 0, time.perf_counter()
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out = cv2.cvtColor(overlay(rgb, segmenter(rgb)), cv2.COLOR_RGB2BGR)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.out / "overlay.mp4"), fourcc, fps, out.shape[1::-1])
            writer.write(out)
            frames += 1
        capture.release()
        if writer is not None:
            writer.release()
        fps_done = frames / (time.perf_counter() - started)
        print(f"{frames} frames, {fps_done:.1f} FPS -> {args.out / 'overlay.mp4'}")
        return

    paths = (
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if args.source.is_dir()
        else [args.source]
    )
    for path in paths:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        labels = segmenter(rgb)
        out = cv2.cvtColor(overlay(rgb, labels), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(args.out / f"{path.stem}_overlay.jpg"), out)
        if args.save_labels:
            cv2.imwrite(str(args.out / f"{path.stem}_labels.png"), labels)
        top = sorted(class_shares(labels, segmenter.classes).items(), key=lambda kv: -kv[1])[:4]
        print(path.name, ", ".join(f"{c} {v:.0%}" for c, v in top))


if __name__ == "__main__":
    main()
