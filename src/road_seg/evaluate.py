"""Evaluate a model on KITTI semantics or Cityscapes val with the shared inference protocol.

    seg-eval --model eomt-l --dataset kitti --root data/kitti_semantics --out runs/eval/eomt-l_kitti.json
    seg-eval --model yolo26s-sem --dataset cityscapes --root data/cityscapes --out runs/eval/yolo26s_cs.json
    seg-eval --model runs/kitti20/weights/best.pt --dataset kitti --split val --guardrail ...

KITTI (Cityscapes-trained models) is a zero-shot transfer test: other city, other camera.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from road_seg.datasets import cityscapes, kitti
from road_seg.datasets.common import load
from road_seg.metrics import ConfusionMatrix
from road_seg.models import Segmenter

log = logging.getLogger("road_seg.evaluate")


def dataset_samples(name: str, root: Path, split: str):
    if name == "kitti":
        return kitti.samples(root) if split == "all" else kitti.split(root)[split]
    return cityscapes.samples(root, "val" if split == "all" else split)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate(segmenter: Segmenter, samples, guardrail: bool, examples: Path | None = None) -> dict:
    cm = ConfusionMatrix(len(segmenter.classes))
    seconds = []
    for i, sample in enumerate(samples):
        rgb, truth = load(sample, guardrail)
        synchronize(segmenter.device)
        started = time.perf_counter()
        prediction = segmenter(rgb)
        synchronize(segmenter.device)
        seconds.append(time.perf_counter() - started)
        cm.update(truth, prediction)
        if examples is not None and i < 6:
            from road_seg.visualize import save_example

            save_example(rgb, truth, prediction, examples / f"{sample.image.stem}.jpg")
        if i % 50 == 0:
            log.info("%s: %d/%d images, running mIoU %.3f", segmenter.name, i + 1, len(samples), cm.miou())
    iou = cm.iou()
    return {
        "images": len(samples),
        "miou": cm.miou(),
        "miou_all_classes": cm.miou(present_only=False),
        "pixel_accuracy": cm.pixel_accuracy(),
        "iou": {c: (None if np.isnan(v) else float(v)) for c, v in zip(segmenter.classes, iou, strict=True)},
        "ms_per_image": 1000 * float(np.median(seconds[1:] if len(seconds) > 1 else seconds)),
        "confusion": cm.matrix.tolist(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="name from road_seg.models.MODELS or a YOLO .pt file")
    parser.add_argument("--dataset", choices=["kitti", "cityscapes"], default="kitti")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="all", help="kitti: all | train | val; cityscapes: val")
    parser.add_argument("--guardrail", action="store_true", help="20 classes: Cityscapes + guard rail")
    parser.add_argument("--scale-to", type=int, default=1024, help="image height fed to the model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--examples", type=Path, help="write a few prediction figures here")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = dataset_samples(args.dataset, args.root, args.split)[: args.max_images]
    segmenter = Segmenter(args.model, args.device, args.half, args.scale_to)
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "device": str(segmenter.device),
        "scale_to": args.scale_to,
        "window": segmenter.spec.window,
        "paper_cityscapes_miou": segmenter.spec.paper_miou,
        **evaluate(segmenter, samples, args.guardrail, args.examples),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    miou, ms = report["miou"], report["ms_per_image"]
    print(f"{args.model} on {args.dataset}/{args.split}: mIoU {miou:.4f}, {ms:.0f} ms")


if __name__ == "__main__":
    main()
