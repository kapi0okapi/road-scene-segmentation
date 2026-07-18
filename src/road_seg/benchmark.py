"""Zero-shot benchmark on KITTI with a clean choice of the input scale.

The best input scale depends on how a model was trained (Ultralytics trains YOLO26-sem on
Cityscapes shrunk to 1024 px wide, the transformers on full-resolution crops), so each model
gets the scale that works best on a quarter of the KITTI *train* split, and is then scored on
the 40 *val* images, the same images the fine-tuned model is scored on.

    seg-benchmark --models yolo26s-sem eomt-l --root data/kitti_semantics --out runs/benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from road_seg.datasets import kitti
from road_seg.evaluate import evaluate
from road_seg.models import Segmenter

log = logging.getLogger("road_seg.benchmark")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[375, 768, 1152], help="image heights")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--guardrail", action="store_true", help="20-class models (fine-tuned on KITTI)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parts = kitti.split(args.root)
    selection = parts["train"][::4]  # 40 images for choosing the scale
    args.out.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        name = Path(model).stem if model.endswith(".pt") else model
        target = args.out / f"{name}.json"
        if target.exists():
            log.info("%s already done", name)
            continue
        sweep = {}
        for scale in args.scales:
            segmenter = Segmenter(model, args.device, args.half, scale)
            sweep[scale] = evaluate(segmenter, selection, args.guardrail)["miou"]
            log.info("%s at %d px: selection mIoU %.4f", name, scale, sweep[scale])
        best = max(sweep, key=sweep.get)
        segmenter = Segmenter(model, args.device, args.half, best)
        result = evaluate(segmenter, parts["val"], args.guardrail, args.out / "examples" / name)
        report = {
            "model": model,
            "scale_to": best,
            "selection_miou": sweep,
            "window": segmenter.spec.window,
            "device": str(segmenter.device),
            "paper_cityscapes_miou": segmenter.spec.paper_miou,
            "paper_source": segmenter.spec.source,
            **result,
        }
        target.write_text(json.dumps(report, indent=2))
        print(f"{name}: scale {best}, val mIoU {report['miou']:.4f}, {report['ms_per_image']:.0f} ms/image")


if __name__ == "__main__":
    main()
