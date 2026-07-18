"""Cityscapes (Cordts et al., CVPR 2016): 2975 train / 500 val images of German and Swiss cities.

The archives need a free account, so they are downloaded by hand or with
``scripts/download_cityscapes.sh`` (your own credentials): ``leftImg8bit_trainvaltest.zip``
(11 GB) and ``gtFine_trainvaltest.zip`` (241 MB), extracted into one folder.

    seg-prepare-cityscapes --root data/cityscapes      # data/cityscapes/yolo19 and yolo20_guardrail
"""

from __future__ import annotations

import argparse
from pathlib import Path

from road_seg.datasets.common import Sample, write_ultralytics


def samples(root: Path, split: str) -> list[Sample]:
    images = sorted((Path(root) / "leftImg8bit" / split).rglob("*_leftImg8bit.png"))
    if not images:
        raise FileNotFoundError(f"no Cityscapes {split} images under {root}/leftImg8bit")
    out = []
    for image in images:
        city = image.parent.name
        label = Path(root) / "gtFine" / split / city / image.name.replace("_leftImg8bit", "_gtFine_labelIds")
        out.append(Sample(image, label))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/cityscapes"))
    args = parser.parse_args(argv)
    parts = {split: samples(args.root, split) for split in ("train", "val")}
    for guardrail, name in ((False, "yolo19"), (True, "yolo20_guardrail")):
        path = write_ultralytics(parts, args.root / name, guardrail=guardrail)
        print(f"{path}: {len(parts['train'])} train / {len(parts['val'])} val")


if __name__ == "__main__":
    main()
