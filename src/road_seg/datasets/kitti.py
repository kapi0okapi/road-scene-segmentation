"""KITTI semantic segmentation benchmark (Alhaija et al., IJCV 2018): 200 annotated frames from
KITTI drives around Karlsruhe, labelled with the Cityscapes classes. Licence: CC BY-NC-SA 3.0.

    seg-prepare-kitti --out data/kitti_semantics                 # download (328 MB) + YOLO layouts

The 200 images are split into a fixed block of 160 for fine-tuning and 40 for validation
(000160-000199). A contiguous block keeps near-duplicate frames of one drive on one side.
Zero-shot evaluation of Cityscapes models uses all 200.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests

from road_seg.datasets.common import Sample, write_ultralytics

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"
VAL_START = 160


def samples(root: Path) -> list[Sample]:
    images = sorted((Path(root) / "training" / "image_2").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no KITTI semantics in {root}; run seg-prepare-kitti")
    return [Sample(p, p.parent.parent / "semantic" / p.name) for p in images]


def split(root: Path) -> dict[str, list[Sample]]:
    all_samples = samples(root)
    return {"train": all_samples[:VAL_START], "val": all_samples[VAL_START:]}


def download(out: Path) -> None:
    if (out / "training" / "semantic").is_dir():
        return
    out.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=600)
    response.raise_for_status()
    zipfile.ZipFile(io.BytesIO(response.content)).extractall(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/kitti_semantics"))
    args = parser.parse_args(argv)
    download(args.out)
    parts = split(args.out)
    for guardrail, name in ((False, "yolo19"), (True, "yolo20_guardrail")):
        path = write_ultralytics(parts, args.out / name, guardrail)
        print(f"{path}: {len(parts['train'])} train / {len(parts['val'])} val")


if __name__ == "__main__":
    main()
