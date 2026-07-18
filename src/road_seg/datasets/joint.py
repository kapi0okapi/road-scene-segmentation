r"""Cityscapes + KITTI with guard rail as the 20th class, for fine-tuning without forgetting.

    seg-prepare-joint --cityscapes data/cityscapes/yolo20_guardrail \
        --kitti data/kitti_semantics/yolo20_guardrail --out data/joint20

Fine-tuning on KITTI's 160 training images alone teaches the guard rail class but costs the
other classes: rare ones (bus, train, rider) vanish from the predictions. Cityscapes annotates
guard rails too (label id 14, left out of its benchmark), so its 2975 training images can
carry the old classes and the new one at the same time. Validation stays on KITTI val, the
target domain. Both datasets use Cityscapes label ids, so one label mapping serves both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def write_joint(cityscapes: Path, kitti: Path, out: Path, kitti_repeats: int = 1) -> Path:
    """A data.yaml training on both image folders; Ultralytics finds each mask by replacing
    ``images`` with ``masks`` in the image path, so the folders stay where they are.
    ``kitti_repeats`` lists the KITTI folder several times to raise its share of the batches."""
    cs = yaml.safe_load((Path(cityscapes) / "data.yaml").read_text())
    kt = yaml.safe_load((Path(kitti) / "data.yaml").read_text())
    if cs["names"] != kt["names"] or cs["label_mapping"] != kt["label_mapping"]:
        raise ValueError("both datasets must use the same classes and label mapping")
    train = [str(Path(cs["path"]) / cs["train"])] + [str(Path(kt["path"]) / kt["train"])] * kitti_repeats
    data = {
        "path": str(Path(out).resolve()),
        "train": train,
        "val": str(Path(kt["path"]) / kt["val"]),
        "masks_dir": "masks",
        "names": kt["names"],
        "label_mapping": kt["label_mapping"],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cityscapes", type=Path, default=Path("data/cityscapes/yolo20_guardrail"))
    parser.add_argument("--kitti", type=Path, default=Path("data/kitti_semantics/yolo20_guardrail"))
    parser.add_argument("--out", type=Path, default=Path("data/joint20"))
    parser.add_argument("--kitti-repeats", type=int, default=4)
    args = parser.parse_args(argv)
    print(write_joint(args.cityscapes, args.kitti, args.out, args.kitti_repeats))


if __name__ == "__main__":
    main()
