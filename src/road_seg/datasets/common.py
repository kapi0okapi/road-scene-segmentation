"""Image/label pairs and the folder layout Ultralytics trains semantic models on."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from road_seg.labels import CLASSES, CLASSES_WITH_GUARDRAIL, to_train_ids, ultralytics_label_mapping


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path  # raw Cityscapes label ids


def load(sample: Sample, guardrail: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and train-id label map of the same size."""
    rgb = np.array(Image.open(sample.image).convert("RGB"))
    return rgb, to_train_ids(np.array(Image.open(sample.label)), guardrail)


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(Path(os.path.abspath(source)))
    except OSError:
        shutil.copy2(source, target)


def write_ultralytics(splits: dict[str, list[Sample]], out: Path, guardrail: bool) -> Path:
    """images/<split>/<name>.png + masks/<split>/<name>.png (raw label ids, linked) + data.yaml.
    Raw ids stay on disk; the YAML's label_mapping turns them into train ids while loading."""
    for split, samples in splits.items():
        for s in samples:
            name = s.image.stem.replace("_leftImg8bit", "") + s.image.suffix
            link(s.image, out / "images" / split / name)
            link(s.label, out / "masks" / split / name)
    names = CLASSES_WITH_GUARDRAIL if guardrail else CLASSES
    data = {
        "path": str(out.resolve()),
        **{split: f"images/{split}" for split in splits},
        "masks_dir": "masks",
        "names": dict(enumerate(names)),
        "label_mapping": ultralytics_label_mapping(guardrail),
    }
    path = out / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
