"""Cityscapes label definitions (Cordts et al., CVPR 2016), shared by KITTI's semantic benchmark.

Annotations store a raw *label id* per pixel (0-33). The benchmark evaluates 19 of them as
*train ids* 0-18; everything else is ignored (255). One ignored class matters for road
inspection: ``guard rail`` (id 14) is annotated, but no Cityscapes-trained model predicts
it. ``TRAIN_IDS_WITH_GUARDRAIL`` adds it as train id 19 for the fine-tuning experiment.
"""

from __future__ import annotations

import numpy as np

IGNORE = 255

# label id -> (train id, name, RGB colour)
_LABELS = {
    7: (0, "road", (128, 64, 128)),
    8: (1, "sidewalk", (244, 35, 232)),
    11: (2, "building", (70, 70, 70)),
    12: (3, "wall", (102, 102, 156)),
    13: (4, "fence", (190, 153, 153)),
    17: (5, "pole", (153, 153, 153)),
    19: (6, "traffic light", (250, 170, 30)),
    20: (7, "traffic sign", (220, 220, 0)),
    21: (8, "vegetation", (107, 142, 35)),
    22: (9, "terrain", (152, 251, 152)),
    23: (10, "sky", (70, 130, 180)),
    24: (11, "person", (220, 20, 60)),
    25: (12, "rider", (255, 0, 0)),
    26: (13, "car", (0, 0, 142)),
    27: (14, "truck", (0, 0, 70)),
    28: (15, "bus", (0, 60, 100)),
    31: (16, "train", (0, 80, 100)),
    32: (17, "motorcycle", (0, 0, 230)),
    33: (18, "bicycle", (119, 11, 32)),
}
GUARD_RAIL_ID = 14
GUARD_RAIL = (19, "guard rail", (255, 128, 0))

CLASSES = [name for _, name, _ in sorted(_LABELS.values())]
CLASSES_WITH_GUARDRAIL = [*CLASSES, GUARD_RAIL[1]]
PALETTE = np.array([color for *_, color in sorted(_LABELS.values())] + [GUARD_RAIL[2]], np.uint8)

TRAIN_IDS = {label_id: train_id for label_id, (train_id, _, _) in _LABELS.items()}
TRAIN_IDS_WITH_GUARDRAIL = {**TRAIN_IDS, GUARD_RAIL_ID: GUARD_RAIL[0]}


def to_train_ids(label_ids: np.ndarray, guardrail: bool = False) -> np.ndarray:
    """Raw Cityscapes/KITTI label ids -> train ids (19 or 20 classes), IGNORE elsewhere."""
    lut = np.full(256, IGNORE, np.uint8)
    for label_id, train_id in (TRAIN_IDS_WITH_GUARDRAIL if guardrail else TRAIN_IDS).items():
        lut[label_id] = train_id
    return lut[label_ids]


def colorize(train_ids: np.ndarray) -> np.ndarray:
    """(H, W) train ids -> (H, W, 3) RGB; ignored pixels are black."""
    rgb = np.zeros((*train_ids.shape, 3), np.uint8)
    valid = train_ids < len(PALETTE)
    rgb[valid] = PALETTE[train_ids[valid]]
    return rgb


def ultralytics_label_mapping(guardrail: bool = False) -> dict[int, int | str]:
    """``label_mapping`` for an Ultralytics semantic dataset YAML (raw ids are kept on disk)."""
    ids = TRAIN_IDS_WITH_GUARDRAIL if guardrail else TRAIN_IDS
    return {i: ids.get(i, "ignore_label") for i in range(-1, 34)}
