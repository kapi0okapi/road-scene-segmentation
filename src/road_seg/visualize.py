"""Figures: image | ground truth | prediction, and overlays for single images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from road_seg.labels import colorize


def overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(rgb, 1 - alpha, colorize(labels), alpha, 0)


def save_example(rgb: np.ndarray, truth: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    """Stacked vertically: image, ground truth, prediction (ignored pixels black in the truth)."""
    panel = np.concatenate([rgb, colorize(truth), overlay(rgb, prediction, 0.6)], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
