"""Confusion-matrix metrics for semantic segmentation (ignore label 255 excluded)."""

from __future__ import annotations

import numpy as np

from road_seg.labels import IGNORE


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.n = num_classes
        self.matrix = np.zeros((num_classes, num_classes), np.int64)  # rows: truth, columns: prediction

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        valid = (truth != IGNORE) & (truth < self.n)
        index = truth[valid].astype(np.int64) * self.n + prediction[valid].astype(np.int64)
        self.matrix += np.bincount(index, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        """Per-class IoU; NaN for classes that occur neither in the truth nor in the prediction."""
        tp = np.diag(self.matrix).astype(float)
        union = self.matrix.sum(0) + self.matrix.sum(1) - tp
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, tp / union, np.nan)

    def miou(self, present_only: bool = True) -> float:
        """Mean IoU over classes present in the ground truth (KITTI lacks e.g. trains), or all."""
        iou = self.iou()
        if present_only:
            iou = iou[self.matrix.sum(1) > 0]
        return float(np.nanmean(iou))

    def pixel_accuracy(self) -> float:
        return float(np.diag(self.matrix).sum() / max(self.matrix.sum(), 1))
