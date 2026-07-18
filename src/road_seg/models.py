"""Four model families behind one inference protocol.

Every model returns per-pixel class scores for a batch of crops; ``Segmenter`` does the rest
the same way for all of them:

1. rescale the image so objects have their Cityscapes size (KITTI frames are 375 px high,
   Cityscapes 1024 px with a ~3x longer focal length; default: height -> ``scale_to`` px),
2. run whole-image inference, or overlapping square windows whose scores are averaged,
3. resize the scores back to the input and take the arg-max.

Models (all trained on Cityscapes train, 19 classes):

* YOLO26-sem (Ultralytics, 2026): a real-time CNN, the newest YOLO with a semantic head.
* SegFormer (Xie et al., NeurIPS 2021): hierarchical transformer encoder + MLP decoder.
* Mask2Former (Cheng et al., CVPR 2022): masked-attention transformer decoder on Swin-L.
* EoMT (Kerssies et al., CVPR 2025): a plain ViT-L (DINOv2) that predicts masks itself, no
  adapter or pixel decoder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelSpec:
    family: str  # yolo | segformer | mask2former | eomt
    weights: str
    window: int  # 0 = whole image (padded to a stride multiple)
    paper_miou: float | None = None  # published Cityscapes val mIoU, single scale
    source: str = ""


MODELS = {
    "yolo26n-sem": ModelSpec("yolo", "yolo26n-sem.pt", 0, 78.3, "Ultralytics docs"),
    "yolo26s-sem": ModelSpec("yolo", "yolo26s-sem.pt", 0, 80.8, "Ultralytics docs"),
    "yolo26m-sem": ModelSpec("yolo", "yolo26m-sem.pt", 0, 82.0, "Ultralytics docs"),
    "segformer-b0": ModelSpec(
        "segformer", "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 1024, 76.2, "SegFormer, Tab. 1"
    ),
    "segformer-b5": ModelSpec(
        "segformer", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", 1024, 82.4, "SegFormer, Tab. 1"
    ),
    "mask2former-swin-l": ModelSpec(
        "mask2former", "facebook/mask2former-swin-large-cityscapes-semantic", 0, 83.3, "Mask2Former"
    ),
    "eomt-l": ModelSpec("eomt", "tue-mps/cityscapes_semantic_eomt_large_1024", 1024, 84.2, "EoMT, Tab. 5"),
}


def select_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Segmenter:
    """``Segmenter("eomt-l")(rgb)`` -> (H, W) uint8 train ids. A path to a ``.pt`` file loads a
    fine-tuned YOLO semantic model (any number of classes)."""

    def __init__(
        self, name: str, device: str = "auto", half: bool = False, scale_to: int = 1024, batch: int = 2
    ):
        self.name = name
        self.batch = batch  # windows per forward pass; large batches of 1024^2 crops exhaust GPU memory
        self.spec = MODELS.get(name) or ModelSpec("yolo", name, 0)
        self.device = select_device(device)
        self.dtype = torch.float16 if half and self.device.type != "cpu" else torch.float32
        self.scale_to = scale_to
        self._load()

    # ----------------------------------------------------------------- loading
    def _load(self) -> None:
        family, weights = self.spec.family, os.path.expandvars(self.spec.weights)
        if family == "yolo":
            from ultralytics import YOLO

            yolo = YOLO(weights)
            self.net = yolo.model.to(self.device, self.dtype).eval()
            self.classes = [yolo.names[i] for i in range(len(yolo.names))]
            self.stride = int(max(self.net.stride))
            self.mean = torch.zeros(3, device=self.device)  # Ultralytics feeds RGB in [0, 1]
            self.std = torch.ones(3, device=self.device)
            return
        import transformers

        cls = {
            "segformer": transformers.SegformerForSemanticSegmentation,
            "mask2former": transformers.Mask2FormerForUniversalSegmentation,
            "eomt": transformers.EomtForUniversalSegmentation,
        }[family]
        processor = transformers.AutoImageProcessor.from_pretrained(weights)
        self.net = cls.from_pretrained(weights, dtype=self.dtype).to(self.device).eval()
        self.classes = [self.net.config.id2label[i] for i in range(len(self.net.config.id2label))]
        self.stride = 32
        self.mean = torch.tensor(processor.image_mean, device=self.device)
        self.std = torch.tensor(processor.image_std, device=self.device)

    # ----------------------------------------------------------------- scores
    @torch.no_grad()
    def scores(self, crops: torch.Tensor) -> torch.Tensor:
        """(B, 3, h, w) RGB in [0, 1] -> (B, C, h, w) class scores (probabilities)."""
        x = ((crops - self.mean[:, None, None]) / self.std[:, None, None]).to(self.dtype)
        h, w = x.shape[-2:]
        family = self.spec.family
        if family == "yolo":
            out = self.net(x)
            logits = out[0] if isinstance(out, list | tuple) else out
            probs = logits.float().softmax(1)
        elif family == "segformer":
            probs = self.net(pixel_values=x).logits.float().softmax(1)
        else:  # mask classification: class distribution per query x mask per query
            out = self.net(pixel_values=x)
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]  # drop "no object"
            masks = out.masks_queries_logits.float().sigmoid()
            probs = torch.einsum("bqc,bqhw->bchw", cls, masks)
        return F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

    # ----------------------------------------------------------------- whole image
    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, return_scores: bool = False):
        H, W = rgb.shape[:2]
        scale = self.scale_to / H
        resized = cv2.resize(rgb, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).to(self.device).permute(2, 0, 1).float() / 255.0
        window = self.spec.window
        probs = self._sliding(image, window) if window else self._whole(image)
        probs = F.interpolate(probs[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        labels = probs.argmax(0).to(torch.uint8).cpu().numpy()
        return (labels, probs.cpu().numpy()) if return_scores else labels

    def _whole(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        ph, pw = (-h) % self.stride, (-w) % self.stride
        padded = F.pad(image, (0, pw, 0, ph))
        return self.scores(padded[None])[0, :, :h, :w]

    def _sliding(self, image: torch.Tensor, window: int, overlap: float = 1 / 3) -> torch.Tensor:
        h, w = image.shape[-2:]
        ph, pw = max(0, window - h), max(0, window - w)
        image = F.pad(image, (0, pw, 0, ph))
        H, W = image.shape[-2:]
        step = int(window * (1 - overlap))
        ys = sorted({*range(0, H - window, step), H - window})
        xs = sorted({*range(0, W - window, step), W - window})
        total = torch.zeros(len(self.classes), H, W, device=self.device)
        count = torch.zeros(1, H, W, device=self.device)
        corners = [(y, x) for y in ys for x in xs]
        for start in range(0, len(corners), self.batch):
            chunk = corners[start : start + self.batch]
            crops = torch.stack([image[:, y : y + window, x : x + window] for y, x in chunk])
            for crop_scores, (y, x) in zip(self.scores(crops), chunk, strict=True):
                total[:, y : y + window, x : x + window] += crop_scores
                count[:, y : y + window, x : x + window] += 1
        return (total / count)[:, :h, :w]
