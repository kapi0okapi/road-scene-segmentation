"""Labels, metrics, the shared inference protocol and the dataset layout, without weights."""

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from road_seg.datasets import joint, kitti
from road_seg.datasets.common import Sample, load, write_ultralytics
from road_seg.evaluate import evaluate
from road_seg.labels import CLASSES, GUARD_RAIL_ID, IGNORE, colorize, to_train_ids
from road_seg.metrics import ConfusionMatrix
from road_seg.models import ModelSpec, Segmenter


class StripeModel(Segmenter):
    """Scores from image content: class = red channel // 32, so the answer is known everywhere."""

    def __init__(self, window: int, scale_to: int):
        self.name, self.spec = "stripes", ModelSpec("fake", "", window)
        self.device, self.dtype, self.scale_to, self.batch = torch.device("cpu"), torch.float32, scale_to, 2
        self.classes, self.stride = CLASSES[:8], 32
        self.mean, self.std = torch.zeros(3), torch.ones(3)
        self.calls = []

    def scores(self, crops):
        self.calls.append(tuple(crops.shape))
        label = (crops[:, 0] * 255 // 32).long().clamp(max=7)
        return torch.nn.functional.one_hot(label, 8).permute(0, 3, 1, 2).float()


def stripes(h=375, w=1242):
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (np.arange(w) // (w / 8) * 32 + 16).astype(np.uint8)[None]  # 8 vertical bands
    return rgb, (np.arange(w) // (w / 8)).astype(np.uint8)[None].repeat(h, 0)


def test_label_ids_map_to_train_ids():
    ids = np.array([[7, 26, GUARD_RAIL_ID, 0, 33]], np.uint8)
    assert to_train_ids(ids).tolist() == [[0, 13, IGNORE, IGNORE, 18]]
    assert to_train_ids(ids, guardrail=True).tolist() == [[0, 13, 19, IGNORE, 18]]
    assert colorize(to_train_ids(ids))[0, 3].tolist() == [0, 0, 0]


def test_confusion_matrix_iou():
    cm = ConfusionMatrix(3)
    truth = np.array([0, 0, 1, 1, 2, IGNORE])
    cm.update(truth, np.array([0, 1, 1, 1, 2, 0]))
    np.testing.assert_allclose(cm.iou(), [0.5, 2 / 3, 1.0])
    assert cm.miou() == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert cm.pixel_accuracy() == pytest.approx(4 / 5)


@pytest.mark.parametrize(("window", "scale_to"), [(0, 375), (0, 512), (256, 512), (1024, 1024)])
def test_protocol_returns_input_resolution_labels(window, scale_to):
    rgb, truth = stripes()
    model = StripeModel(window, scale_to)
    labels = model(rgb)
    assert labels.shape == truth.shape
    inner = np.abs(np.diff(truth, axis=1, prepend=truth[:, :1])) == 0  # ignore pixels at band edges
    assert (labels == truth)[inner].mean() > 0.99
    if window:
        assert all(c[-2:] == (window, window) for c in model.calls) and sum(c[0] for c in model.calls) > 1


def test_evaluate_scores_a_perfect_model(tmp_path):
    rgb, truth = stripes(64, 200)
    Image.fromarray(rgb).save(tmp_path / "a.png")
    label_ids = np.array([7, 8, 11, 12, 13, 17, 19, 20], np.uint8)[truth]  # train ids 0..7
    Image.fromarray(label_ids).save(tmp_path / "a_label.png")
    report = evaluate(StripeModel(0, 64), [Sample(tmp_path / "a.png", tmp_path / "a_label.png")], False)
    assert report["miou"] > 0.97 and report["images"] == 1


def test_kitti_layout_for_ultralytics(tmp_path):
    root = tmp_path / "kitti"
    (root / "training" / "image_2").mkdir(parents=True)
    (root / "training" / "semantic").mkdir()
    for i in range(200):
        name = f"{i:06d}_10.png"
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "training" / "image_2" / name)
        Image.fromarray(np.full((8, 8), 14, np.uint8)).save(root / "training" / "semantic" / name)
    parts = kitti.split(root)
    assert len(parts["train"]) == 160 and parts["val"][0].image.name == "000160_10.png"
    path = write_ultralytics(parts, tmp_path / "yolo20", guardrail=True)
    data = yaml.safe_load(path.read_text())
    assert data["names"][19] == "guard rail" and data["label_mapping"][GUARD_RAIL_ID] == 19
    assert data["label_mapping"][0] == "ignore_label" and data["label_mapping"][7] == 0
    assert (tmp_path / "yolo20" / "masks" / "val" / "000199_10.png").exists()
    _, labels = load(parts["val"][0], guardrail=True)
    assert (labels == 19).all()


def test_joint_dataset_trains_on_both_and_validates_on_kitti(tmp_path):
    def dataset(name, n):
        samples = []
        for i in range(n):
            image, label = tmp_path / name / "src" / f"{i}.png", tmp_path / name / "src" / f"{i}_l.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image)
            Image.fromarray(np.full((8, 8), 14, np.uint8)).save(label)
            samples.append(Sample(image, label))
        return write_ultralytics(
            {"train": samples, "val": samples[:1]}, tmp_path / name / "yolo20", True
        ).parent

    path = joint.write_joint(dataset("cs", 3), dataset("kitti", 2), tmp_path / "joint", kitti_repeats=2)
    data = yaml.safe_load(path.read_text())
    assert len(data["train"]) == 3 and data["train"][1] == data["train"][2]
    assert data["val"].endswith("kitti/yolo20/images/val") and data["names"][19] == "guard rail"
