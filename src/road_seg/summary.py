"""Markdown tables from benchmark/evaluation JSON files.

seg-summary --runs runs/benchmark runs/benchmark_ft19 --out runs/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FOCUS = ("road", "sidewalk", "fence", "pole", "traffic sign", "vegetation", "car", "guard rail")


def load(dirs: list[Path]) -> list[dict]:
    reports = []
    for d in dirs:
        for path in sorted(d.glob("*.json")):
            report = json.loads(path.read_text())
            model = report["model"]
            report["label"] = f"{Path(model).stem} ({d.name})" if model.endswith(".pt") else model
            reports.append(report)
    return reports


def table(reports: list[dict]) -> str:
    focus = " | ".join(FOCUS)
    head = f"| Model | Cityscapes val mIoU (paper) | KITTI val mIoU | {focus} | Scale | ms/image |"
    lines = [head, "|---" * (len(FOCUS) + 6) + "|"]
    for r in sorted(reports, key=lambda r: r["miou"]):
        paper = r.get("paper_cityscapes_miou")
        cells = [
            r["label"],
            f"{paper:.1f}" if paper else "–",
            f"{100 * r['miou']:.1f}",
            *[f"{100 * r['iou'][c]:.1f}" if r["iou"].get(c) is not None else "–" for c in FOCUS],
            str(r.get("scale_to", "–")),
            f"{r['ms_per_image']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text = table(load(args.runs))
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
