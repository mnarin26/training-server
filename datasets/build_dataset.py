"""Build a stratified dataset split for a single ROI from the lake.

Class folders under the ROI directory are discovered dynamically:
  EMPTY, FILLED, plus any defect labels (e.g. EZIK, CIZIK).

Usage:
    python build_dataset.py --roi-dir ../data/lake/Product_A/surface_1/roi_1 \
        --out ../data/datasets/Product_A_s1_roi1.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import discover_classes  # noqa: E402


def _split(items: List[str], val: float, test: float, seed: int) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test)
    n_val = int(n * val)
    return items[n_test + n_val:], items[n_test:n_test + n_val], items[:n_test]


def build(roi_dir: Path, val: float = 0.15, test: float = 0.15, seed: int = 42) -> Dict:
    classes = discover_classes(roi_dir)
    split = {"classes": classes, "roi_dir": str(roi_dir), "train": [], "val": [], "test": [], "counts": {}}
    for label in classes:
        label_dir = roi_dir / label
        files = sorted(str(p) for p in label_dir.glob("*.jpg")) if label_dir.exists() else []
        split["counts"][label] = len(files)
        tr, va, te = _split(files, val, test, seed)
        for name, subset in (("train", tr), ("val", va), ("test", te)):
            split[name].extend([{"path": f, "label": label} for f in subset])
    for name in ("train", "val", "test"):
        random.Random(seed).shuffle(split[name])
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()
    split = build(Path(args.roi_dir), args.val, args.test)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(split, indent=2))
    print(json.dumps({"out": args.out, "classes": split["classes"], "counts": split["counts"],
                      "train": len(split["train"]), "val": len(split["val"]),
                      "test": len(split["test"])}, indent=2))


if __name__ == "__main__":
    main()
