"""On-disk versioned model registry + acceptance gate + model cards."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import sha256_file  # noqa: E402

GATE = {"FILLED_recall": 0.99, "EMPTY_recall": 0.97}


def acceptance_gate(metrics: Dict) -> Dict:
    recall = (metrics.get("test", {}) or {}).get("recall", {})
    reasons = []
    if recall.get("FILLED", 0.0) < GATE["FILLED_recall"]:
        reasons.append(f"FILLED recall {recall.get('FILLED', 0):.3f} < {GATE['FILLED_recall']}")
    if recall.get("EMPTY", 0.0) < GATE["EMPTY_recall"]:
        reasons.append(f"EMPTY recall {recall.get('EMPTY', 0):.3f} < {GATE['EMPTY_recall']}")
    return {"passed": not reasons, "reasons": reasons}


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _roi_dir(self, product: str, surface_index: int, roi_index: int) -> Path:
        return self.root / product / f"surface_{surface_index}" / f"roi_{roi_index}"

    def register(self, *, product: str, surface_index: int, roi_index: int,
                 roi_name: Optional[str], onnx_path: Path, classes, input_spec: Dict,
                 metrics: Dict, run_id: str, force: bool = False) -> Dict:
        gate = acceptance_gate(metrics)
        if not gate["passed"] and not force:
            return {"status": "rejected", "gate": gate}

        version = datetime.now().strftime("%Y.%m.%d-%H%M%S")
        dest = self._roi_dir(product, surface_index, roi_index) / version
        dest.mkdir(parents=True, exist_ok=True)
        model_dest = dest / "model.onnx"
        shutil.copy2(onnx_path, model_dest)

        card = {
            "product": product, "surface_index": surface_index, "roi_index": roi_index,
            "roi_name": roi_name, "version": version, "classes": classes,
            "input_spec": input_spec, "metrics": metrics, "training_run_id": run_id,
            "checksum_sha256": sha256_file(model_dest),
            "gate": gate, "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (dest / "card.json").write_text(json.dumps(card, indent=2))
        (self._roi_dir(product, surface_index, roi_index) / "latest").write_text(version)
        return {"status": "registered", "version": version, "path": str(model_dest), "card": card}


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a trained+exported ONNX model")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--surface", type=int, required=True)
    ap.add_argument("--roi", type=int, required=True)
    ap.add_argument("--roi-name", default=None)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--input-spec", required=True)
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    spec_arg = args.input_spec
    spec = json.loads(Path(spec_arg).read_text()) if Path(spec_arg).exists() else json.loads(spec_arg)
    counts = metrics.get("counts") or {}
    if counts:
        classes = []
        for base in ("EMPTY", "FILLED"):
            if base in counts:
                classes.append(base)
        for lbl in sorted(counts.keys()):
            if lbl not in classes:
                classes.append(lbl)
    else:
        classes = ["EMPTY", "FILLED"]
    reg = ModelRegistry(Path(args.registry))
    result = reg.register(
        product=args.product, surface_index=args.surface, roi_index=args.roi,
        roi_name=args.roi_name, onnx_path=Path(args.onnx), classes=classes,
        input_spec=spec, metrics=metrics, run_id=args.run_id, force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
