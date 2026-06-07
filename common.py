"""Shared helpers for the training server: paths, checksums, input spec, manifest.

Kept dependency-light (numpy/onnx only) so ingest/dataset/deploy steps run
without PyTorch; training/export import torch on demand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Canonical V1 preprocessing contract shared with the station.
DEFAULT_INPUT_SPEC: Dict = {
    "layout": "NCHW",
    "size": [224, 224],
    "color": "RGB",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "scale": 1.0 / 255.0,
}

BASE_CLASSES = ["EMPTY", "FILLED"]
CLASSES = BASE_CLASSES  # backward-compatible default for binary ROI models
INPUT_SIZE = int(DEFAULT_INPUT_SPEC["size"][0])


def discover_classes(roi_dir: Path) -> List[str]:
    """Discover class labels from subdirectories under an ROI lake folder.

    Order contract: EMPTY, FILLED, then remaining labels alphabetically.
    """
    if not roi_dir.exists():
        return list(BASE_CLASSES)
    labels = sorted(
        d.name.upper()
        for d in roi_dir.iterdir()
        if d.is_dir() and any(d.glob("*.jpg"))
    )
    if not labels:
        return list(BASE_CLASSES)
    ordered: List[str] = []
    for base in BASE_CLASSES:
        if base in labels:
            ordered.append(base)
    for lbl in labels:
        if lbl not in ordered:
            ordered.append(lbl)
    return ordered


def input_spec(size: int = INPUT_SIZE) -> Dict:
    spec = dict(DEFAULT_INPUT_SPEC)
    spec["size"] = [size, size]
    return spec


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelEntry:
    surface_index: int
    roi_index: int
    roi_name: Optional[str]
    version: str
    onnx_file: str
    classes: List[str]
    input_spec: Dict
    checksum_sha256: str
    metrics: Optional[Dict] = None
    training_run_id: Optional[str] = None


@dataclass
class Manifest:
    product_name: str
    product_barcode: Optional[str] = None
    recipe_version: Optional[int] = None
    bundle_version: str = "1.0"
    created_at: Optional[str] = None
    source: str = "manual_usb"
    models: List[ModelEntry] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)
