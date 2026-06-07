"""Assemble a deployable model bundle ZIP for a product from the registry."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import Manifest, ModelEntry, sha256_file  # noqa: E402


def _discover(registry: Path, product: str) -> List[dict]:
    entries = []
    product_dir = registry / product
    if not product_dir.exists():
        return entries
    for surface_dir in sorted(product_dir.glob("surface_*")):
        s_idx = int(surface_dir.name.split("_")[1])
        for roi_dir in sorted(surface_dir.glob("roi_*")):
            r_idx = int(roi_dir.name.split("_")[1])
            ptr = roi_dir / "latest"
            if not ptr.exists():
                continue
            version = ptr.read_text().strip()
            card_file = roi_dir / version / "card.json"
            onnx = roi_dir / version / "model.onnx"
            if not (card_file.exists() and onnx.exists()):
                continue
            card = json.loads(card_file.read_text())
            entries.append({"surface_index": s_idx, "roi_index": r_idx, "onnx": onnx, "card": card})
    return entries


def build(registry: Path, product: str, out_zip: Path, barcode: str | None,
          recipe_version: int | None) -> dict:
    entries = _discover(registry, product)
    if not entries:
        raise SystemExit(f"no registered models found for product '{product}'")

    manifest = Manifest(
        product_name=product, product_barcode=barcode, recipe_version=recipe_version,
        created_at=datetime.now().isoformat(timespec="seconds"), source="manual_usb",
    )

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for e in entries:
            card = e["card"]
            arc = f"models/surface_{e['surface_index']}_roi_{e['roi_index']}.onnx"
            zf.write(e["onnx"], arc)
            manifest.models.append(ModelEntry(
                surface_index=e["surface_index"], roi_index=e["roi_index"],
                roi_name=card.get("roi_name"), version=card["version"], onnx_file=arc,
                classes=card["classes"], input_spec=card["input_spec"],
                checksum_sha256=card["checksum_sha256"], metrics=card.get("metrics"),
                training_run_id=card.get("training_run_id"),
            ))
        zf.writestr("manifest.json", manifest.to_json())

    return {"bundle": str(out_zip), "models": len(manifest.models),
            "checksum_sha256": sha256_file(out_zip)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--barcode", default=None)
    ap.add_argument("--recipe-version", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    info = build(Path(args.registry), args.product, Path(args.out), args.barcode, args.recipe_version)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
