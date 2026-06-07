"""Monthly retrain orchestrator for one product."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(args: list[str]) -> dict:
    print("+", " ".join(args))
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        raise SystemExit(f"step failed: {args[1] if len(args) > 1 else args}")
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"stdout": out.stdout}


def discover_rois(lake: Path, product: str):
    base = lake / product
    for surface_dir in sorted(base.glob("surface_*")):
        s_idx = int(surface_dir.name.split("_")[1])
        for roi_dir in sorted(surface_dir.glob("roi_*")):
            r_idx = int(roi_dir.name.split("_")[1])
            yield s_idx, r_idx, roi_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", required=True)
    ap.add_argument("--product", required=True)
    ap.add_argument("--barcode", default=None)
    ap.add_argument("--recipe-version", type=int, default=1)
    ap.add_argument("--workdir", default=str(HERE / "data" / "work"))
    ap.add_argument("--registry", default=str(HERE / "registry"))
    ap.add_argument("--out", default=str(HERE / "data" / "bundles"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--force-register", action="store_true")
    args = ap.parse_args()

    lake = Path(args.lake)
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    results = []
    for s_idx, r_idx, roi_dir in discover_rois(lake, args.product):
        tag = f"{args.product}_s{s_idx}_roi{r_idx}"
        split = work / f"{tag}.json"
        model_dir = work / tag
        onnx = model_dir / "model.onnx"

        _run([PY, str(HERE / "datasets" / "build_dataset.py"),
              "--roi-dir", str(roi_dir), "--out", str(split)])
        _run([PY, str(HERE / "training" / "train_roi.py"),
              "--split", str(split), "--out", str(model_dir),
              "--epochs", str(args.epochs), "--run-id", tag])
        export_info = _run([PY, str(HERE / "export" / "export_onnx.py"),
                            "--ckpt", str(model_dir / "model.pt"), "--out", str(onnx)])
        spec = export_info.get("input_spec") or {}

        reg_args = [PY, str(HERE / "registry" / "registry.py"),
                    "--registry", args.registry, "--product", args.product,
                    "--surface", str(s_idx), "--roi", str(r_idx),
                    "--onnx", export_info.get("onnx", str(onnx)),
                    "--metrics", str(model_dir / "metrics.json"),
                    "--input-spec", json.dumps(spec), "--run-id", tag]
        if args.force_register:
            reg_args.append("--force")
        results.append(_run(reg_args))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{args.product}_{datetime.date.today().isoformat()}.zip"
    bundle_info = _run([PY, str(HERE / "deploy" / "build_bundle.py"),
                        "--registry", args.registry, "--product", args.product,
                        "--recipe-version", str(args.recipe_version),
                        *(["--barcode", args.barcode] if args.barcode else []),
                        "--out", str(bundle)])
    print(json.dumps({"rois": len(results), "bundle": bundle_info}, indent=2))


if __name__ == "__main__":
    main()
