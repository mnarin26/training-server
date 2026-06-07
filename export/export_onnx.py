"""Export a trained ROI checkpoint to ONNX (opset 17), verify parity, optional INT8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import CLASSES, INPUT_SIZE, input_spec, sha256_file  # noqa: E402

OPSET = 17


def export(ckpt_path: Path, out_path: Path, quantize: bool) -> dict:
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small

    blob = torch.load(ckpt_path, map_location="cpu")
    classes = blob.get("classes", CLASSES)
    spec = blob.get("input_spec", input_spec())
    size = spec["size"][0]

    model = mobilenet_v3_small()
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, size, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        opset_version=OPSET, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    parity = _parity_check(model, out_path, dummy)

    final_path = out_path
    if quantize:
        final_path = _quantize(out_path)

    return {
        "onnx": str(final_path),
        "classes": classes,
        "input_spec": spec,
        "opset": OPSET,
        "parity_max_abs_diff": parity,
        "checksum_sha256": sha256_file(final_path),
        "quantized": quantize,
    }


def _parity_check(torch_model, onnx_path: Path, dummy) -> float:
    import onnxruntime as ort
    import torch

    with torch.no_grad():
        torch_out = torch_model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["output"], {"input": dummy.numpy()})[0]
    diff = float(np.max(np.abs(torch_out - onnx_out)))
    if diff > 1e-3:
        print(f"WARNING: ONNX parity diff {diff:.2e} exceeds 1e-3")
    return diff


def _quantize(onnx_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = onnx_path.with_suffix(".int8.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    return q_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantize", action="store_true")
    args = ap.parse_args()
    info = export(Path(args.ckpt), Path(args.out), args.quantize)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
