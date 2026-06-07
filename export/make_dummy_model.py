"""Generate a tiny brightness-threshold ONNX presence model (placeholder)."""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = ["EMPTY", "FILLED"]


def build(size: int = 64) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="half")
    gain = numpy_helper.from_array(np.array([12.0], dtype=np.float32), name="gain")
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="unsq_axes")

    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], keepdims=0, axes=[0, 1, 2, 3]),
        helper.make_node("Sub", ["mean", "half"], ["d"]),
        helper.make_node("Mul", ["d", "gain"], ["filled"]),
        helper.make_node("Neg", ["filled"], ["empty"]),
        helper.make_node("Concat", ["empty", "filled"], ["logits"], axis=0),
        helper.make_node("Unsqueeze", ["logits", "unsq_axes"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "presence_dummy", [x], [out], initializer=[half, gain, axes]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    model = build(args.size)
    onnx.save(model, args.out)
    print(json.dumps({"path": args.out, "classes": CLASSES}))


if __name__ == "__main__":
    main()
