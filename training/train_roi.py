"""Train one per-ROI EMPTY/FILLED classifier (MobileNetV3-Small) on the GPU.

Reads a dataset split JSON (from datasets/build_dataset.py), trains with
exposure/lighting-style augmentation, and writes a checkpoint + metrics.

Usage:
    python train_roi.py --split ../data/datasets/Product_A_s1_roi1.json \
        --out ../data/models/Product_A_s1_roi1 --epochs 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import BASE_CLASSES, INPUT_SIZE, input_spec  # noqa: E402


def _load_image(path: str, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img, dtype=np.uint8)


def _augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lightweight exposure/lighting + geometry augmentation without heavy deps."""
    x = img.astype(np.float32)
    x *= rng.uniform(0.6, 1.5)
    x = 255.0 * np.power(np.clip(x / 255.0, 0, 1), rng.uniform(0.7, 1.4))
    x += rng.normal(0, rng.uniform(0, 8), x.shape)
    if rng.random() < 0.5:
        x = x[:, ::-1, :]
    return np.clip(x, 0, 255).astype(np.uint8)


def _to_tensor(img: np.ndarray) -> np.ndarray:
    spec = input_spec()
    arr = img.astype(np.float32) * spec["scale"]
    arr = (arr - np.array(spec["mean"], dtype=np.float32)) / np.array(spec["std"], dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))


class RoiDataset:
    def __init__(self, items: List[Dict], classes: List[str], size: int, train: bool, seed: int = 0):
        self.items = items
        self.classes = classes
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def get(self, i: int) -> Tuple[np.ndarray, int]:
        it = self.items[i]
        img = _load_image(it["path"], self.size)
        if self.train:
            img = _augment(img, self.rng)
        return _to_tensor(img), self.classes.index(it["label"])


def train(split_path: Path, out_dir: Path, epochs: int, batch: int, lr: float, run_id: str) -> Dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    split = json.loads(split_path.read_text())
    classes = split.get("classes") or BASE_CLASSES
    size = INPUT_SIZE

    class TorchDS(Dataset):
        def __init__(self, items, is_train):
            self.ds = RoiDataset(items, classes, size, is_train)
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            x, y = self.ds.get(i)
            return torch.from_numpy(x), y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(TorchDS(split["train"], True), batch_size=batch, shuffle=True, num_workers=4)
    val_dl = DataLoader(TorchDS(split["val"], False), batch_size=batch, num_workers=2)

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        metrics = evaluate(model, val_dl, device, classes)
        if metrics["accuracy"] >= best_acc:
            best_acc = metrics["accuracy"]
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "arch": "mobilenet_v3_small", "input_spec": input_spec()}, ckpt)
        print(f"epoch {epoch+1}/{epochs} val={metrics}")

    test_dl = DataLoader(TorchDS(split["test"], False), batch_size=batch, num_workers=2)
    test_metrics = evaluate(model, test_dl, device, classes)
    metrics_out = {"run_id": run_id, "best_val_accuracy": best_acc, "test": test_metrics,
                   "counts": split.get("counts", {})}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    return {"checkpoint": str(ckpt), "metrics": metrics_out}


def evaluate(model, dl, device, classes: List[str]) -> Dict:
    import torch

    model.eval()
    tp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            for p, t in zip(pred, y):
                total += 1
                correct += int(p == t)
                cls = classes[t]
                if p == t:
                    tp[cls] += 1
                else:
                    fn[cls] += 1
    recalls = {c: (tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0) for c in classes}
    return {"accuracy": correct / total if total else 0.0,
            "recall": recalls, "n": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--run-id", default="local")
    args = ap.parse_args()
    result = train(Path(args.split), Path(args.out), args.epochs, args.batch, args.lr, args.run_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
