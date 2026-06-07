"""Ingest SSD-3 data (teaching samples + field ROI archives) into a data lake.

Input sources (copied off the station via USB now, network later):
  - teaching/<Product>/surface_<i>/<EMPTY|FILLED>/roi_<idx>/*.jpg
  - roi_archive/inspection_xxxxxx/{roi_*.jpg, metadata.json}

Output data lake (per-ROI, per-label), deduplicated by content hash:
  data/lake/<Product>/surface_<i>/roi_<idx>/<EMPTY|FILLED>/<sha>.jpg

Usage:
    python ingest.py --src /media/usb/2026-08-01 --lake ../data/lake
    python ingest.py --product-dir /path/to/deneme_a --lake ../data/lake
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import sha256_file  # noqa: E402


def _copy_dedup(src: Path, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(src)
    dest = dest_dir / f"{digest}.jpg"
    if dest.exists():
        return False
    shutil.copy2(src, dest)
    return True


def ingest_teaching(teaching_root: Path, lake: Path) -> int:
    count = 0
    for img in teaching_root.rglob("*.jpg"):
        parts = img.relative_to(teaching_root).parts
        if len(parts) < 5:
            continue
        product, surface, label, roi = parts[0], parts[1], parts[2], parts[3]
        dest = lake / product / surface / roi / label.upper()
        if _copy_dedup(img, dest):
            count += 1
    return count


def ingest_product(product_dir: Path, lake: Path, product_name: str | None = None) -> int:
    """Ingest a single product folder into the lake.

    Expected source layout:
        <product_dir>/surface_<i>/<EMPTY|FILLED>/roi_<idx>/*.jpg
    Lake layout:
        <lake>/<product>/surface_<i>/roi_<idx>/<EMPTY|FILLED>/<sha>.jpg
    """
    product = product_name or product_dir.name
    count = 0
    for img in product_dir.rglob("*.jpg"):
        parts = img.relative_to(product_dir).parts
        if len(parts) < 4:
            continue
        surface, label, roi = parts[0], parts[1], parts[2]
        dest = lake / product / surface / roi / label.upper()
        if _copy_dedup(img, dest):
            count += 1
    return count


def ingest_roi_archive(archive_root: Path, lake: Path) -> int:
    count = 0
    for meta_file in archive_root.rglob("metadata.json"):
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue
        product = meta.get("product", "unknown")
        surface = f"surface_{meta.get('surface', 1)}"
        roi_results = meta.get("roi_results", {})
        insp_dir = meta_file.parent
        for key, label in roi_results.items():
            idx = key.replace("roi", "")
            roi = f"roi_{idx}"
            img = insp_dir / f"roi_{idx}.jpg"
            if not img.exists():
                continue
            dest = lake / product / surface / roi / str(label).upper()
            if _copy_dedup(img, dest):
                count += 1
    return count


def extract_zip_sources(src: Path, work: Path) -> Path:
    if src.is_file() and src.suffix == ".zip":
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(work)
        return work
    return src


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest SSD-3 data into the data lake")
    ap.add_argument("--src", help="SSD-3 root, a daily export dir, or a .zip")
    ap.add_argument("--product-dir", help="Single product folder: <dir>/surface_<i>/<LABEL>/roi_<idx>/*.jpg")
    ap.add_argument("--product-name", default=None, help="Override product name (default: product-dir name)")
    ap.add_argument("--lake", default=str(Path(__file__).resolve().parents[1] / "data" / "lake"))
    args = ap.parse_args()

    lake = Path(args.lake)
    lake.mkdir(parents=True, exist_ok=True)

    if args.product_dir:
        pdir = Path(args.product_dir)
        n = ingest_product(pdir, lake, args.product_name)
        print(json.dumps({"ingested_product": n, "product": args.product_name or pdir.name,
                          "lake": str(lake)}, indent=2))
        return

    if not args.src:
        ap.error("either --src or --product-dir is required")

    src = Path(args.src)
    work = lake.parent / "_ingest_tmp"
    root = extract_zip_sources(src, work)

    teaching_root = root / "teaching" if (root / "teaching").exists() else root
    archive_root = root / "roi_archive" if (root / "roi_archive").exists() else root

    t = ingest_teaching(teaching_root, lake) if teaching_root.exists() else 0
    a = ingest_roi_archive(archive_root, lake) if archive_root.exists() else 0

    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    print(json.dumps({"ingested_teaching": t, "ingested_field": a, "lake": str(lake)}, indent=2))


if __name__ == "__main__":
    main()
