"""Basit eğitim arayüzü (Streamlit).

Beklenen ürün klasörü düzeni:
    <ürün>/surface_<i>/<SINIF>/roi_<idx>/*.jpg
    SINIF = EMPTY | FILLED | hata etiketleri (ör. EZIK, CIZIK)

Çalıştırma:
    bash start.sh
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from help_content import HELP_MARKDOWN

HERE = Path(__file__).resolve().parent
PY = sys.executable
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def is_product_dir(path: Path) -> bool:
    return path.is_dir() and any(
        p.is_dir() and p.name.startswith("surface_") for p in path.iterdir()
    )


def find_products(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        c for c in root.iterdir()
        if c.is_dir() and not c.name.startswith(".") and is_product_dir(c)
    )


def summarize(product_dir: Path) -> dict:
    summary = {"surfaces": {}, "total_images": 0}
    for surface in sorted(product_dir.glob("surface_*")):
        rois: dict = {}
        for label_dir in sorted(surface.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.upper()
            for roi_dir in sorted(label_dir.glob("roi_*")):
                n = sum(1 for p in roi_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                rois.setdefault(roi_dir.name, {})[label] = n
                summary["total_images"] += n
        summary["surfaces"][surface.name] = rois
    return summary


def stream_command(args: list[str], log_box) -> int:
    lines: list[str] = []
    proc = subprocess.Popen(args, cwd=str(HERE), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-400:]), language="text")
    proc.wait()
    return proc.returncode


st.set_page_config(page_title="MVQC Eğitim", page_icon="🧠", layout="centered")
st.title("🧠 MVQC Eğitim Sunucusu")

tab_train, tab_help = st.tabs(["Eğitim", "Help"])

with tab_help:
    st.markdown(HELP_MARKDOWN)

with tab_train:
    st.caption("Bir ürün seç, tanıt ve modeli eğit.")

    with st.sidebar:
        st.header("Ayarlar")
        products_root = st.text_input("Ürünler kök dizini", value=str(HERE.parent))
        lake_dir = st.text_input("Veri gölü (lake)", value=str(HERE / "data" / "lake"))
        registry_dir = st.text_input("Registry", value=str(HERE / "registry"))
        bundles_dir = st.text_input("Bundle çıktı", value=str(HERE / "data" / "bundles"))
        epochs = st.number_input("Epoch sayısı", min_value=1, max_value=200, value=15)
        barcode = st.text_input("Barkod (opsiyonel)", value="")
        recipe_version = st.number_input("Reçete sürümü", min_value=1, value=1)
        force = st.checkbox("Kabul kapısını atla (--force-register)", value=True)

    root = Path(products_root)
    products = find_products(root)

    if not products:
        st.warning(
            f"`{products_root}` altında ürün klasörü bulunamadı.\n\n"
            "Beklenen düzen: `ürün/surface_<i>/<SINIF>/roi_<idx>/*.jpg` "
            "(SINIF: EMPTY, FILLED veya hata etiketleri)"
        )
        st.stop()

    names = [p.name for p in products]
    choice = st.selectbox("Tanıtılacak ürün", names)
    product_dir = next(p for p in products if p.name == choice)

    summary = summarize(product_dir)
    st.subheader(f"📦 {choice}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Yüzey", len(summary["surfaces"]))
    col2.metric("Toplam ROI", sum(len(r) for r in summary["surfaces"].values()))
    col3.metric("Toplam Resim", summary["total_images"])

    with st.expander("Detaylı yapı", expanded=True):
        for surface, rois in summary["surfaces"].items():
            st.markdown(f"**{surface}**")
            all_labels = sorted({lbl for labels in rois.values() for lbl in labels})
            rows = [{"ROI": roi, **{lbl: labels.get(lbl, 0) for lbl in all_labels}}
                    for roi, labels in rois.items()]
            st.table(rows)

    st.divider()
    start = st.button("🚀 Eğitimi Başlat", type="primary", use_container_width=True)

    if start:
        log_box = st.empty()

        st.info("1/2 — Veri gölüne aktarılıyor (ingest)...")
        rc = stream_command([
            PY, str(HERE / "ingest" / "ingest.py"),
            "--product-dir", str(product_dir),
            "--product-name", choice,
            "--lake", lake_dir,
        ], log_box)
        if rc != 0:
            st.error("Ingest başarısız oldu.")
            st.stop()

        st.info("2/2 — Model eğitiliyor...")
        cmd = [
            PY, str(HERE / "pipelines" / "monthly_retrain.py"),
            "--lake", lake_dir, "--product", choice,
            "--registry", registry_dir, "--out", bundles_dir,
            "--recipe-version", str(int(recipe_version)),
            "--epochs", str(int(epochs)),
        ]
        if barcode.strip():
            cmd += ["--barcode", barcode.strip()]
        if force:
            cmd += ["--force-register"]

        rc = stream_command(cmd, log_box)
        if rc == 0:
            st.success(f"✅ Eğitim tamamlandı. Bundle: {bundles_dir}")
            out = Path(bundles_dir)
            if out.is_dir():
                zips = sorted(out.glob(f"{choice}*.zip"))
                if zips:
                    latest = zips[-1]
                    with open(latest, "rb") as fh:
                        st.download_button("⬇️ Bundle indir", fh.read(),
                                           file_name=latest.name, mime="application/zip")
        else:
            st.error("Eğitim başarısız oldu.")
