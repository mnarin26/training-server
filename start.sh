#!/usr/bin/env bash
# MVQC eğitim arayüzünü başlatır.
# Bilgisayarı açınca sadece bunu çalıştır:
#   cd /home/ubuntu/training/training-server && bash start.sh

set -euo pipefail
cd "$(dirname "$0")"

echo "==> MVQC Eğitim Sunucusu başlatılıyor..."

# 1) Sanal ortam yoksa oluştur.
if [ ! -f .venv/bin/activate ]; then
  echo "==> Sanal ortam oluşturuluyor (.venv)..."
  if command -v virtualenv &>/dev/null; then
    virtualenv .venv
  else
    python3 -m venv .venv 2>/dev/null || {
      echo "HATA: python3-venv kurulu değil. Çalıştır: sudo apt install python3.10-venv"
      exit 1
    }
  fi
fi
source .venv/bin/activate

# 2) Temel paketler eksikse kur.
if ! python -c "import torch" 2>/dev/null; then
  echo "==> PyTorch (CUDA) kuruluyor... (ilk seferde uzun sürebilir)"
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install onnx onnxruntime numpy Pillow albumentations scikit-learn PyYAML
fi

if ! python -c "import streamlit" 2>/dev/null; then
  echo "==> Streamlit kuruluyor..."
  pip install "streamlit==1.39.0"
fi

# 3) Streamlit ön yüz dosyalarını doğrula.
STATIC="$(python -c 'import streamlit, os; print(os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html"))')"
if [ ! -f "$STATIC" ]; then
  echo "==> Streamlit ön yüz dosyaları eksik, yeniden kuruluyor..."
  pip install --force-reinstall --no-cache-dir "streamlit==1.39.0"
fi

# 4) Veri dizinlerini oluştur.
mkdir -p data/{lake,datasets,models,work,bundles} registry

# 5) Eski süreci kapat.
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

# 6) Arayüzü başlat.
echo ""
echo "  Arayüz: http://localhost:8501"
echo "  Kapatmak için: Ctrl+C"
echo ""
exec streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false
