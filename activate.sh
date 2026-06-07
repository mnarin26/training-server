#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAINING_SERVER_ROOT="$ROOT"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
echo "MVQC Training Server aktif: $ROOT"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'yok')"
echo "  CUDA: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'bilinmiyor')"
