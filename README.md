# MVQC Training Server

ROI başına çok sınıflı (EMPTY / FILLED / defect) model eğiten GPU sunucusu.
CM5 istasyonu eğitim yapmaz; buradan üretilen ONNX bundle'ları import eder.

**Tam kullanım kılavuzu:** [KULLANIM.md](KULLANIM.md) · arayüzde **Help** sekmesi

## Hızlı başlangıç

```bash
cd /home/ubuntu/training/training-server
bash start.sh
# Tarayıcı: http://localhost:8501
```

## Ürün klasörü düzeni

```
<ürün>/surface_<i>/<EMPTY|FILLED>/roi_<idx>/*.jpg
```

Ürünleri `/home/ubuntu/training/` altına koy; arayüzden seç ve eğit.
