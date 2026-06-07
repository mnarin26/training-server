"""Help tab içeriği — MVQC Training Server."""

HELP_MARKDOWN = """
## MVQC Eğitim Sunucusu — Kullanım Kılavuzu

Bu sunucu (Ubuntu + NVIDIA GPU), CM5 istasyonundan toplanan öğrenim resimlerini kullanarak
**ROI başına çok sınıflı sınıflandırıcı** modelleri eğitir. Çıktı: istasyona import edilecek **bundle ZIP**.

---

### Hızlı başlangıç

```bash
cd /home/ubuntu/training/training-server
bash start.sh
```

Tarayıcı: **http://localhost:8501** — **Eğitim** sekmesinden ürün seç, **Eğitimi Başlat**.

---

### Sekmeler

| Sekme | Açıklama |
|-------|----------|
| **Eğitim** | Ürün seçimi, özet, pipeline başlatma, log, bundle indirme |
| **Help** | Bu kılavuz |

---

### Ürün klasör yapısı

```
<ürün_adı>/
└── surface_<numara>/
    ├── EMPTY/
    │   └── roi_<numara>/*.jpg
    ├── FILLED/
    │   └── roi_<numara>/*.jpg
    ├── TERS/          ← defect (hata) sınıfı — isteğe bağlı, ROI başına
    │   └── roi_2/*.jpg
    └── EZIK/
        └── roi_1/*.jpg
```

**Kurallar**

- Ürün klasörü `Ürünler kök dizini` altında olmalı (varsayılan: `/home/ubuntu/training/`)
- `EMPTY` / `FILLED` → her ROI için gerekli (var/yok öğrenimi)
- Defect klasörleri (`TERS`, `EZIK`, …) → yalnızca o ROI'de tanımlı ve örnek çekilmiş sınıflar
- CM5'te defect capture yalnızca ilgili ROI'ye yazar; klasör adı = büyük harf etiket

---

### Kenar çubuğu ayarları

| Ayar | Varsayılan | Açıklama |
|------|------------|----------|
| **Ürünler kök dizini** | `../` (training/) | `surface_*` içeren klasörler ürün sayılır |
| **Veri gölü (lake)** | `data/lake` | Ingest sonrası kalıcı arşiv; hash ile tekrar eklenmez |
| **Registry** | `registry` | Kayıtlı ONNX sürümleri + `card.json` metrikleri |
| **Bundle çıktı** | `data/bundles` | CM5'e gidecek ZIP (`ürün_tarih.zip`) |
| **Epoch sayısı** | 15 | ROI modeli eğitim turu (1=test, 15=normal, 30+=daha iyi) |
| **Barkod** | boş | `manifest.json`'a yazılır; CM5 import'ta ürün eşleştirme |
| **Reçete sürümü** | 1 | ROI yapısı değişince artır |
| **Kabul kapısını atla** | açık | Test için açık; üretimde kapat (bkz. aşağı) |

---

### Eğitim sekmesi — adım adım

1. **Tanıtılacak ürün** — açılır listeden seç (ör. `deneme_a`)
2. **Özet kartları** — yüzey, ROI, toplam resim sayısı
3. **Detaylı yapı** — her ROI için sınıf başına resim sayısı (EMPTY, FILLED, defect'ler)
4. **Eğitimi Başlat** — pipeline çalışır; log canlı akar
5. **Bundle indir** — bitince ZIP indir → USB veya ağ ile CM5 **Models** sekmesine

---

### Pipeline (arka planda ne olur?)

```
Ürün klasörü
    → INGEST (lake'e kopyala, dedup)
    → BUILD DATASET (train/val/test %70/15/15, sınıflar otomatik keşif)
    → TRAIN (MobileNetV3-Small, GPU, ROI başına)
    → EXPORT ONNX (opset 17)
    → REGISTRY (kabul kapısı + card.json; classes manifest'e yazılır)
    → BUILD BUNDLE (manifest.json + models/*.onnx)
```

Her `roi_*` klasörü lake'te varsa ayrı model eğitilir. **CM5'te olmayan roi_6** gibi fazlalar bundle'a girer ama import'ta *skipped* olur — lake/registry'den silip yeniden eğit veya CM5'e ROI ekle.

---

### Çok sınıflı / defect modelleri

- Sınıf listesi ROI klasöründeki alt klasörlerden otomatik: `EMPTY`, `FILLED`, sonra alfabetik defect'ler
- Registry `card.json` ve bundle `manifest.json` içindeki `classes` dizisi eğitilen tüm sınıfları taşır
- CM5 karar mantığı: `FILLED`→OK, `EMPTY`→HATA, defect+ERROR→HATA, defect+WARNING→UYARI

Defect modeli için CM5'te örnek çek → export → buraya kopyala → eğit → import.

---

### Çıktılar

| Konum | İçerik |
|-------|--------|
| `data/lake/<ürün>/surface_*/roi_*/SINIF/` | Ingest edilmiş resimler |
| `data/work/` | Geçici split, checkpoint, ONNX |
| `registry/<ürün>/surface_*/roi_*/<sürüm>/` | `model.onnx` + `card.json` |
| `data/bundles/<ürün>_<tarih>.zip` | CM5 import paketi |

**Bundle içeriği örneği**

```
manifest.json
models/surface_1_roi_1.onnx
models/surface_1_roi_2.onnx
...
```

`manifest.json`: ürün adı, barkod, her model için `surface_index`, `roi_index`, `version`, `classes`, `checksum_sha256`.

---

### Kabul kapısı (Acceptance Gate)

| Metrik | Minimum |
|--------|---------|
| FILLED recall | %99 |
| EMPTY recall | %97 |

Geçemezse model `rejected` — bundle'a girmez. **Kabul kapısını atla** işaretliyse test verisiyle de kaydedilir.

---

### CM5 ile birlikte tam akış

1. CM5 **Products** → ürün oluştur
2. CM5 **Teaching** → ROI + defect tanımla, EMPTY/FILLED/defect capture
3. SSD export → bu sunucuda ürün klasörüne kopyala (`/home/ubuntu/training/<ürün>/...`)
4. Burada **Eğitimi Başlat** → bundle indir
5. CM5 **Models** → Import & activate
6. CM5 **Inspect** → canlı denetim

---

### Komut satırı (arayüz yerine)

```bash
cd /home/ubuntu/training/training-server
source .venv/bin/activate

python ingest/ingest.py --product-dir ../deneme_a --product-name deneme_a --lake data/lake

python pipelines/monthly_retrain.py \\
  --lake data/lake --product deneme_a \\
  --registry registry --out data/bundles \\
  --epochs 15 --recipe-version 1 --force-register
```

---

### Sorun giderme

| Sorun | Çözüm |
|-------|--------|
| Ürün listesi boş | Kök dizinde `surface_*` alt klasörü olan ürün klasörü var mı? |
| CUDA False | NVIDIA sürücü + `finish-setup.sh` / `nvidia-smi` |
| Eğitim rejected | Daha fazla resim veya kabul kapısını atla |
| CM5 import skipped roi_6 | CM5'te 6. ROI yok; ROI ekle veya lake/registry'den roi_6 sil |
| Defect tanınmıyor | ROI klasöründe defect alt klasörü ve yeterli resim var mı? Yeniden eğit+import |
| Bundle eski classes | Registry düzeltmesi sonrası **yeniden eğit** gerekir |

---

### Sunucuyu kapatma

Terminalde `Ctrl+C`. Tekrar açmak için `bash start.sh`.
"""
