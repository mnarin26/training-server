# MVQC Eğitim Sunucusu — Kullanım Kılavuzu

Bu sunucu, görsel kalite kontrol istasyonundan toplanan öğrenim (teaching) resimlerini
kullanarak **ROI başına EMPTY/FILLED sınıflandırıcı** modelleri eğitir. Eğitim burada
(Ubuntu + NVIDIA GPU) yapılır; istasyon (Raspberry Pi / CM5) sadece hazır ONNX modellerini
import eder.

---

## İçindekiler

1. [Hızlı Başlangıç](#1-hızlı-başlangıç)
2. [Klasör Yapısı](#2-klasör-yapısı)
3. [Ürün Verisi Nasıl Hazırlanır?](#3-ürün-verisi-nasıl-hazırlanır)
4. [Web Arayüzü](#4-web-arayüzü)
5. [Ayarlar (Kenar Çubuğu)](#5-ayarlar-kenar-çubuğu)
6. [Eğitim Akışı (Adım Adım)](#6-eğitim-akışı-adım-adım)
7. [Çıktılar Nereye Yazılır?](#7-çıktılar-nereye-yazılır)
8. [Komut Satırından Kullanım](#8-komut-satırından-kullanım)
9. [Kabul Kapısı (Acceptance Gate)](#9-kabul-kapısı-acceptance-gate)
10. [Sorun Giderme](#10-sorun-giderme)

---

## 1. Hızlı Başlangıç

### Bilgisayarı her açtığında

```bash
cd /home/ubuntu/training/training-server
bash start.sh
```

Tarayıcıda aç: **http://localhost:8501**

Kapatmak için terminalde `Ctrl+C`.

### İlk kurulum (tek seferlik)

Sunucuda NVIDIA sürücüsü ve CUDA kurulu olmalı. Doğrulama:

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Beklenen: CUDA: True
```

`start.sh` ilk çalıştırmada eksik paketleri otomatik kurar (PyTorch, Streamlit vb.).
İlk sefer 5–15 dakika sürebilir; sonraki açılışlar hızlıdır.

---

## 2. Klasör Yapısı

```
/home/ubuntu/training/
├── deneme_a/                  ← ürün klasörü (örnek)
│   └── surface_1/
│       ├── EMPTY/
│       │   ├── roi_1/*.jpg
│       │   └── roi_2/*.jpg
│       └── FILLED/
│           ├── roi_1/*.jpg
│           └── roi_2/*.jpg
├── aa/                        ← başka bir ürün
└── training-server/           ← eğitim sunucusu (bu proje)
    ├── start.sh               ← arayüzü başlat
    ├── app.py                 ← web arayüzü
    ├── data/
    │   ├── lake/              ← ingest sonrası arşiv
    │   ├── work/              ← geçici eğitim dosyaları
    │   └── bundles/           ← istasyona gidecek ZIP'ler
    ├── registry/              ← kayıtlı model sürümleri
    └── .venv/                 ← Python sanal ortamı
```

---

## 3. Ürün Verisi Nasıl Hazırlanır?

Her ürün, `/home/ubuntu/training/` altında kendi klasörüne sahip olmalıdır.

### Beklenen düzen

```
<ürün_adı>/
└── surface_<numara>/
    ├── EMPTY/
    │   └── roi_<numara>/
    │       └── *.jpg
    └── FILLED/
        └── roi_<numara>/
            └── *.jpg
```

### Kurallar

| Kural | Açıklama |
|---|---|
| Ürün adı | Klasör adı = ürün adı (`deneme_a`, `aa`, …) |
| Yüzey | `surface_1`, `surface_2`, … (birden fazla yüzey olabilir) |
| Etiket | `EMPTY` veya `FILLED` (büyük harf) |
| ROI | `roi_1`, `roi_2`, … (her ROI ayrı model alır) |
| Resim | `.jpg`, `.jpeg` veya `.png` |

### Örnek

```
deneme_a/surface_1/EMPTY/roi_1/  → 20 resim
deneme_a/surface_1/FILLED/roi_1/ → 20 resim
```

Yeni ürün eklemek için klasörü `/home/ubuntu/training/` altına koyman yeterli;
arayüz otomatik bulur.

---

## 4. Web Arayüzü

`bash start.sh` sonrası **http://localhost:8501** adresinde açılır.

### Ana ekran

1. **Tanıtılacak ürün** — açılır menüden ürün seç
2. **Özet kartları** — yüzey sayısı, ROI sayısı, toplam resim
3. **Detaylı yapı** — her ROI için EMPTY/FILLED resim sayıları tablosu
4. **Eğitimi Başlat** — pipeline'ı çalıştırır
5. **Canlı log** — ingest ve eğitim çıktısı terminal gibi akar
6. **Bundle indir** — eğitim bitince ZIP indirme butonu çıkar

### Tipik kullanım

```
1. start.sh çalıştır
2. Tarayıcıda localhost:8501 aç
3. Ürünü seç (ör. deneme_a)
4. Özeti kontrol et (resim sayıları doğru mu?)
5. "Eğitimi Başlat"a bas
6. Log bitene kadar bekle
7. "Bundle indir" ile ZIP'i al
8. ZIP'i USB ile istasyona taşı
```

---

## 5. Ayarlar (Kenar Çubuğu)

Sol paneldeki ayarlar eğitim pipeline'ının yollarını ve parametrelerini belirler.

### Ürünler kök dizini
**Varsayılan:** `/home/ubuntu/training`

Ürün klasörlerinin arandığı üst dizin. İçinde `surface_*` alt klasörü olan
her dizin ürün olarak listelenir.

### Veri gölü (lake)
**Varsayılan:** `training-server/data/lake`

Seçilen ürünün resimlerinin kalıcı arşive kopyalandığı yer. Ingest sırasında
kaynak düzen (`surface/LABEL/roi`) lake düzenine (`surface/roi/LABEL`) dönüştürülür.
Aynı resim (içerik hash'i) tekrar eklenmez.

### Registry
**Varsayılan:** `training-server/registry`

Eğitilmiş ONNX modellerinin sürüm kaydı. Her ROI için tarih damgalı klasör:
`registry/<ürün>/surface_1/roi_1/<sürüm>/model.onnx`

### Bundle çıktı
**Varsayılan:** `training-server/data/bundles`

İstasyona gidecek ZIP paketinin yazıldığı klasör.
Örnek: `deneme_a_2026-06-07.zip`

### Epoch sayısı
**Varsayılan:** 15 (aralık: 1–200)

Her ROI modelinin kaç tur eğitileceği.

| Değer | Ne zaman? |
|---|---|
| 1 | Hızlı test |
| 15 | Normal kullanım |
| 30+ | Çok veri, daha iyi doğruluk |

### Barkod (opsiyonel)
Ürün barkod numarası. Bundle `manifest.json` dosyasına yazılır.
İstasyon import ederken ürünü tanımak için kullanılır. Boş bırakılabilir.

### Reçete sürümü
**Varsayılan:** 1

Ürün reçetesinin versiyon numarası. ROI yapısı değiştiğinde artırılır.

### Kabul kapısını atla (--force-register)
**Varsayılan:** Açık

Model kayıt aşamasındaki kalite kontrolünü atlar. Az veriyle test ederken
açık bırak; gerçek üretimde kapat (bkz. [Bölüm 9](#9-kabul-kapısı-acceptance-gate)).

---

## 6. Eğitim Akışı (Adım Adım)

"Eğitimi Başlat" butonu şu adımları sırayla çalıştırır:

```
Seçilen ürün klasörü
        │
        ▼
  1. INGEST ──► data/lake/<ürün>/surface_*/roi_*/EMPTY|FILLED/
        │
        ▼
  2. BUILD DATASET ──► Her ROI için train/val/test split (%70/15/15)
        │
        ▼
  3. TRAIN (GPU) ──► MobileNetV3-Small, ROI başına model.pt + metrics.json
        │
        ▼
  4. EXPORT ONNX ──► model.onnx (opset 17) + parity kontrolü
        │
        ▼
  5. REGISTRY ──► Kabul kapısı → kayıt veya red
        │
        ▼
  6. BUILD BUNDLE ──► data/bundles/<ürün>_<tarih>.zip
```

Her ROI için 3–6 tekrarlanır. 5 ROI ve 15 epoch ile tipik süre **10–20 dakika** (GPU'ya bağlı).

---

## 7. Çıktılar Nereye Yazılır?

| Dosya / Klasör | İçerik |
|---|---|
| `data/lake/<ürün>/` | Ingest edilmiş tüm resimler (hash ile dedup) |
| `data/work/` | Geçici split JSON, checkpoint, ONNX (eğitim sırasında) |
| `registry/<ürün>/` | Kayıtlı model sürümleri + `card.json` (metrikler) |
| `data/bundles/<ürün>_<tarih>.zip` | İstasyona gidecek paket |

### Bundle içeriği

```
deneme_a_2026-06-07.zip
├── manifest.json          ← ürün bilgisi, model listesi, checksum'lar
├── models/surface_1_roi_1.onnx
├── models/surface_1_roi_2.onnx
└── ...
```

Bundle'ı istasyonda **Models sekmesi** veya CLI ile import et.

---

## 8. Komut Satırından Kullanım

Arayüz yerine adımları tek tek de çalıştırabilirsin:

```bash
cd /home/ubuntu/training/training-server
source activate.sh

# 1. Ürünü lake'e aktar
python ingest/ingest.py \
    --product-dir /home/ubuntu/training/deneme_a \
    --product-name deneme_a \
    --lake data/lake

# 2. Tek komutla tam pipeline
python pipelines/monthly_retrain.py \
    --lake data/lake \
    --product deneme_a \
    --registry registry \
    --out data/bundles \
    --epochs 15 \
    --recipe-version 1 \
    --force-register
```

### Tek ROI eğitimi (manuel)

```bash
python datasets/build_dataset.py \
    --roi-dir data/lake/deneme_a/surface_1/roi_1 \
    --out data/work/deneme_a_s1_roi1.json

python training/train_roi.py \
    --split data/work/deneme_a_s1_roi1.json \
    --out data/work/deneme_a_s1_roi1 \
    --epochs 15

python export/export_onnx.py \
    --ckpt data/work/deneme_a_s1_roi1/model.pt \
    --out data/work/deneme_a_s1_roi1/model.onnx
```

---

## 9. Kabul Kapısı (Acceptance Gate)

Model registry'ye kaydedilmeden önce test setinde şu eşikler kontrol edilir:

| Metrik | Minimum |
|---|---|
| FILLED recall | %99 |
| EMPTY recall | %97 |

- **Geçerse** → model kaydedilir, bundle'a dahil edilir
- **Geçemezse** → model reddedilir (`rejected`)
- **`--force-register` / arayüzde "Kabul kapısını atla"** → eşikler yok sayılır

Az veriyle test (`deneme_a` gibi 20'şer resim) için kapıyı atla.
Gerçek üretimde bol veri + kapı açık kullan.

---

## 10. Sorun Giderme

### Arayüz açılmıyor / "Server error"

```bash
cd /home/ubuntu/training/training-server
source .venv/bin/activate
pip install --force-reinstall --no-cache-dir "streamlit==1.39.0"
bash start.sh
```

Streamlit 1.58+ sürümlerinde `static/index.html` eksikliği bilinen bir sorundur;
`requirements.txt` ve `start.sh` 1.39.0 kullanır.

### Ürün listede görünmüyor

- Klasör `/home/ubuntu/training/` altında mı? (veya ayarlardaki kök dizin doğru mu?)
- İçinde en az bir `surface_*` klasörü var mı?
- Klasör adı `.` ile başlamıyor mu?

### CUDA: False

```bash
nvidia-smi          # sürücü kurulu mu?
source activate.sh  # PyTorch CUDA wheel kurulu mu?
```

### Eğitim "rejected" ile biter

Arayüzde **"Kabul kapısını atla"** işaretli mi kontrol et.
Veya daha fazla resim ekle (ROI başına en az 50–100 önerilir).

### Port 8501 meşgul

```bash
pkill -f "streamlit run app.py"
bash start.sh
```

### `.venv` veya proje dosyaları silinmiş

```bash
bash start.sh   # venv ve paketleri yeniden kurar
```

Proje kaynak dosyaları (`app.py`, `training/`, vb.) silinmişse git veya yedekten geri yükle.

---

## Özet Komutlar

| İşlem | Komut |
|---|---|
| Arayüzü başlat | `bash start.sh` |
| Arayüz adresi | http://localhost:8501 |
| Ortamı aktive et | `source activate.sh` |
| CUDA kontrol | `python -c "import torch; print(torch.cuda.is_available())"` |
| Arayüzü kapat | Terminalde `Ctrl+C` |
