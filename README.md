# Local Document AI Pipeline for Irregular Kids Magazine Pages

## Proje Amacı
Bu proje, düzensiz/serbest yerleşimli çocuk dergisi sayfalarında metin ve görsel bölgelerini ayrıştırmak için yerel (offline) çalışan bir pipeline sağlar.

Öncelik sırası:
1. Text vs non-text ayrımı
2. Mantıksal text block birleştirme
3. `headline/content/other_text` rol ataması
4. Çok küçük görsel adaylarını eleme
5. JSON + crop + debug çıktıları

## Mimari Akış
1. OCR line tespiti (`Surya OCR`, MPS öncelikli)
2. OCR line'ları mantıksal text block'lara graph tabanlı birleştirme
3. Headline/content heuristic sınıflama
4. Headline-content eşleme ile `article_group_id` atama
5. Text mask üretimi ve dilation
6. Non-text aday maskesi (`foreground AND NOT(text_mask)`)
7. Görsel adaylarını çıkarma + filtreleme
8. Final bbox, crop, JSON, debug görsellerini yazma

## Klasör Yapısı
```text
project_root/
  README.md
  requirements.txt
  .gitignore
  images/
  outputs/
    json/
    crops/
    debug/
  src/
    __init__.py
    config.py
    run_batch.py
    run_single.py
    backends/
      __init__.py
      ocr_base.py
      paddle_ocr_backend.py
      surya_backend.py
    pipeline/
      __init__.py
      page_pipeline.py
      text_detection.py
      text_grouping.py
      article_grouping.py
      text_classification.py
      text_mask.py
      non_text_mask.py
      visual_candidates.py
      cropper.py
      debug_viz.py
      models.py
    utils/
      __init__.py
      geometry.py
      image_ops.py
      io.py
      logging_utils.py
  tests/
    test_geometry.py
    test_text_grouping.py
    test_visual_filters.py
    test_pipeline_empty_ocr.py
```

## Kurulum (macOS + venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m src.run_batch --input images --output outputs
```

## Kurulum (Windows + venv)
PowerShell:
```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m src.run_batch --input images --output outputs
```

CMD:
```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m src.run_batch --input images --output outputs
```

## Çalıştırma
Batch (görsel klasörü):
```bash
python -m src.run_batch --input images --output outputs
```

Tek görsel:
```bash
python -m src.run_single --input images/page_001.png --output outputs
```

PDF klasörü (otomatik sayfa render + ara PNG kaydetmeden doğrudan işlem):
```bash
python -m src.run_batch --input inputs --output outputs
```

Tek PDF:
```bash
python -m src.run_batch --input inputs/t_cocuk_part_01_pages_0001_0003.pdf --output outputs
```

Tek PDF + tek sayfa:
```bash
python -m src.run_single --input inputs/t_cocuk_part_01_pages_0001_0003.pdf --page 1 --output outputs
```

Debug kapalı çalıştırma:
```bash
python -m src.run_batch --input images --output outputs --no-debug
```

İlerleme barlarını aç (varsayılan sade terminal çıktısı):
```bash
python -m src.run_batch --input images --output outputs --show-progress
```

Hız odaklı PDF mikro-batch (Surya):
```bash
python -m src.run_batch --input inputs --output outputs --no-debug --ocr-page-batch 3
```

## JSON Çıkış Sözleşmesi
Her sayfa için `outputs/json/<page_stem>.json` üretilir:
- `image_path`
- `page_size`
- `text_blocks[]`
  - `block_id`, `article_group_id`, `role`, `bbox_px`, `text`
- `visual_blocks[]`
  - `block_id`, `bbox_px`, `crop_path`

Ek teşhis çıktısı:
- `outputs/json_debug/<page_stem>.json`
  - `text_blocks_debug` (`line_count`, `avg_line_confidence`, `regions_px`)
  - `rejected_visual_candidates` (`bbox_px`, `reason`, `area_ratio`)

## Debug Çıktıları
`outputs/debug/` altında sayfa başına:
- `_01_ocr_lines.png`
- `_02_text_blocks.png`
- `_03_roles.png`
- `_04_text_mask.png`
- `_05_non_text_mask.png`
- `_06_visual_boxes.png`

## Threshold Tuning Notları
Tüm ayarlar `src/config.py` içindedir.

Önemli görsel filtre eşikleri:
- `min_visual_area_ratio = 0.0125`
- `min_visual_short_side_ratio = 0.05`
- `min_visual_width_px = 48`
- `min_visual_height_px = 48`
- `max_visual_area_ratio = 0.65`
- `max_text_overlap_ratio = 0.22`
- `split_trigger_area_ratio = 0.75` (non-text maske tek dev parçaya çökünce bölme fallback'i)
- `split_min_component_area_ratio = 0.0035`
- `split_texture_std_threshold = 16.0`

OCR kalite iyileştirme:
- `text_correction.enabled = true`
- Türkçe sözlük tabanlı post-correction (`src/pipeline/text_correction.py`)
- İsterseniz `text_correction.lexicon_path` ile özel kelime listesi ekleyebilirsiniz.

Surya performans ayarları (`src/config.py`):
- `ocr.surya_detection_batch_size`
- `ocr.surya_recognition_batch_size`
- `ocr.surya_enable_adaptive_batch` (OOM olduğunda otomatik batch küçültme)
- `ocr.surya_page_batch_size` (PDF sayfa mikro-batch boyutu)

Gerekçe:
- Hem relatif hem mutlak eşik kullanarak farklı çözünürlüklerde küçük/dekoratif adayları bastırmak.
- Sadece piksel tabanlı filtrelemede yüksek çözünürlükte minik dekorlar kaçabilir; sadece oran tabanlı filtrelemede de düşük çözünürlükte faydalı görseller düşebilir.

Ayarlama önerisi:
- Çok fazla küçük ikon geliyorsa: `min_visual_area_ratio` ve `min_edge_density` artırın.
- Büyük görseller kaçıyorsa: `min_fill_ratio` ve `min_edge_density` azaltın.
- Metin içine taşma varsa: `text_dilation_ratio` düşürün.
- Metin dışı maskede parçalanma varsa: `non_text_close_ratio` artırın.
- Tam sayfa görsel false-positive geliyorsa: `max_visual_area_ratio` veya `edge_touch_reject_area_ratio` düşürün.

## Testler
```bash
python -m pytest -q
```

Kapsam:
1. bbox normalize/de-normalize
2. küçük görsel filtreleme
3. text line birleştirme
4. headline/content heuristic smoke
5. OCR boşken pipeline çökmeden çalışma

## Bilinen Sınırlamalar
- Aşırı dekoratif/stilize yazılarda OCR line kaçırabilir.
- Çok düz ve arka planla benzer görseller foreground maskesinde zayıf kalabilir.
- Headline/content ayrımı semantik değil heuristic olduğu için her sayfada kusursuz değildir.
- Paddle backend kodu projede duruyor, fakat varsayılan akış Surya OCR ile çalışır.

## Not
Bu pipeline, üretim için sağlam bir başlangıçtır; veri setine göre `src/config.py` eşiklerinin kalibrasyonu kritik önemdedir.
