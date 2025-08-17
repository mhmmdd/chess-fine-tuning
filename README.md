# Gemma 3 270M ile Satranç Fine-Tuning Projesi

Bu proje, Google'ın Gemma 3 270M modelini, `Thytu/ChessInstruct` veri setini kullanarak satranç üzerine özelleştirmek (fine-tuning) için gerekli betikleri ve yapıyı içerir. Proje, `unsloth` kütüphanesi kullanılarak LoRA (Low-Rank Adaptation) ile verimli bir eğitim süreci sunar.

## Projenin Amacı

Bu projenin temel amacı, genel amaçlı bir dil modeli olan Gemma 3 270M'yi, satrançla ilgili soruları anlama, pozisyonları değerlendirme ve hamle önerme gibi konularda daha yetenekli hale getirmektir. Fine-tuning öncesi ve sonrası model performansını karşılaştırarak, yapılan eğitimin etkisini net bir şekilde ölçmeyi hedefler.

## Proje Yapısı

```
.
├── data/
│   └── (Bu klasör, prepare_data.py çalıştırıldığında oluşturulan veri setini barındırır)
├── models/
│   └── (Bu klasör, fine_tune.py çalıştırıldığında eğitilmiş modeli barındırır)
├── scripts/
│   ├── prepare_data.py       # Veri setini indirir ve eğitim için hazırlar.
│   ├── fine_tune.py          # Modeli hazrlanan veri ile eğitir.
│   ├── evaluate_base_model.py # Fine-tuning yapılmamış temel modelin performansını test eder.
│   └── evaluate.py           # Fine-tuning yapılmış modelin performansını test eder.
├── README.md                 # Bu dosya.
└── requirements.txt          # Gerekli Python kütüphaneleri.
```

## Adım Adım Kullanım

### 1. Kurulum

Projeyi klonladıktan sonra, gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install -r requirements.txt
```

Virtual Environment aktif et
```bash
source .venv/bin/activate
```

*Not: Bu projenin GPU (özellikle NVIDIA CUDA) üzerinde çalışması tavsiye edilir.*

### 2. Temel Modelin Değerlendirilmesi (Opsiyonel ama Önerilir)

Fine-tuning işlemine başlamadan önce, orijinal modelin satranç konusundaki mevcut performansını görmek için aşağıdaki betiği çalıştırın. Bu, fine-tuning sonrası elde edeceğiniz gelişmeyi karşılaştırmanız için bir başlangıç noktası (baseline) oluşturacaktır.

```bash
python scripts/evaluate_base_model.py
```

### 3. Veri Setinin Hazırlanması

Modeli eğitmek için kullanılacak olan `Thytu/ChessInstruct` veri setini indirip ChatML formatına dönüştürmek için aşağıdaki komutu çalıştırın. İşlem tamamlandığında, `data/chess_instruct_chatml.json` dosyası oluşturulacaktır.

```bash
python scripts/prepare_data.py
```

### 4. Modelin Eğitilmesi (Fine-Tuning)

Hazırlanan veri setini kullanarak Gemma 3 270M modelini fine-tune etmek için aşağıdaki betiği çalıştırın. Bu işlem, donanımınıza bağlı olarak zaman alabilir. Eğitim tamamlandığında, eğitilmiş model dosyaları `models/gemma-3-270m-chess` klasörüne kaydedilecektir.

```bash
python scripts/fine_tune.py
```

### 5. Eğitilmiş Modelin Değerlendirilmesi

Fine-tuning işleminin sonuçlarını görmek ve modelin satranç yeteneklerinin ne kadar geliştiğini test etmek için aşağıdaki komutu çalıştırın. Bu betik, 2. adımda test ettiğiniz aynı soruya bu sefer eğitilmiş modelin verdiği cevabı gösterecektir.

```bash
python scripts/evaluate.py
```

İki değerlendirme betiğinin (`evaluate_base_model.py` ve `evaluate.py`) çıktılarını karşılaştırarak fine-tuning işleminin başarısını ölçebilirsiniz.