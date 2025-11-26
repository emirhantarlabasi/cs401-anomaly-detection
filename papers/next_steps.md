## CS401 Anomaly Detection – Yakın Yol Haritası

### 1. Veri Sağlığı & Notebooks
- `notebook/01_eda_overview.ipynb` içindeki hücreleri çalıştırıp veri boyutu, sütun tipleri ve sınıf dağılımını doğrula.
- Eksik değer kontrolü ve uç değer özetlerini ekle; sonuçları README’ye kısa bir notla taşı.

### 2. Feature Engineering
- `src/features.py` içinde yeniden kullanılabilir fonksiyonlar yaz:
  - Flow süresi normalizasyonu (`duration_sec`)
  - Paket/byte hızları (`bytes_per_sec`, `packets_per_sec`)
  - Yön oranları (`fwd_bwd_ratio`)
- Feature setini `config/base_features.yaml` gibi bir dosyada listeleyip notebook’larda aynı kaynaktan oku.

### 3. İstatistiksel Taban
- Z-score ve MAD tabanlı eşik fonksiyonlarını `src/baselines/stat_thresholds.py` altında merkezileştir.
- Notebook’ta ROC/PR çıktıları ve yanlış pozitif/negatif analizi oluştur; görselleri `reports/figures/` dizinine kaydet.

### 4. Unsupervised Modeller
- `src/models/train_isolation_forest.py` → yalnızca normal etiketli subset üzerinde eğitim.
- `src/models/train_oneclass_svm.py` ve `train_lof.py` için ortak preprocessing pipeline’ı (`StandardScaler`, `PCA` opsiyonel).
- Modelleri `models/` dizinine `joblib` ile kaydet; meta bilgiyi `models/model_registry.json` içinde tut.

### 5. Prototip & İnferans
- `src/pipeline/infer.py` dosyasında tek girişli komut satırı aracı: CSV al, modeli yükle, anomaly skorlarını ve etiketleri döndür.
- Notebook’ta skor dağılımlarını ve örnek akış açıklamalarını görselleştirerek “proof-of-concept” demosu üret.

### 6. Gerçek Trafik Entegrasyonu (Mini Deneme)
- `papers/wireshark_notes.md` içine capture adımları, filtreler ve anonimleştirme prosedürünü yaz.
- Zeek/pyshark çıktısını CICIDS sütunlarına eşleyen dönüştürücü script (`src/ingest/pcap_to_flow.py`).
- Gerçek trafikle inference çalıştırıp farklarını kısa raporla özetle.

### 7. CS402 Hazırlığı
- Streaming gereksinimleri: pencere boyutu, latency hedefleri, güncelleme stratejileri.
- Daha gelişmiş modeller (Autoencoder, Deep SVDD) için literatür listesi oluşturup `papers/lit_review.md` dosyasına ekle.

