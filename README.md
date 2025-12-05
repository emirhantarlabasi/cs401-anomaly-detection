# CS401 Anomaly Detection Prototype

Bu repo, CICIDS2017 cleaned veri setini kullanarak ağ trafiğindeki anomalileri tespit eden CS401 bitirme projesi prototipini içerir. Proje şu anda tamamen offline çalışır ve normal trafik örneklerinden Isolation Forest modeli eğitip karışık test verisi üzerinde performansını raporlar. Notebook tabanlı EDA adımlarının aynısı `src/baseline_model.py` içinde fonksiyonel olarak paketlenmiştir.

## Kurulum

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> Not: Autoencoder denemeleri için `torch` paketinin kurulu olması gerekir (requirements içinde yer alır).

## Projeyi Çalıştırma

```bash
python src/baseline_model.py --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --report-path baseline_report.txt
```

- `--row-limit`: Büyük CSV'yi kısmi yüklemek için opsiyonel satır sınırı.
- `--contamination`: Test setindeki beklenen saldırı oranı (Isolation Forest parametresi).
- `--report-path`: Confusion matrix ve classification report çıktısının kaydedileceği dosya.

Komut çalıştıktan sonra terminalde metrikler yazdırılır ve kök dizinde `baseline_report.txt` dosyası oluşur. Notebook tarafında (`notebook/01_eda_overview.ipynb`) aynı pipeline görselleştirmelerle birlikte adım adım belgelenmiştir.

Autoencoder tabanlı deneyler notebook’un 9. bölümünde yer alır; reconstruction error eşiğini değiştirerek Isolation Forest sonuçlarıyla karşılaştırabilirsiniz.

