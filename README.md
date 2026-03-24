# CS401 Anomaly Detection Prototype

Bu repo, CICIDS2017 cleaned veri setini kullanarak ağ trafiğindeki anomalileri tespit eden bir CS401 dönem projesi prototipini içerir. Proje şu anda tamamen offline çalışır ve normal trafik örneklerinden Isolation Forest modeli eğitip karışık test verisi üzerinde performansını raporlar. Notebook tabanlı EDA adımlarının aynısı `src/baseline_model.py` içinde fonksiyonel olarak paketlenmiştir.

## Kurulum

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> Not: Autoencoder denemeleri için `torch` paketinin kurulu olması gerekir (requirements içinde yer alır).

## Projeyi Çalıştırma

Tek giriş noktası olarak `src/pipeline.py` kullanılabilir.

### 1) Baseline (cleaned veri ile)

```bash
python src/pipeline.py --pipeline baseline --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --report-path baseline_report.txt
```

- `--pipeline baseline`: Isolation Forest baseline train + eval.
- `--row-limit`: Büyük CSV'yi kısmi yüklemek için opsiyonel satır sınırı.
- `--contamination`: Test setindeki beklenen saldırı oranı (Isolation Forest parametresi).
- `--report-path`: Confusion matrix ve classification report çıktısının kaydedileceği dosya.

### 2) Feature Selection (train/test sample setleri)

```bash
python src/pipeline.py --pipeline feature-selection --train-path data/train_normal_sample.csv --test-path data/test_attack_sample.csv --output-train data/train_optimized.csv --output-test data/test_optimized.csv --corr-threshold 0.95
```

### 3) Isolation Forest Grid Search (optimized setler)

```bash
python src/pipeline.py --pipeline optimize-if --output-train data/train_optimized.csv --output-test data/test_optimized.csv --grid-output-path reports/tables/if_grid_search_results.csv
```

### 4) Mixed Evaluation + Figure üretimi

```bash
python src/pipeline.py --pipeline mixed-eval --output-train data/train_optimized.csv --output-test data/test_optimized.csv --contamination 0.10
```

### 5) Autoencoder (cleaned veri ile)

```bash
python src/pipeline.py --pipeline autoencoder --data-path data/cicids2017_cleaned.csv --row-limit 250000 --ae-epochs 20 --ae-tune-threshold --ae-max-fpr 0.35
```

### 6) Baseline vs Autoencoder Benchmark

```bash
python src/pipeline.py --pipeline benchmark --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --ae-epochs 20 --benchmark-output-path reports/tables/model_benchmark.csv
```

### 7) Hybrid Model (IF + Autoencoder score fusion)

```bash
python src/pipeline.py --pipeline hybrid --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --ae-epochs 20 --weight-if 0.5 --ae-tune-threshold --ae-max-fpr 0.35
```

### 8) Full Benchmark (IF vs AE vs Hybrid)

```bash
python src/pipeline.py --pipeline benchmark-all --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --ae-epochs 20 --weight-if 0.5 --benchmark-output-path reports/tables/model_benchmark_all.csv
```

### 9) Hybrid Weight Sweep (savunma için seçim tablosu)

```bash
python src/pipeline.py --pipeline hybrid-weight-sweep --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --ae-epochs 20 --weight-grid 0.2,0.4,0.6,0.8 --hybrid-weight-sweep-output-path reports/tables/hybrid_weight_sweep.csv
```

### 10) Final Freeze + Artifact Paketi

Experiment mode (onerilen gelistirme modu):

```bash
python src/pipeline.py --pipeline finalize --config-path config/experiment_config.json
```

Official frozen final mode:

```bash
python src/pipeline.py --pipeline finalize --config-path config/final_config.json
```

Bu komut sonunda ayrica `reports/final_locked_params.json` olusur. Bu dosya:
- frozen config degerlerini
- secilen AE/Hybrid threshold degerlerini
- onerilen hybrid weight bilgisini
tek yerde saklar.

`config/final_config.json` icinde dondurulan ana alanlar:
- `seed`
- `row_limit`
- `contamination`
- `ae_fixed_threshold` (null ise tune edilir)
- `hybrid_fixed_threshold` (null ise tune edilir)
- `weight_if`

`config/experiment_config.json` icinde:
- fixed threshold alanlari `null` birakilir
- `run_mode: experiment` kullanilir
- lock dosyasi yazilmaz (sadece summary + tablolar yazilir)

### 11) Replay-Based Dashboard

Model secimine gore tablo dosyalari:
- IF: `reports/tables/if_mixed_results.csv`
- AE: `reports/tables/ae_mixed_results.csv`
- Hybrid: `reports/tables/hybrid_mixed_results.csv`

Uretmek icin:

```bash
python src/pipeline.py --pipeline benchmark-all --data-path data/cicids2017_cleaned.csv --row-limit 250000 --contamination 0.36 --ae-epochs 20 --weight-if 0.5 --benchmark-output-path reports/tables/model_benchmark_all.csv
```

Dashboard:

```bash
streamlit run app/app.py
```

### 12) Fair Model Selection (train_normal / val_mixed / test_mixed)

```bash
python src/pipeline.py --pipeline experiment-select --config-path config/model_selection_experiment.json
```

Bu komut tek protokolde IF/AE/Hybrid adaylarini karsilastirip su dosyalari uretir:
- `reports/tables/experiment_if_candidates.csv`
- `reports/tables/experiment_ae_candidates.csv`
- `reports/tables/experiment_hybrid_candidates.csv`
- `reports/tables/experiment_model_comparison_test.csv`
- `reports/tables/experiment_selection_report.md`

### 13) Advanced IF Tuning (contamination + score threshold)

```bash
python src/pipeline.py --pipeline if-advanced-tune --data-path data/cicids2017_cleaned.csv --row-limit 250000 --seed 42 --ae-max-fpr 0.5
```

Bu komut su dosyalari uretir:
- `reports/tables/if_threshold_sweep_validation.csv`
- `reports/tables/if_advanced_tuning_summary.csv`
- `reports/tables/if_best_config.json`

### 14) PyOD Benchmark (ECOD / COPOD / HBOS)

```bash
python src/pipeline.py --pipeline pyod-benchmark --config-path config/pyod_benchmark.json
```

Bu komut su dosyalari uretir:
- `reports/tables/pyod_validation_sweep.csv`
- `reports/tables/pyod_model_comparison_test.csv`
- `reports/tables/pyod_hbos_mixed_results.csv` (dashboard replay icin)
- `reports/tables/pyod_hbos_bins*_mixed_results.csv` (bin tuning detaylari)
- `reports/tables/pyod_hbos_ensemble_mixed_results.csv` (ensemble kullanildiysa)
- `reports/tables/model_comparison_with_pyod.csv` (eger `experiment_model_comparison_test.csv` varsa)
- `reports/tables/pyod_benchmark_report.md`

### 15) PyOD Stability Check (HBOS vs AE)

```bash
python src/pipeline.py --pipeline pyod-stability-check --config-path config/pyod_stability_check.json
```

Bu komut su dosyalari uretir:
- `reports/tables/pyod_stability_runs.csv`
- `reports/tables/pyod_stability_summary.csv`
- `reports/tables/pyod_leakage_sanity.csv`
- `reports/tables/pyod_stability_report.md`

Not: stability check tablosunda `AE+HBOS Fusion` modeli de yer alir; boylece
AE / HBOS / Fusion mean-std karsilastirmasi tek dosyada gorulebilir.

### 16) AE + HBOS Score Fusion

```bash
python src/pipeline.py --pipeline ae-hbos-fusion --config-path config/ae_hbos_fusion.json
```

Bu komut su dosyalari uretir:
- `reports/tables/ae_hbos_fusion_candidates.csv`
- `reports/tables/ae_hbos_fusion_results.csv`
- `reports/tables/ae_hbos_fusion_summary.json`

Komutlara göre raporlar ve tablolar `reports/tables` ve `reports/figures` altına yazılır. Notebook tarafında (`notebook/01_eda_overview.ipynb`) aynı pipeline görselleştirmelerle birlikte adım adım belgelenmiştir.

Autoencoder tabanlı deneyler notebook’un 9. bölümünde yer alır; reconstruction error eşiğini değiştirerek Isolation Forest sonuçlarıyla karşılaştırabilirsiniz.

## Hedef Klasör Yapısı (Faz 1-2)

`src` dizini kademeli olarak aşağıdaki yapıya evrilecektir:

- `data_preprocessing.py`
- `feature_selection.py`
- `baseline_model.py`
- `autoencoder_model.py`
- `benchmark_models.py`
- `hybrid_model.py`
- `evaluate.py`
- `stream_simulator.py`
- `finalize_experiment.py`
- `explainability.py`
- `pipeline.py`

