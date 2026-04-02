from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from evaluate import evaluate_binary


TRAIN_OPT_PATH = Path("data/train_optimized.csv")
TEST_OPT_PATH = Path("data/test_optimized.csv")


def test_mixed_model(
    train_opt_path: Path = TRAIN_OPT_PATH,
    test_opt_path: Path = TEST_OPT_PATH,
    contamination: float = 0.10,
) -> None:
    print("Final test (karma veri) basliyor...")

    # 1. Verileri Yükle
    if not train_opt_path.exists() or not test_opt_path.exists():
        raise FileNotFoundError("Optimize edilmis train/test dosyalari bulunamadi.")

    print("Veriler yukleniyor...")
    df_normal = pd.read_csv(train_opt_path)
    df_attack = pd.read_csv(test_opt_path)

    # 2. Normal Veriyi İkiye Böl (Train / Test)
    # Normal verinin %80'ini eğitim, %20'sini test için ayıralım.
    # Böylece modelin hiç görmediği normal verilerle de test etmiş oluruz.
    X_train_normal, X_test_normal = train_test_split(df_normal, test_size=0.2, random_state=42)

    print(f"Egitim seti (sadece normal): {len(X_train_normal)} satir")
    print(f"Test seti (normal): {len(X_test_normal)} satir")
    print(f"Test seti (saldiri): {len(df_attack)} satir")

    # 3. Test Setini Oluştur (Normal + Saldırı Karışık)
    # Normallere etiket 0 verelim
    y_test_normal = [0] * len(X_test_normal)
    
    # Saldırılara etiket 1 verelim
    y_test_attack = [1] * len(df_attack)
    
    # Saldırı isimlerini sakla, sonra sütunu düşür
    attack_types_list = df_attack["Attack Type"].tolist() if "Attack Type" in df_attack.columns else ["Unknown Attack"] * len(df_attack)
    X_test_attack = df_attack.drop(columns=['Attack Type'], errors='ignore')

    # Hepsini birleştirelim
    X_test_final = pd.concat([X_test_normal, X_test_attack])
    y_test_final = y_test_normal + y_test_attack
    
    # Normal ve Saldırı tiplerini birleştir
    combined_attack_types = ["Normal Traffic"] * len(X_test_normal) + attack_types_list

    # 4. Modeli Eğit (Sadece X_train_normal ile)
    print("Isolation Forest egitiliyor...")
    clf = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    
    # Eğitim setinde 'Attack Type' kalmışsa temizle
    if 'Attack Type' in X_train_normal.columns:
        X_train_normal = X_train_normal.drop(columns=['Attack Type'])
        
    clf.fit(X_train_normal)

    # 5. Tahmin Yap
    print("Tahmin yapiliyor...")
    y_pred_raw = clf.predict(X_test_final)
    
    # Isolation Forest çıktılarını çevir (1 -> 0, -1 -> 1)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    # 6. Sonuçlar
    print("\n" + "="*50)
    print("GERCEK PERFORMANS SONUCLARI")
    print("="*50)

    cm, report_text = evaluate_binary(pd.Series(y_test_final), np.array(y_pred))
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report_text)

    # 7. Sonuçları ve görselleri kaydet
    print("\nSonuclar ve gorseller kaydediliyor...")
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    # 7.1. Karma veri sonuç tablosu (best/worst case analizi için)
    scores = clf.decision_function(X_test_final)
    results_df = X_test_final.copy()
    results_df["true_label"] = y_test_final
    results_df["pred_label"] = y_pred
    results_df["anomaly_score"] = scores
    results_df["attack_type"] = combined_attack_types
    results_path = os.path.join("reports", "tables", "if_mixed_results.csv")
    results_df.to_csv(results_path, index_label="index")

    # 7.2. Confusion matrix görseli
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Normal", "Pred Attack"],
        yticklabels=["True Normal", "True Attack"],
    )
    plt.title("Isolation Forest – Confusion Matrix (Mixed Test Set)")
    plt.tight_layout()
    cm_path = os.path.join("reports", "figures", "if_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 7.3. Zaman içinde anomaly score (index ~ zaman)
    plt.figure(figsize=(10, 4))
    idx = np.arange(len(results_df))
    plt.plot(idx, results_df["anomaly_score"], label="Anomaly score", linewidth=0.8)

    # Tahmin edilen anomalileri işaretle (best / worst case görseli için)
    mask_anom = results_df["pred_label"] == 1
    x_anom = idx[mask_anom.values]
    y_anom = results_df.loc[mask_anom, "anomaly_score"].values
    plt.scatter(
        x_anom,
        y_anom,
        color="red",
        s=6,
        alpha=0.6,
        label="Predicted anomalies",
    )

    plt.xlabel("Flow index (proxy for time)")
    plt.ylabel("Isolation Forest anomaly score")
    plt.title("Anomaly scores over mixed test set")
    plt.legend(loc="best")
    plt.tight_layout()
    scores_path = os.path.join("reports", "figures", "if_anomaly_scores_over_time.png")
    plt.savefig(scores_path, dpi=150)
    plt.close()

    # 7.4. Zaman pencerelerine göre alarm yoğunluğu (ör: her 500 flow)
    window_size = 500
    window_ids = idx // window_size
    window_df = (
        pd.DataFrame(
            {
                "window": window_ids,
                "is_anomaly": results_df["pred_label"],
            }
        )
        .groupby("window")
        .agg(anomaly_rate=("is_anomaly", "mean"), count=("is_anomaly", "size"))
        .reset_index()
    )

    plt.figure(figsize=(10, 4))
    plt.bar(window_df["window"], window_df["anomaly_rate"], width=0.9)
    plt.xlabel(f"Window index (size={window_size} flows)")
    plt.ylabel("Predicted anomaly rate")
    plt.title("Predicted anomaly rate by window")
    plt.tight_layout()
    rate_path = os.path.join("reports", "figures", "if_anomaly_rate_by_window.png")
    plt.savefig(rate_path, dpi=150)
    plt.close()

    print(f"Tablo kaydedildi: {results_path}")
    print(f"Confusion matrix gorseli: {cm_path}")
    print(f"Anomaly score zaman grafigi: {scores_path}")
    print(f"Alarm yogunlugu grafigi: {rate_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate optimized IF model on mixed test set")
    parser.add_argument("--train-path", type=Path, default=TRAIN_OPT_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_OPT_PATH)
    parser.add_argument("--contamination", type=float, default=0.10)
    args = parser.parse_args()
    test_mixed_model(
        train_opt_path=args.train_path,
        test_opt_path=args.test_path,
        contamination=args.contamination,
    )
