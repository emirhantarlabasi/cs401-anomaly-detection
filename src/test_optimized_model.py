import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

# Dosya yolları
TRAIN_OPT_PATH = '../data/train_optimized.csv'
TEST_OPT_PATH = '../data/test_optimized.csv'

def test_mixed_model():
    print("🚀 FİNAL TEST (KARMA VERİ İLE) BAŞLIYOR...")

    # 1. Verileri Yükle
    if not os.path.exists(TRAIN_OPT_PATH):
        print("❌ Dosya bulunamadı.")
        return

    print("⏳ Veriler yükleniyor...")
    # Normal veriler (Eğitim seti)
    df_normal = pd.read_csv(TRAIN_OPT_PATH)
    # Saldırı verileri (Test seti)
    df_attack = pd.read_csv(TEST_OPT_PATH)

    # 2. Normal Veriyi İkiye Böl (Train / Test)
    # Normal verinin %80'ini eğitim, %20'sini test için ayıralım.
    # Böylece modelin hiç görmediği normal verilerle de test etmiş oluruz.
    X_train_normal, X_test_normal = train_test_split(df_normal, test_size=0.2, random_state=42)

    print(f"📊 Eğitim Seti (Sadece Normal): {len(X_train_normal)} satır")
    print(f"📊 Test Seti (Normal): {len(X_test_normal)} satır")
    print(f"📊 Test Seti (Saldırı): {len(df_attack)} satır")

    # 3. Test Setini Oluştur (Normal + Saldırı Karışık)
    # Normallere etiket 0 verelim
    y_test_normal = [0] * len(X_test_normal)
    
    # Saldırılara etiket 1 verelim
    y_test_attack = [1] * len(df_attack)
    
    # Saldırı verisinden 'Attack Type' sütununu atalım (Sayısal olması için)
    X_test_attack = df_attack.drop(columns=['Attack Type'], errors='ignore')

    # Hepsini birleştirelim
    X_test_final = pd.concat([X_test_normal, X_test_attack])
    y_test_final = y_test_normal + y_test_attack

    # 4. Modeli Eğit (Sadece X_train_normal ile)
    print("🌲 Isolation Forest eğitiliyor...")
    clf = IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1)
    
    # Eğitim setinde 'Attack Type' kalmışsa temizle
    if 'Attack Type' in X_train_normal.columns:
        X_train_normal = X_train_normal.drop(columns=['Attack Type'])
        
    clf.fit(X_train_normal)

    # 5. Tahmin Yap
    print("🔮 Tahmin yapılıyor...")
    y_pred_raw = clf.predict(X_test_final)
    
    # Isolation Forest çıktılarını çevir (1 -> 0, -1 -> 1)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    # 6. Sonuçlar
    print("\n" + "="*50)
    print("🏆 GERÇEK PERFORMANS SONUÇLARI")
    print("="*50)
    print("Confusion Matrix:")
    # [[ TN, FP ]
    #  [ FN, TP ]]
    print(confusion_matrix(y_test_final, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_final, y_pred, digits=4))

if __name__ == "__main__":
    test_mixed_model()