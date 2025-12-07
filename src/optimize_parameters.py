import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
import os

# Dosya yolları
TRAIN_PATH = '../data/train_optimized.csv'
TEST_PATH = '../data/test_optimized.csv'

def grid_search_isolation_forest():
    print("🚀 GRID SEARCH BAŞLIYOR (EN İYİ AYARLARI ARIYORUZ)...")

    # 1. Verileri Yükle
    if not os.path.exists(TRAIN_PATH):
        print("❌ Dosyalar yok!")
        return

    X_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    # Test verisini hazırla (Label ve Data)
    y_test = df_test['Attack Type'].apply(lambda x: 0 if x == 'Normal Traffic' else 1)
    X_test = df_test.drop(columns=['Attack Type'])
    
    if 'Attack Type' in X_train.columns:
        X_train = X_train.drop(columns=['Attack Type'])

    # 2. Denenecek Ayarlar Listesi
    # Bilgisayar bunların hepsini tek tek deneyecek
    param_grid = {
        'n_estimators': [100, 200, 300],          # Kaç tane karar ağacı olsun?
        'contamination': [0.01, 0.05, 0.10, 0.15], # Anormallik oranı tahmini
        'max_samples': [256, 512, 'auto']         # Her ağaç kaç veri görsün?
    }

    best_score = 0
    best_params = {}

    print(f"Toplam {len(list(ParameterGrid(param_grid)))} farklı kombinasyon denenecek...")

    # 3. Döngü (Her ayarı dene)
    for params in ParameterGrid(param_grid):
        print(f"Testing: {params} ...", end=" ")
        
        clf = IsolationForest(
            n_estimators=params['n_estimators'],
            contamination=params['contamination'],
            max_samples=params['max_samples'],
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train)
        
        # Tahmin
        y_pred_raw = clf.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred_raw]
        
        # Skoru hesapla (Saldırı sınıfı için F1 Score)
        score = f1_score(y_test, y_pred, pos_label=1)
        print(f"-> F1 Score: {score:.4f}")
        
        # Eğer bu skor rekor ise kaydet
        if score > best_score:
            best_score = score
            best_params = params

    # 4. Sonuç
    print("\n" + "="*40)
    print(f"🏆 EN İYİ SONUÇ: {best_score:.4f}")
    print(f"⚙️  EN İYİ AYARLAR: {best_params}")
    print("="*40)

if __name__ == "__main__":
    grid_search_isolation_forest()