import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------
# Dosya yollarını senin klasör yapına göre ayarladım
TRAIN_PATH = '../data/train_normal_sample.csv'
TEST_PATH = '../data/test_attack_sample.csv'

OUTPUT_TRAIN = '../data/train_optimized.csv'
OUTPUT_TEST = '../data/test_optimized.csv'

# Korelasyon Eşiği: %95 ve üzeri benzeyen sütunları atacağız
CORRELATION_THRESHOLD = 0.95

def perform_feature_selection():
    print("🚀 Feature Selection işlemi başlıyor...")
    
    # 1. Verileri Yükle
    if not os.path.exists(TRAIN_PATH):
        print(f"HATA: Dosya bulunamadı -> {TRAIN_PATH}")
        return

    print("⏳ Dosyalar okunuyor...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    
    initial_col_count = len(df_train.columns)
    print(f"📊 Başlangıç Sütun Sayısı: {initial_col_count}")

    # -----------------------------------------------------
    # 2. Sabit (Constant) Sütunları Temizle
    # (İçindeki tüm değerler aynı olan sütunlar çöp'tür)
    # -----------------------------------------------------
    # Sadece sayısal sütunlara bakıyoruz
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    
    # Standart sapması 0 olan (hiç değişmeyen) sütunları bul
    constant_cols = [col for col in numeric_cols if df_train[col].std() == 0]
    
    df_train.drop(columns=constant_cols, inplace=True)
    df_test.drop(columns=constant_cols, inplace=True)
    
    print(f"🗑️  Sabit Sütunlar Silindi ({len(constant_cols)} adet):")
    # print(constant_cols) # Merak edersen burayı açıp bakabilirsin

    # -----------------------------------------------------
    # 3. Yüksek Korelasyonlu (İkiz) Sütunları Temizle
    # (Birbiriyle %95 aynı olan sütunlardan birini atacağız)
    # -----------------------------------------------------
    print("⏳ Korelasyon analizi yapılıyor (Biraz sürebilir)...")
    
    # Korelasyon matrisini oluştur
    corr_matrix = df_train.select_dtypes(include=[np.number]).corr().abs()
    
    # Matrisin üst üçgenini al (çünkü alt taraf aynasıdır)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Eşik değerinden yüksek olanları bul
    to_drop = [column for column in upper.columns if any(upper[column] > CORRELATION_THRESHOLD)]
    
    df_train.drop(columns=to_drop, inplace=True)
    df_test.drop(columns=to_drop, inplace=True)
    
    print(f"👯 İkiz (Yüksek Korelasyonlu) Sütunlar Silindi ({len(to_drop)} adet)")

    # -----------------------------------------------------
    # 4. Kaydet
    # -----------------------------------------------------
    print("-" * 30)
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"📉 Eski Sütun Sayısı: {initial_col_count}")
    print(f"📈 Yeni Sütun Sayısı: {len(df_train.columns)}")
    
    df_train.to_csv(OUTPUT_TRAIN, index=False)
    df_test.to_csv(OUTPUT_TEST, index=False)
    
    print(f"💾 Yeni dosyalar kaydedildi:\n   -> {OUTPUT_TRAIN}\n   -> {OUTPUT_TEST}")

if __name__ == "__main__":
    perform_feature_selection()