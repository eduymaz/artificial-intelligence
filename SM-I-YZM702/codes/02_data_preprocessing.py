"""
=================================================================================
SCRIPT 2: VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ
=================================================================================
Amaç: Diabetik veri setini model eğitimine hazır hale getirmek için kapsamlı
      ön işleme ve özellik mühendisliği uygulamak

Yazar: Machine Learning Final Project
Tarih: Aralık 2024
=================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle

warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dizin yapısını oluştur
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'preprocessing'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. VERİ YÜKLEME
# =================================================================================
print("\n[1] Veri yükleniyor...")

df = pd.read_csv(DATA_DIR / 'diabetic_data.csv')
df_original = df.copy()

print(f"✓ Orijinal veri: {df.shape[0]} satır, {df.shape[1]} sütun")

# =================================================================================
# 2. DUPLIKE SATIRLARI ÇIKARMA
# =================================================================================
print("\n[2] Duplike satırlar kontrol ediliyor...")

initial_shape = df.shape[0]
df = df.drop_duplicates()
removed_duplicates = initial_shape - df.shape[0]

print(f"  • Çıkarılan duplike satır: {removed_duplicates}")
print(f"  • Kalan satır sayısı: {df.shape[0]}")

# =================================================================================
# 3. HEDEF DEĞİŞKENİ HAZIRLA
# =================================================================================
print("\n[3] Hedef değişken (readmitted) hazırlanıyor...")

# Binary classification: <30 gün readmission = 1, diğerleri = 0
# Bu klinik açıdan daha anlamlı (30 gün içinde tekrar yatış problematik)
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)

print(f"  • Orijinal hedef dağılımı:")
print(f"    {df['readmitted'].value_counts().to_dict()}")
print(f"  • Binary hedef dağılımı:")
print(f"    {df['readmitted_binary'].value_counts().to_dict()}")

# =================================================================================
# 4. GEREKSİZ SÜTUNLARI ÇIKARMA
# =================================================================================
print("\n[4] Gereksiz sütunlar çıkarılıyor...")

# encounter_id ve patient_nbr: Sadece identifier, model için gereksiz
# weight: %96 eksik değer
# payer_code: %40 eksik değer ve düşük öngörücü değer
# medical_specialty: %49 eksik değer

columns_to_drop = [
    'encounter_id',
    'patient_nbr', 
    'weight',  # Çok yüksek eksik değer
    'payer_code',  # Yüksek eksik değer
    'medical_specialty',  # Yüksek eksik değer
    'readmitted'  # Binary version kullanacağız
]

# Sadece var olan sütunları drop et
columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=columns_to_drop)

print(f"  • Çıkarılan sütun sayısı: {len(columns_to_drop)}")
print(f"  • Kalan sütun sayısı: {df.shape[1]}")

# =================================================================================
# 5. EKSİK DEĞER YÖNETİMİ
# =================================================================================
print("\n[5] Eksik değerler yönetiliyor...")

# '?' karakteri eksik değer olarak işaretlenmiş
df = df.replace('?', np.nan)

# Eksik değer durumunu incele
missing_before = df.isnull().sum()
missing_pct = (missing_before / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    'missing_count': missing_before,
    'missing_percentage': missing_pct
}).sort_values('missing_count', ascending=False)

print(f"\n  Eksik değere sahip sütunlar:")
for col, row in missing_summary[missing_summary['missing_count'] > 0].iterrows():
    print(f"    • {col}: {row['missing_count']} (%{row['missing_percentage']})")

# Strateji:
# - %50'den fazla eksik değere sahip sütunları çıkar
# - Kategorik değişkenler için mode imputation
# - Numerik değişkenler için median imputation

high_missing_cols = missing_summary[missing_summary['missing_percentage'] > 50].index.tolist()
if high_missing_cols:
    print(f"\n  %50'den fazla eksik değere sahip sütunlar çıkarılıyor: {high_missing_cols}")
    df = df.drop(columns=high_missing_cols)

# Kategorik ve numerik sütunları ayır
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Hedef değişkeni çıkar
if 'readmitted_binary' in numeric_cols:
    numeric_cols.remove('readmitted_binary')

# Kategorik değişkenler için mode imputation
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_value, inplace=True)
        print(f"  • {col}: Mode ile dolduruldu ({mode_value})")

# Numerik değişkenler için median imputation
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"  • {col}: Median ile dolduruldu ({median_value:.2f})")

# Eksik değer kontrolü
remaining_missing = df.isnull().sum().sum()
print(f"\n  ✓ Kalan toplam eksik değer: {remaining_missing}")

# =================================================================================
# 6. ÖZELLİK MÜHENDİSLİĞİ
# =================================================================================
print("\n[6] Özellik mühendisliği yapılıyor...")

# 6.1 Yaş gruplarını sayısal değere çevir
age_mapping = {
    '[0-10)': 5,
    '[10-20)': 15,
    '[20-30)': 25,
    '[30-40)': 35,
    '[40-50)': 45,
    '[50-60)': 55,
    '[60-70)': 65,
    '[70-80)': 75,
    '[80-90)': 85,
    '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_mapping)
print(f"  • age_numeric: Yaş kategorileri sayısal değere dönüştürüldü")

# 6.2 Toplam ilaç değişikliği sayısı
medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                  'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                  'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                  'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                  'glyburide-metformin', 'glipizide-metformin', 
                  'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                  'metformin-pioglitazone']

# Mevcut ilaç sütunlarını filtrele
available_meds = [col for col in medication_cols if col in df.columns]

# Her ilaç için kullanım durumu (0: No, 1: değişti)
df['num_medications_changed'] = 0
for med in available_meds:
    df['num_medications_changed'] += (df[med] != 'No').astype(int)

print(f"  • num_medications_changed: Toplam {len(available_meds)} ilaç değişikliği sayısı hesaplandı")

# 6.3 Toplam prosedür sayısı
df['total_procedures'] = df['num_lab_procedures'] + df['num_procedures']
print(f"  • total_procedures: Laboratuvar ve klinik prosedürler toplamı")

# 6.4 Acil durum geçmişi (emergency veya inpatient)
df['has_emergency_history'] = ((df['number_emergency'] > 0) | (df['number_inpatient'] > 0)).astype(int)
print(f"  • has_emergency_history: Acil durum veya yatış geçmişi")

# 6.5 Hastanede kalış süresi kategorisi
df['los_category'] = pd.cut(df['time_in_hospital'], 
                             bins=[0, 3, 7, 14], 
                             labels=['short', 'medium', 'long'])
print(f"  • los_category: Hastanede kalış süresi kategorisi (short/medium/long)")

# 6.6 İlaç tedavisi var mı?
df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
print(f"  • on_diabetes_med: Diyabet ilacı kullanımı (binary)")

# 6.7 İlaç değişikliği yapıldı mı?
df['med_changed'] = (df['change'] == 'Ch').astype(int)
print(f"  • med_changed: İlaç değişikliği yapıldı mı (binary)")

# 6.8 Yoğun bakım oranı
df['procedure_intensity'] = df['num_procedures'] / (df['time_in_hospital'] + 1)  # +1 to avoid division by zero
print(f"  • procedure_intensity: Günlük prosedür yoğunluğu")

# 6.9 İlaç yoğunluğu
df['medication_intensity'] = df['num_medications'] / (df['time_in_hospital'] + 1)
print(f"  • medication_intensity: Günlük ilaç yoğunluğu")

# =================================================================================
# 7. KATEGORİK DEĞİŞKENLERİ ENCODE ETME
# =================================================================================
print("\n[7] Kategorik değişkenler encode ediliyor...")

# Binary kategorik değişkenler (0/1)
binary_mappings = {
    'gender': {'Male': 0, 'Female': 1},
}

for col, mapping in binary_mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
        # Eğer mapping'de olmayan değerler varsa, onları en sık değerle doldur
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        print(f"  • {col}: Binary encoding yapıldı")

# Race için one-hot encoding (çünkü ordinal değil)
if 'race' in df.columns:
    race_dummies = pd.get_dummies(df['race'], prefix='race', drop_first=True)
    df = pd.concat([df, race_dummies], axis=1)
    print(f"  • race: One-hot encoding yapıldı ({len(race_dummies.columns)} yeni sütun)")

# Ordinal değişkenler için label encoding
ordinal_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

label_encoders = {}
for col in ordinal_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  • {col}: Label encoding yapıldı ({len(le.classes_)} sınıf)")

# İlaç sütunları için encoding (No=0, Steady=1, Up=2, Down=3)
medication_mapping = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}
for med in available_meds:
    if med in df.columns:
        df[med] = df[med].map(medication_mapping)
        # Eğer mapping'de olmayan değerler varsa 0 (No) ile doldur
        df[med].fillna(0, inplace=True)

print(f"  • {len(available_meds)} ilaç sütunu encode edildi")

# Diagnosis kodları için basitleştirme
# ICD-9 kodları: İlk üç rakam ana kategoriyi gösterir
for diag_col in ['diag_1', 'diag_2', 'diag_3']:
    if diag_col in df.columns:
        # Numerik olmayan değerleri handle et
        df[f'{diag_col}_category'] = df[diag_col].apply(lambda x: str(x)[:3] if pd.notna(x) else '999')
        # Frequency encoding
        freq_encoding = df[f'{diag_col}_category'].value_counts(normalize=True).to_dict()
        df[f'{diag_col}_freq'] = df[f'{diag_col}_category'].map(freq_encoding)
        print(f"  • {diag_col}: Kategori ve frequency encoding yapıldı")

# Gereksiz kategorik sütunları çıkar
cols_to_drop_after_encoding = ['age', 'race', 'diabetesMed', 'change', 'los_category',
                                'diag_1', 'diag_2', 'diag_3', 
                                'diag_1_category', 'diag_2_category', 'diag_3_category']
cols_to_drop_after_encoding = [col for col in cols_to_drop_after_encoding if col in df.columns]
df = df.drop(columns=cols_to_drop_after_encoding)

# =================================================================================
# 8. OUTLIER TESPİTİ VE YÖNETİMİ
# =================================================================================
print("\n[8] Outlier tespiti yapılıyor...")

# IQR metoduyla outlier tespiti
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'readmitted_binary' in numeric_features:
    numeric_features.remove('readmitted_binary')

outlier_stats = []
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 3*IQR kullanarak daha toleranslı
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_pct = (outliers / len(df)) * 100
    
    if outliers > 0:
        outlier_stats.append({
            'feature': col,
            'outlier_count': outliers,
            'outlier_percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })

outlier_df = pd.DataFrame(outlier_stats).sort_values('outlier_percentage', ascending=False)
print(f"\n  Outlier içeren özellik sayısı: {len(outlier_df)}")
for idx, row in outlier_df.head(10).iterrows():
    print(f"    • {row['feature']}: {row['outlier_count']} (%{row['outlier_percentage']:.2f})")

# Outlier'ları cap etme (winsorization) yerine bırakıyoruz
# Çünkü medikal verilerde extreme değerler anlamlı olabilir
print(f"  • Outlier'lar bırakıldı (medikal veri için anlamlı olabilir)")

# Outlier raporunu kaydet
if len(outlier_df) > 0:
    outlier_df.to_csv(OUTPUT_DIR / 'outlier_report.csv', index=False)

# =================================================================================
# 9. ÖZELLİK ÖLÇEKLENDİRME
# =================================================================================
print("\n[9] Özellikler ölçeklendiriliyor...")

# Hedef değişkeni ayır
y = df['readmitted_binary'].values
X = df.drop(columns=['readmitted_binary'])

# Tüm özelliklerin numerik olduğundan emin ol
X = X.select_dtypes(include=[np.number])

# StandardScaler kullan
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f"  • {X_scaled_df.shape[1]} özellik ölçeklendirildi")
print(f"  • Hedef değişken boyutu: {len(y)}")

# =================================================================================
# 10. FİNAL VERİ SETİNİ KAYDET
# =================================================================================
print("\n[10] İşlenmiş veri kaydediliyor...")

# Hedef değişkeni ekle
X_scaled_df['readmitted_binary'] = y

# CSV olarak kaydet
X_scaled_df.to_csv(PROCESSED_DATA_DIR / 'processed_data.csv', index=False)
print(f"  ✓ processed_data.csv kaydedildi")

# Scaler ve encoders'ı kaydet
with open(PROCESSED_DATA_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ scaler.pkl kaydedildi")

with open(PROCESSED_DATA_DIR / 'label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✓ label_encoders.pkl kaydedildi")

# Feature names kaydet
feature_names = X.columns.tolist()
with open(PROCESSED_DATA_DIR / 'feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))
print(f"  ✓ feature_names.txt kaydedildi ({len(feature_names)} özellik)")

# =================================================================================
# 11. ÖN İŞLEME ÖZETİ VE GÖRSELLEŞTİRME
# =================================================================================
print("\n[11] Ön işleme özeti oluşturuluyor...")

# Özellik sayısı karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Önce-sonra karşılaştırması
stages = ['Orijinal', 'Duplike\nÇıkarma', 'Gereksiz\nSütun Çıkarma', 
          'Eksik Değer\nYönetimi', 'Feature\nEngineering', 'Final']
feature_counts = [
    df_original.shape[1],
    df_original.shape[1],  # Duplike satır, sütun etkilemez
    df_original.shape[1] - len(columns_to_drop),
    df_original.shape[1] - len(columns_to_drop) - len(high_missing_cols),
    X.shape[1] - 10,  # Yaklaşık
    X.shape[1]
]

axes[0].plot(stages, feature_counts, marker='o', linewidth=2, markersize=10, color='#3498db')
axes[0].set_ylabel('Özellik Sayısı', fontsize=12)
axes[0].set_title('Ön İşleme Aşamalarında Özellik Sayısı', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Yeni eklenen özellikler
new_features = ['age_numeric', 'num_medications_changed', 'total_procedures',
                'has_emergency_history', 'on_diabetes_med', 'med_changed',
                'procedure_intensity', 'medication_intensity']
new_feature_values = [1] * len(new_features)

axes[1].barh(range(len(new_features)), new_feature_values, color='#2ecc71')
axes[1].set_yticks(range(len(new_features)))
axes[1].set_yticklabels(new_features)
axes[1].set_xlabel('Eklendi', fontsize=12)
axes[1].set_title('Yeni Oluşturulan Özellikler', fontsize=13, fontweight='bold')
axes[1].set_xlim([0, 2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_preprocessing_summary.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 01_preprocessing_summary.png")

# Hedef değişken dağılımı (önce-sonra)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Orijinal
df_original['readmitted'].value_counts().plot(kind='bar', ax=axes[0], color='#e74c3c')
axes[0].set_title('Orijinal Hedef Değişken', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Readmitted', fontsize=11)
axes[0].set_ylabel('Frekans', fontsize=11)
axes[0].tick_params(axis='x', rotation=0)

# Binary
pd.Series(y).value_counts().plot(kind='bar', ax=axes[1], color='#3498db')
axes[1].set_title('Binary Hedef Değişken (< 30 gün)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Readmitted < 30 gün', fontsize=11)
axes[1].set_ylabel('Frekans', fontsize=11)
axes[1].set_xticklabels(['Hayır (0)', 'Evet (1)'], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_target_transformation.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 02_target_transformation.png")

# Ölçeklendirilmiş özelliklerin dağılımı
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.ravel()

sample_features = X_scaled_df.columns[:9]  # İlk 9 özellik
for idx, col in enumerate(sample_features):
    if col != 'readmitted_binary':
        axes[idx].hist(X_scaled_df[col], bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
        axes[idx].set_title(f'{col} (Scaled)', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Değer', fontsize=9)
        axes[idx].set_ylabel('Frekans', fontsize=9)
        axes[idx].axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_scaled_features_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 03_scaled_features_distribution.png")

# =================================================================================
# 12. MARKDOWN RAPOR
# =================================================================================
print("\n[12] Markdown rapor oluşturuluyor...")

preprocessing_report = f"""# VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ RAPORU

## 1. Veri Temizleme

### 1.1 Duplike Satırlar
- **Çıkarılan Duplike Satır**: {removed_duplicates}
- **Kalan Satır Sayısı**: {df.shape[0]:,}

### 1.2 Gereksiz Sütunlar
Aşağıdaki sütunlar analiz için uygun olmadığından çıkarıldı:
"""

for col in columns_to_drop:
    preprocessing_report += f"- `{col}`\n"

preprocessing_report += f"""
**Toplam Çıkarılan Sütun**: {len(columns_to_drop)}

## 2. Eksik Değer Yönetimi

### 2.1 Strateji
- **%50'den fazla eksik değere sahip sütunlar**: Veri setinden çıkarıldı
- **Kategorik değişkenler**: Mode (en sık değer) ile dolduruldu
- **Numerik değişkenler**: Median ile dolduruldu

### 2.2 Yüksek Eksik Değere Sahip Çıkarılan Sütunlar
"""

if high_missing_cols:
    for col in high_missing_cols:
        preprocessing_report += f"- `{col}`\n"
else:
    preprocessing_report += "- Yok\n"

preprocessing_report += f"""
**Kalan Toplam Eksik Değer**: {remaining_missing}

## 3. Hedef Değişken Dönüşümü

Orijinal hedef değişken (`readmitted`) 3 kategoriye sahipti:
- `<30`: 30 günden önce tekrar yatış
- `>30`: 30 günden sonra tekrar yatış  
- `NO`: Tekrar yatış yok

**Yeni Binary Hedef**: `readmitted_binary`
- **1**: 30 günden önce tekrar yatış (<30)
- **0**: Diğer durumlar (>30 veya NO)

**Gerekçe**: Klinik açıdan 30 gün içinde tekrar yatış problematik ve önlenebilir olarak kabul edilir.

![Hedef Değişken Dönüşümü](preprocessing/02_target_transformation.png)

### Dağılım
- **Sınıf 0**: {(y == 0).sum():,} (%{(y == 0).sum() / len(y) * 100:.2f})
- **Sınıf 1**: {(y == 1).sum():,} (%{(y == 1).sum() / len(y) * 100:.2f})

**Class Imbalance Durumu**: {'Var - Model eğitiminde class_weight veya SMOTE kullanılacak' if (y == 0).sum() / len(y) > 0.8 else 'Hafif dengesizlik var'}

## 4. Özellik Mühendisliği

### 4.1 Oluşturulan Yeni Özellikler

| Özellik | Açıklama | Gerekçe |
|---------|----------|---------|
| `age_numeric` | Yaş kategorilerinin numerik versiyonu | Model için sürekli değişken daha kullanışlı |
| `num_medications_changed` | Değişiklik yapılan toplam ilaç sayısı | Tedavi değişikliği readmission ile ilişkili olabilir |
| `total_procedures` | Lab + klinik prosedür toplamı | Toplam medikal müdahale yoğunluğu |
| `has_emergency_history` | Acil/yatış geçmişi var mı (binary) | Kronik hastalık göstergesi |
| `on_diabetes_med` | Diyabet ilacı kullanıyor mu (binary) | Hastalık şiddeti göstergesi |
| `med_changed` | İlaç değişikliği yapıldı mı (binary) | Tedavi etkinliği göstergesi |
| `procedure_intensity` | Günlük prosedür sayısı | Tedavi yoğunluğu normalizasyonu |
| `medication_intensity` | Günlük ilaç sayısı | İlaç yoğunluğu normalizasyonu |

### 4.2 Diagnosis Code Engineering
ICD-9 kodları için:
- **Kategori**: İlk 3 rakam (ana hastalık kategorisi)
- **Frequency Encoding**: Her kategorinin veri setindeki sıklığı

## 5. Kategorik Değişken Encoding

### 5.1 Binary Encoding
- `gender`: Male=0, Female=1

### 5.2 One-Hot Encoding
- `race`: Farklı ırklar için dummy variables

### 5.3 Label Encoding
- `admission_type_id`
- `discharge_disposition_id`
- `admission_source_id`

### 5.4 İlaç Değişkenleri
Tüm ilaç sütunları ordinal encoding:
- No = 0
- Steady = 1
- Up = 2
- Down = 3

## 6. Outlier Yönetimi

**Tespit Metodu**: IQR (Interquartile Range) - 3×IQR

**Outlier İçeren Özellik Sayısı**: {len(outlier_df)}

### En Çok Outlier İçeren Özellikler
"""

if len(outlier_df) > 0:
    preprocessing_report += "\n| Özellik | Outlier Sayısı | Yüzde |\n|---------|----------------|-------|\n"
    for idx, row in outlier_df.head(10).iterrows():
        preprocessing_report += f"| {row['feature']} | {row['outlier_count']} | %{row['outlier_percentage']:.2f} |\n"
else:
    preprocessing_report += "\nOutlier tespit edilmedi.\n"

preprocessing_report += f"""
**Karar**: Outlier'lar bırakıldı. Medikal verilerde extreme değerler klinik olarak anlamlı olabilir ve 
model için önemli bilgi taşıyabilir. Robust scaler kullanımı ile etkileri azaltılacak.

## 7. Feature Scaling

**Metod**: StandardScaler (Z-score normalization)

$$
z = \\frac{{x - \\mu}}{{\\sigma}}
$$

- Tüm özellikler ortalama=0, standart sapma=1 olacak şekilde ölçeklendirildi
- Model performansını artırmak ve gradient descent optimizasyonunu hızlandırmak için gerekli

![Ölçeklendirilmiş Özellikler](preprocessing/03_scaled_features_distribution.png)

## 8. Final Veri Seti

### 8.1 Boyutlar
- **Örnek Sayısı**: {len(X_scaled_df):,}
- **Özellik Sayısı**: {len(feature_names)}
- **Hedef Değişken**: readmitted_binary (binary)

### 8.2 Veri Tipi
- Tüm özellikler numerik (float64)
- Ölçeklendirilmiş (standardized)
- Eksik değer yok

### 8.3 Kaydedilen Dosyalar
1. `processed_data.csv` - İşlenmiş ve ölçeklendirilmiş veri
2. `scaler.pkl` - StandardScaler objesi (yeni veri için)
3. `label_encoders.pkl` - Label encoder'lar (yeni veri için)
4. `feature_names.txt` - Özellik isimleri

![Ön İşleme Özeti](preprocessing/01_preprocessing_summary.png)

## 9. Sonuç ve Model Eğitimine Hazırlık

### Başarıyla Tamamlanan Adımlar ✓
- [x] Veri temizleme (duplikeler, gereksiz sütunlar)
- [x] Eksik değer yönetimi (imputation)
- [x] Hedef değişken dönüşümü (binary classification)
- [x] Özellik mühendisliği (8 yeni özellik)
- [x] Kategorik değişken encoding
- [x] Outlier tespiti ve analizi
- [x] Feature scaling (standardization)

### Sonraki Adımlar
1. **Train-Test Split**: Stratified split ile class balance korunacak
2. **Class Imbalance Handling**: SMOTE veya class_weight kullanılacak
3. **Model Selection**: En az 3 farklı algoritma denenecek
4. **Hyperparameter Tuning**: GridSearchCV ile optimize edilecek
5. **Model Evaluation**: Comprehensive metrics (ROC-AUC, F1, Precision, Recall)

---
*Rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(OUTPUT_DIR / 'preprocessing_report.md', 'w', encoding='utf-8') as f:
    f.write(preprocessing_report)

print(f"  ✓ Markdown rapor kaydedildi: preprocessing_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ TAMAMLANDI")
print("=" * 80)
print(f"\n✓ Final veri seti: {len(X_scaled_df):,} satır × {len(feature_names)} özellik")
print(f"✓ Hedef değişken: readmitted_binary (class 0: {(y==0).sum():,}, class 1: {(y==1).sum():,})")
print(f"✓ Tüm dosyalar '{PROCESSED_DATA_DIR}' dizinine kaydedildi")
print(f"✓ Tüm raporlar '{OUTPUT_DIR}' dizinine kaydedildi")
print("\nSonraki Adım: 03_model_training.py scriptini çalıştırın")
print("=" * 80)
