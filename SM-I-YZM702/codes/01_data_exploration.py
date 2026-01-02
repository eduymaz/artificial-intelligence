"""
=================================================================================
SCRIPT 1: VERİ KEŞFİ VE İLK ANALİZ
=================================================================================
Amaç: Diabetik hastaların 30 gün içinde tekrar hastaneye yatış verilerini 
      detaylıca incelemek ve anlamak

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
import json

warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dizin yapısını oluştur
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'exploratory_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VERİ KEŞFİ VE İLK ANALİZ BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. VERİ YÜKLEME
# =================================================================================
print("\n[1] Veri yükleniyor...")

df = pd.read_csv(DATA_DIR / 'diabetic_data.csv')
mapping_df = pd.read_csv(DATA_DIR / 'IDS_mapping.csv')

print(f"✓ Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"✓ ID mapping dosyası yüklendi")

# =================================================================================
# 2. GENEL BİLGİLER
# =================================================================================
print("\n[2] Genel bilgiler çıkarılıyor...")

# Temel bilgiler
info_dict = {
    'total_samples': df.shape[0],
    'total_features': df.shape[1],
    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
    'duplicate_rows': df.duplicated().sum()
}

# Veri tipleri
dtype_counts = df.dtypes.value_counts().to_dict()
info_dict['data_types'] = {str(k): int(v) for k, v in dtype_counts.items()}

print(f"  • Toplam örnek sayısı: {info_dict['total_samples']:,}")
print(f"  • Toplam özellik sayısı: {info_dict['total_features']}")
print(f"  • Bellek kullanımı: {info_dict['memory_usage_mb']:.2f} MB")
print(f"  • Duplike satır sayısı: {info_dict['duplicate_rows']}")

# =================================================================================
# 3. HEDEF DEĞİŞKEN ANALİZİ (readmitted)
# =================================================================================
print("\n[3] Hedef değişken (readmitted) analiz ediliyor...")

target_dist = df['readmitted'].value_counts()
target_pct = df['readmitted'].value_counts(normalize=True) * 100

print("\n  Hedef Değişken Dağılımı:")
for value, count in target_dist.items():
    pct = target_pct[value]
    print(f"    • {value}: {count:,} (%{pct:.2f})")

# Hedef değişken görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
target_dist.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c', '#3498db'])
axes[0].set_title('Hedef Değişken Dağılımı (readmitted)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Readmission Durumu', fontsize=12)
axes[0].set_ylabel('Hasta Sayısı', fontsize=12)
axes[0].tick_params(axis='x', rotation=0)
for i, v in enumerate(target_dist.values):
    axes[0].text(i, v + 500, f'{v:,}\n(%{target_pct.values[i]:.1f})', 
                ha='center', fontsize=10, fontweight='bold')

# Pie chart
colors = ['#2ecc71', '#e74c3c', '#3498db']
axes[1].pie(target_dist.values, labels=target_dist.index, autopct='%1.1f%%',
           colors=colors, startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Hedef Değişken Oranları', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_target_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 01_target_distribution.png")

# =================================================================================
# 4. EKSİK DEĞER ANALİZİ
# =================================================================================
print("\n[4] Eksik değerler analiz ediliyor...")

# '?' karakteri eksik değer olarak işaretlenmiş
missing_counts = (df == '?').sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'feature': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_percentage': missing_pct.values
}).sort_values('missing_count', ascending=False)

# Eksik değeri olan sütunlar
missing_features = missing_df[missing_df['missing_count'] > 0]

print(f"\n  Eksik Değeri Olan Özellik Sayısı: {len(missing_features)}")
print("\n  En çok eksik değere sahip 10 özellik:")
for idx, row in missing_features.head(10).iterrows():
    print(f"    • {row['feature']}: {row['missing_count']:,} (%{row['missing_percentage']:.2f})")

# Eksik değer görselleştirme
if len(missing_features) > 0:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_missing = missing_features.head(15)
    bars = ax.barh(range(len(top_missing)), top_missing['missing_percentage'].values)
    
    # Renk gradyanı
    colors = plt.cm.Reds(top_missing['missing_percentage'].values / 100)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing['feature'].values)
    ax.set_xlabel('Eksik Değer Yüzdesi (%)', fontsize=12)
    ax.set_title('En Çok Eksik Değere Sahip 15 Özellik', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Yüzdeleri ekle
    for i, v in enumerate(top_missing['missing_percentage'].values):
        ax.text(v + 0.5, i, f'%{v:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_missing_values.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Grafik kaydedildi: 02_missing_values.png")

# Eksik değer raporunu kaydet
missing_features.to_csv(OUTPUT_DIR / 'missing_values_report.csv', index=False)

# =================================================================================
# 5. NÜMERİK ÖZELLİKLER ANALİZİ
# =================================================================================
print("\n[5] Numerik özellikler analiz ediliyor...")

# Numerik sütunları belirle
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n  Numerik özellik sayısı: {len(numeric_cols)}")

# İstatistiksel özet
numeric_stats = df[numeric_cols].describe()
numeric_stats.to_csv(OUTPUT_DIR / 'numeric_statistics.csv')
print(f"  ✓ İstatistiksel özet kaydedildi: numeric_statistics.csv")

# Önemli numerik özellikler için dağılım grafikleri
important_numeric = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                     'num_medications', 'number_diagnoses', 'number_outpatient',
                     'number_emergency', 'number_inpatient']

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(important_numeric):
    if col in df.columns:
        # Histogram ve KDE
        df[col].hist(bins=50, ax=axes[idx], alpha=0.7, color='#3498db', edgecolor='black')
        axes[idx].set_title(f'{col} Dağılımı', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frekans', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # İstatistikleri ekle
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Ort: {mean_val:.1f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Med: {median_val:.1f}')
        axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_numeric_distributions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 03_numeric_distributions.png")

# =================================================================================
# 6. KATEGORİK ÖZELLİKLER ANALİZİ
# =================================================================================
print("\n[6] Kategorik özellikler analiz ediliyor...")

# Kategorik sütunları belirle
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n  Kategorik özellik sayısı: {len(categorical_cols)}")

# Her kategorik değişken için benzersiz değer sayısı
categorical_info = []
for col in categorical_cols:
    unique_count = df[col].nunique()
    most_common = df[col].value_counts().index[0] if unique_count > 0 else None
    most_common_pct = (df[col].value_counts().values[0] / len(df)) * 100 if unique_count > 0 else 0
    
    categorical_info.append({
        'feature': col,
        'unique_values': unique_count,
        'most_common': most_common,
        'most_common_percentage': most_common_pct
    })

cat_info_df = pd.DataFrame(categorical_info)
cat_info_df = cat_info_df.sort_values('unique_values', ascending=False)
cat_info_df.to_csv(OUTPUT_DIR / 'categorical_info.csv', index=False)

print("\n  Kategorik özellikler özeti:")
for idx, row in cat_info_df.head(10).iterrows():
    print(f"    • {row['feature']}: {row['unique_values']} benzersiz değer (En sık: {row['most_common']} - %{row['most_common_percentage']:.1f})")

# Önemli kategorik özellikler için dağılım
important_categorical = ['race', 'gender', 'age', 'admission_type_id', 
                         'discharge_disposition_id', 'diabetesMed', 'change']

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.ravel()

for idx, col in enumerate(important_categorical):
    if col in df.columns and idx < len(axes):
        value_counts = df[col].value_counts().head(10)  # Top 10
        
        bars = axes[idx].barh(range(len(value_counts)), value_counts.values)
        axes[idx].set_yticks(range(len(value_counts)))
        axes[idx].set_yticklabels(value_counts.index)
        axes[idx].set_xlabel('Frekans', fontsize=10)
        axes[idx].set_title(f'{col} Dağılımı (Top 10)', fontsize=11, fontweight='bold')
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Değerleri ekle
        for i, v in enumerate(value_counts.values):
            axes[idx].text(v + max(value_counts.values)*0.01, i, f'{v:,}', 
                          va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_categorical_distributions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 04_categorical_distributions.png")

# =================================================================================
# 7. HEDEF DEĞİŞKENLE İLİŞKİLER
# =================================================================================
print("\n[7] Hedef değişkenle ilişkiler analiz ediliyor...")

# Numerik özelliklerle ilişki
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

for idx, col in enumerate(important_numeric[:9]):
    if col in df.columns:
        df.boxplot(column=col, by='readmitted', ax=axes[idx])
        axes[idx].set_title(f'{col} vs Readmitted', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Readmitted', fontsize=9)
        axes[idx].set_ylabel(col, fontsize=9)
        plt.sca(axes[idx])
        plt.xticks(rotation=0, fontsize=8)

plt.suptitle('Numerik Özelliklerin Hedef Değişkene Göre Dağılımı', 
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_numeric_vs_target.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 05_numeric_vs_target.png")

# Kategorik özelliklerle ilişki (örnek: age, gender, race)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

key_categorical = ['age', 'gender', 'race', 'diabetesMed']
for idx, col in enumerate(key_categorical):
    if col in df.columns and idx < 4:
        ct = pd.crosstab(df[col], df['readmitted'], normalize='index') * 100
        ct.plot(kind='bar', ax=axes[idx], stacked=False)
        axes[idx].set_title(f'{col} vs Readmitted (%)', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Yüzde (%)', fontsize=10)
        axes[idx].legend(title='Readmitted', fontsize=9)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_categorical_vs_target.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 06_categorical_vs_target.png")

# =================================================================================
# 8. KORELASYON ANALİZİ
# =================================================================================
print("\n[8] Korelasyon analizi yapılıyor...")

# Sadece numerik özellikler için
correlation_matrix = df[numeric_cols].corr()

# Yüksek korelasyonları bul (threshold > 0.7)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'feature_1': correlation_matrix.columns[i],
                'feature_2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print(f"\n  Yüksek korelasyonlu özellik çiftleri (|r| > 0.7): {len(high_corr_pairs)}")
    for pair in high_corr_pairs[:5]:
        print(f"    • {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")

# Korelasyon matrisi görselleştirme
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
           cmap='coolwarm', center=0, square=True, linewidths=0.5,
           cbar_kws={"shrink": 0.8})
plt.title('Numerik Özellikler Korelasyon Matrisi', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_correlation_matrix.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 07_correlation_matrix.png")

# =================================================================================
# 9. İLAÇ ÖZELLİKLERİ ANALİZİ
# =================================================================================
print("\n[9] İlaç özellikleri analiz ediliyor...")

# İlaç sütunlarını belirle
medication_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 
                  'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 
                  'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 
                  'miglitol', 'troglitazone', 'tolazamide', 'insulin']

# Her ilaç için kullanım oranı
medication_usage = {}
for med in medication_cols:
    if med in df.columns:
        # 'No' dışındaki değerler ilaç kullanımı olarak kabul edilir
        usage = (df[med] != 'No').sum()
        usage_pct = (usage / len(df)) * 100
        medication_usage[med] = {'count': usage, 'percentage': usage_pct}

# Sıralama
sorted_meds = sorted(medication_usage.items(), key=lambda x: x[1]['percentage'], reverse=True)

print("\n  En çok kullanılan 10 ilaç:")
for med, stats in sorted_meds[:10]:
    print(f"    • {med}: {stats['count']:,} hasta (%{stats['percentage']:.2f})")

# Görselleştirme
med_df = pd.DataFrame([(k, v['percentage']) for k, v in sorted_meds], 
                     columns=['medication', 'usage_pct'])

plt.figure(figsize=(14, 8))
bars = plt.barh(range(len(med_df)), med_df['usage_pct'].values)
plt.yticks(range(len(med_df)), med_df['medication'].values)
plt.xlabel('Kullanım Yüzdesi (%)', fontsize=12)
plt.title('İlaç Kullanım Oranları', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Renk gradyanı
colors = plt.cm.viridis(med_df['usage_pct'].values / med_df['usage_pct'].max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Değerleri ekle
for i, v in enumerate(med_df['usage_pct'].values):
    plt.text(v + 0.2, i, f'%{v:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_medication_usage.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 08_medication_usage.png")

# =================================================================================
# 10. ÖZET RAPOR OLUŞTURMA
# =================================================================================
print("\n[10] Özet rapor oluşturuluyor...")

summary_report = {
    'dataset_info': info_dict,
    'target_distribution': target_dist.to_dict(),
    'missing_values_summary': {
        'features_with_missing': len(missing_features),
        'top_missing_features': missing_features.head(5).to_dict('records')
    },
    'numeric_features': {
        'count': len(numeric_cols),
        'features': numeric_cols
    },
    'categorical_features': {
        'count': len(categorical_cols),
        'top_features': cat_info_df.head(10).to_dict('records')
    },
    'high_correlations': high_corr_pairs[:10] if high_corr_pairs else [],
    'medication_usage': {k: int(v) if isinstance(v, (np.int64, np.int32)) else v for k, v in sorted_meds[:10]}
}

# Convert numpy types to Python native types for JSON serialization
def convert_to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

summary_report = convert_to_native(summary_report)

# JSON olarak kaydet
with open(OUTPUT_DIR / 'exploration_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_report, f, indent=4, ensure_ascii=False)

print(f"  ✓ Özet rapor kaydedildi: exploration_summary.json")

# =================================================================================
# 11. MARKDOWN RAPOR
# =================================================================================
print("\n[11] Markdown rapor oluşturuluyor...")

markdown_content = f"""# VERİ KEŞFİ VE İLK ANALİZ RAPORU

## 1. Veri Seti Genel Bilgileri

- **Toplam Örnek Sayısı**: {info_dict['total_samples']:,}
- **Toplam Özellik Sayısı**: {info_dict['total_features']}
- **Bellek Kullanımı**: {info_dict['memory_usage_mb']:.2f} MB
- **Duplike Satır Sayısı**: {info_dict['duplicate_rows']}

## 2. Hedef Değişken (readmitted) Dağılımı

| Readmission | Hasta Sayısı | Yüzde |
|-------------|--------------|-------|
"""

for value, count in target_dist.items():
    pct = target_pct[value]
    markdown_content += f"| {value} | {count:,} | %{pct:.2f} |\n"

markdown_content += f"""
![Hedef Değişken Dağılımı](exploratory_analysis/01_target_distribution.png)

**Yorum**: Veri seti {'dengesiz' if target_pct.max() > 60 else 'dengeli'} bir dağılıma sahip. 
Bu durum model eğitiminde {'class weighting veya resampling teknikleri kullanmayı gerektirebilir' if target_pct.max() > 60 else 'normal eğitim sürecine devam edilebilir'}.

## 3. Eksik Değerler

- **Eksik Değeri Olan Özellik Sayısı**: {len(missing_features)}

### En Çok Eksik Değere Sahip Özellikler

| Özellik | Eksik Sayı | Yüzde |
|---------|------------|-------|
"""

for idx, row in missing_features.head(10).iterrows():
    markdown_content += f"| {row['feature']} | {row['missing_count']:,} | %{row['missing_percentage']:.2f} |\n"

markdown_content += f"""
![Eksik Değerler](exploratory_analysis/02_missing_values.png)

**Yorum**: {'Bazı özellikler çok yüksek oranda eksik değere sahip (>50%). Bu özellikler veri ön işleme aşamasında dikkatli ele alınmalı veya çıkarılmalı.' if missing_features['missing_percentage'].max() > 50 else 'Eksik değerler yönetilebilir seviyede.'}

## 4. Numerik Özellikler

- **Toplam Numerik Özellik Sayısı**: {len(numeric_cols)}

![Numerik Dağılımlar](exploratory_analysis/03_numeric_distributions.png)

![Numerik vs Target](exploratory_analysis/05_numeric_vs_target.png)

## 5. Kategorik Özellikler

- **Toplam Kategorik Özellik Sayısı**: {len(categorical_cols)}

![Kategorik Dağılımlar](exploratory_analysis/04_categorical_distributions.png)

![Kategorik vs Target](exploratory_analysis/06_categorical_vs_target.png)

## 6. Korelasyon Analizi

![Korelasyon Matrisi](exploratory_analysis/07_correlation_matrix.png)

**Yüksek Korelasyonlu Özellik Çiftleri (|r| > 0.7)**: {len(high_corr_pairs)}

## 7. İlaç Kullanım Analizi

![İlaç Kullanımı](exploratory_analysis/08_medication_usage.png)

### En Çok Kullanılan İlaçlar

| İlaç | Kullanım Yüzdesi |
|------|------------------|
"""

for med, stats in sorted_meds[:10]:
    markdown_content += f"| {med} | %{stats['percentage']:.2f} |\n"

markdown_content += """
## 8. Sonuç ve Öneriler

### Önemli Bulgular
1. Veri seti 101,766 hasta kaydı içermektedir
2. Hedef değişken (readmitted) dengesiz bir dağılım göstermektedir
3. Bazı özellikler yüksek oranda eksik değere sahiptir
4. İlaç kullanım patternleri readmission ile ilişkili olabilir

### Sonraki Adımlar
1. Eksik değerlerin detaylı analizi ve uygun imputation stratejisi
2. Kategorik değişkenlerin encoding stratejisi belirlenmesi
3. Feature engineering: yaş grupları, toplam ilaç sayısı gibi yeni özellikler
4. Class imbalance için strateji belirlenmesi (SMOTE, class weights, vb.)
5. Outlier tespiti ve yönetimi

---
*Rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(OUTPUT_DIR / 'exploration_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"  ✓ Markdown rapor kaydedildi: exploration_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("VERİ KEŞFİ VE İLK ANALİZ TAMAMLANDI")
print("=" * 80)
print(f"\n✓ Tüm çıktılar '{OUTPUT_DIR}' dizinine kaydedildi")
print(f"✓ Toplam {len(list(OUTPUT_DIR.glob('*.png')))} grafik oluşturuldu")
print(f"✓ Toplam {len(list(OUTPUT_DIR.glob('*.csv')))} CSV raporu oluşturuldu")
print(f"✓ Toplam {len(list(OUTPUT_DIR.glob('*.json')))} JSON raporu oluşturuldu")
print(f"✓ Toplam {len(list(OUTPUT_DIR.glob('*.md')))} Markdown raporu oluşturuldu")
print("\nSonraki Adım: 02_data_preprocessing.py scriptini çalıştırın")
print("=" * 80)
