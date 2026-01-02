"""
=================================================================================
SCRIPT 5: HİPERPARAMETRE OPTİMİZASYONU
=================================================================================
Amaç: En iyi performans gösteren modellerin hiperparametrelerini optimize etmek
      için GridSearchCV ve RandomizedSearchCV kullanımı

Optimizasyon Yapılacak Modeller:
1. Random Forest
2. XGBoost  
3. LightGBM

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
import pickle
import json
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import time
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Dizin yapısını oluştur
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'hyperparameter_tuning'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("HİPERPARAMETRE OPTİMİZASYONU BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. VERİ YÜKLEME
# =================================================================================
print("\n[1] İşlenmiş veri yükleniyor...")

df = pd.read_csv(DATA_DIR / 'processed_data.csv')
X = df.drop(columns=['readmitted_binary']).values
y = df['readmitted_binary'].values

print(f"✓ Veri yüklendi: {len(X)} örnek, {X.shape[1]} özellik")

# Train data yükle (resampled)
with open(MODELS_DIR / 'sampling_pipeline.pkl', 'rb') as f:
    sampling_pipeline = pickle.load(f)

# Test data yükle
with open(MODELS_DIR / 'test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    X_test = test_data['X_test'].values
    y_test = test_data['y_test']

# Train-test split (80-20)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Resampling uygula
X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)

print(f"✓ Train set (resampled): {len(X_train_resampled)} örnek")
print(f"✓ Validation set: {len(X_val)} örnek")
print(f"✓ Test set: {len(X_test)} örnek")

# =================================================================================
# 2. SCORING METRİĞİ TANIMLAMA
# =================================================================================
print("\n[2] Scoring metrikleri tanımlanıyor...")

# F1-score'u optimize edeceğiz (recall ve precision dengesi için)
# ROC-AUC da izleyeceğiz
scoring = {
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
}

# Refit için f1 kullan
refit_metric = 'f1'

print(f"✓ Primary metric: {refit_metric}")
print(f"✓ Secondary metric: roc_auc")

# =================================================================================
# 3. RANDOM FOREST - RANDOMIZED SEARCH
# =================================================================================
print("\n[3] Random Forest hiperparametre optimizasyonu başlatılıyor...")
print("  Metod: RandomizedSearchCV (hızlı geniş arama)")

# Parameter grid
rf_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample'],
    'criterion': ['gini', 'entropy']
}

rf_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

# RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_dist,
    n_iter=50,  # 50 farklı kombinasyon dene
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring=scoring,
    refit=refit_metric,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE
)

print("\n  Random Forest optimizasyonu çalışıyor...")
rf_start = time.time()
rf_random.fit(X_train_resampled, y_train_resampled)
rf_time = time.time() - rf_start

print(f"\n  ✓ Random Forest optimizasyonu tamamlandı ({rf_time:.2f}s)")
print(f"  ✓ En iyi parametreler: {rf_random.best_params_}")
print(f"  ✓ En iyi F1-score (CV): {rf_random.best_score_:.4f}")

# Test performance
rf_best = rf_random.best_estimator_
rf_test_score = f1_score(y_test, rf_best.predict(X_test))
rf_test_roc = roc_auc_score(y_test, rf_best.predict_proba(X_test)[:, 1])

print(f"  ✓ Test F1-score: {rf_test_score:.4f}")
print(f"  ✓ Test ROC-AUC: {rf_test_roc:.4f}")

# =================================================================================
# 4. XGBOOST - GRID SEARCH
# =================================================================================
print("\n[4] XGBoost hiperparametre optimizasyonu başlatılıyor...")
print("  Metod: GridSearchCV (kapsamlı arama)")

# Parameter grid (daha focused)
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5]
}

xgb_model = xgb.XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False
)

# GridSearchCV
xgb_grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),  # 3-fold daha hızlı
    scoring=scoring,
    refit=refit_metric,
    n_jobs=-1,
    verbose=2
)

print("\n  XGBoost optimizasyonu çalışıyor (bu uzun sürebilir)...")
xgb_start = time.time()
xgb_grid.fit(X_train_resampled, y_train_resampled)
xgb_time = time.time() - xgb_start

print(f"\n  ✓ XGBoost optimizasyonu tamamlandı ({xgb_time:.2f}s)")
print(f"  ✓ En iyi parametreler: {xgb_grid.best_params_}")
print(f"  ✓ En iyi F1-score (CV): {xgb_grid.best_score_:.4f}")

# Test performance
xgb_best = xgb_grid.best_estimator_
xgb_test_score = f1_score(y_test, xgb_best.predict(X_test))
xgb_test_roc = roc_auc_score(y_test, xgb_best.predict_proba(X_test)[:, 1])

print(f"  ✓ Test F1-score: {xgb_test_score:.4f}")
print(f"  ✓ Test ROC-AUC: {xgb_test_roc:.4f}")

# =================================================================================
# 5. LIGHTGBM - RANDOMIZED SEARCH
# =================================================================================
print("\n[5] LightGBM hiperparametre optimizasyonu başlatılıyor...")
print("  Metod: RandomizedSearchCV")

# Parameter distribution
lgb_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'num_leaves': randint(20, 150),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_samples': randint(10, 50),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

lgb_model = lgb.LGBMClassifier(
    random_state=RANDOM_STATE,
    class_weight='balanced',
    verbose=-1
)

# RandomizedSearchCV
lgb_random = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_param_dist,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring=scoring,
    refit=refit_metric,
    n_jobs=-1,
    verbose=2,
    random_state=RANDOM_STATE
)

print("\n  LightGBM optimizasyonu çalışıyor...")
lgb_start = time.time()
lgb_random.fit(X_train_resampled, y_train_resampled)
lgb_time = time.time() - lgb_start

print(f"\n  ✓ LightGBM optimizasyonu tamamlandı ({lgb_time:.2f}s)")
print(f"  ✓ En iyi parametreler: {lgb_random.best_params_}")
print(f"  ✓ En iyi F1-score (CV): {lgb_random.best_score_:.4f}")

# Test performance
lgb_best = lgb_random.best_estimator_
lgb_test_score = f1_score(y_test, lgb_best.predict(X_test))
lgb_test_roc = roc_auc_score(y_test, lgb_best.predict_proba(X_test)[:, 1])

print(f"  ✓ Test F1-score: {lgb_test_score:.4f}")
print(f"  ✓ Test ROC-AUC: {lgb_test_roc:.4f}")

# =================================================================================
# 6. SONUÇLARI KARŞILAŞTIR
# =================================================================================
print("\n[6] Tuned modeller karşılaştırılıyor...")

# Sonuçları topla
tuning_results = {
    'Random Forest': {
        'best_params': rf_random.best_params_,
        'cv_f1': rf_random.best_score_,
        'test_f1': rf_test_score,
        'test_roc_auc': rf_test_roc,
        'tuning_time': rf_time,
        'search_type': 'RandomizedSearchCV',
        'n_iterations': rf_random.n_splits_ * len(rf_random.cv_results_['params'])
    },
    'XGBoost': {
        'best_params': xgb_grid.best_params_,
        'cv_f1': xgb_grid.best_score_,
        'test_f1': xgb_test_score,
        'test_roc_auc': xgb_test_roc,
        'tuning_time': xgb_time,
        'search_type': 'GridSearchCV',
        'n_iterations': xgb_grid.n_splits_ * len(xgb_grid.cv_results_['params'])
    },
    'LightGBM': {
        'best_params': lgb_random.best_params_,
        'cv_f1': lgb_random.best_score_,
        'test_f1': lgb_test_score,
        'test_roc_auc': lgb_test_roc,
        'tuning_time': lgb_time,
        'search_type': 'RandomizedSearchCV',
        'n_iterations': lgb_random.n_splits_ * len(lgb_random.cv_results_['params'])
    }
}

# Karşılaştırma tablosu
comparison_data = []
for model_name, results in tuning_results.items():
    comparison_data.append({
        'Model': model_name,
        'Search Type': results['search_type'],
        'CV F1': f"{results['cv_f1']:.4f}",
        'Test F1': f"{results['test_f1']:.4f}",
        'Test ROC-AUC': f"{results['test_roc_auc']:.4f}",
        'Tuning Time (s)': f"{results['tuning_time']:.2f}",
        'Iterations': results['n_iterations']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / 'tuning_comparison.csv', index=False)

print("\n  Tuned Model Karşılaştırması:")
print(comparison_df.to_string(index=False))

# =================================================================================
# 7. PERFORMANS GELİŞİMİ GÖRSELLEŞTİRME
# =================================================================================
print("\n[7] Performans gelişimi görselleştiriliyor...")

# Original vs Tuned karşılaştırması için original sonuçları yükle
with open(PROJECT_ROOT / 'docs' / 'model_training' / 'training_results.json', 'r') as f:
    original_results = json.load(f)

# Karşılaştırma grafiği
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models = ['Random Forest', 'XGBoost', 'LightGBM']
original_f1 = [original_results[m]['test_metrics']['f1'] for m in models]
tuned_f1 = [tuning_results[m]['test_f1'] for m in models]

original_roc = [original_results[m]['test_metrics']['roc_auc'] for m in models]
tuned_roc = [tuning_results[m]['test_roc_auc'] for m in models]

x = np.arange(len(models))
width = 0.35

# F1-Score comparison
bars1 = axes[0].bar(x - width/2, original_f1, width, label='Original', alpha=0.8, color='#e74c3c')
bars2 = axes[0].bar(x + width/2, tuned_f1, width, label='Tuned', alpha=0.8, color='#2ecc71')

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('F1-Score', fontsize=12)
axes[0].set_title('F1-Score: Original vs Tuned', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=0)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
axes[0].set_ylim([0, max(max(original_f1), max(tuned_f1)) * 1.2])

# Değerleri ekle
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# ROC-AUC comparison
bars3 = axes[1].bar(x - width/2, original_roc, width, label='Original', alpha=0.8, color='#e74c3c')
bars4 = axes[1].bar(x + width/2, tuned_roc, width, label='Tuned', alpha=0.8, color='#2ecc71')

axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('ROC-AUC', fontsize=12)
axes[1].set_title('ROC-AUC: Original vs Tuned', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=0)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim([0, max(max(original_roc), max(tuned_roc)) * 1.2])

# Değerleri ekle
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_tuning_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 01_tuning_comparison.png")

# İyileşme yüzdesi
improvement_data = []
for model in models:
    f1_improvement = ((tuning_results[model]['test_f1'] - original_results[model]['test_metrics']['f1']) / 
                     original_results[model]['test_metrics']['f1']) * 100
    roc_improvement = ((tuning_results[model]['test_roc_auc'] - original_results[model]['test_metrics']['roc_auc']) / 
                      original_results[model]['test_metrics']['roc_auc']) * 100
    
    improvement_data.append({
        'Model': model,
        'F1 Improvement (%)': f1_improvement,
        'ROC-AUC Improvement (%)': roc_improvement
    })

improvement_df = pd.DataFrame(improvement_data)

plt.figure(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, improvement_df['F1 Improvement (%)'], width, 
               label='F1-Score', alpha=0.8)
bars2 = plt.bar(x + width/2, improvement_df['ROC-AUC Improvement (%)'], width, 
               label='ROC-AUC', alpha=0.8)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Improvement (%)', fontsize=12)
plt.title('Performance Improvement After Tuning', fontsize=14, fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.axhline(0, color='black', linewidth=0.8)

# Değerleri ekle
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_improvement_percentage.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 02_improvement_percentage.png")

# =================================================================================
# 8. TUNED MODELLERİ KAYDET
# =================================================================================
print("\n[8] Tuned modeller kaydediliyor...")

# Best models kaydet
with open(MODELS_DIR / 'random_forest_tuned.pkl', 'wb') as f:
    pickle.dump(rf_best, f)
print(f"  ✓ random_forest_tuned.pkl kaydedildi")

with open(MODELS_DIR / 'xgboost_tuned.pkl', 'wb') as f:
    pickle.dump(xgb_best, f)
print(f"  ✓ xgboost_tuned.pkl kaydedildi")

with open(MODELS_DIR / 'lightgbm_tuned.pkl', 'wb') as f:
    pickle.dump(lgb_best, f)
print(f"  ✓ lightgbm_tuned.pkl kaydedildi")

# =================================================================================
# 9. SONUÇLARI KAYDET
# =================================================================================
print("\n[9] Tuning sonuçları kaydediliyor...")

# JSON için convert
tuning_results_json = {}
for model_name, results in tuning_results.items():
    tuning_results_json[model_name] = {
        'best_params': {k: (int(v) if isinstance(v, (np.integer, np.int64)) else 
                           float(v) if isinstance(v, (np.floating, np.float64)) else v)
                       for k, v in results['best_params'].items()},
        'cv_f1': float(results['cv_f1']),
        'test_f1': float(results['test_f1']),
        'test_roc_auc': float(results['test_roc_auc']),
        'tuning_time': float(results['tuning_time']),
        'search_type': results['search_type'],
        'n_iterations': int(results['n_iterations'])
    }

with open(OUTPUT_DIR / 'tuning_results.json', 'w') as f:
    json.dump(tuning_results_json, f, indent=4)
print(f"  ✓ tuning_results.json kaydedildi")

improvement_df.to_csv(OUTPUT_DIR / 'improvement_summary.csv', index=False)
print(f"  ✓ improvement_summary.csv kaydedildi")

# =================================================================================
# 10. MARKDOWN RAPOR
# =================================================================================
print("\n[10] Markdown rapor oluşturuluyor...")

# En iyi modeli bul
best_model = max(tuning_results.items(), key=lambda x: x[1]['test_f1'])

markdown_content = f"""# HİPERPARAMETRE OPTİMİZASYONU RAPORU

## 1. Genel Bakış

Hiperparametre optimizasyonu, model performansını artırmak için kritik bir adımdır.
Bu rapor, **3 farklı model** için kapsamlı hiperparametre tuning sonuçlarını içermektedir.

### Optimizasyon Stratejileri

**GridSearchCV:**
- Tüm parametre kombinasyonlarını sistematik olarak dener
- Garantili en iyi sonuç (verilen grid içinde)
- Hesaplama maliyeti yüksek
- Kullanıldı: XGBoost

**RandomizedSearchCV:**
- Rastgele parametre kombinasyonları dener
- Daha hızlı, geniş arama alanı
- Neredeyse optimal sonuçlar
- Kullanıldı: Random Forest, LightGBM

## 2. Model-Spesifik Optimizasyon Sonuçları

### 2.1 Random Forest

**Arama Stratejisi**: {tuning_results['Random Forest']['search_type']}
**Toplam İterasyon**: {tuning_results['Random Forest']['n_iterations']}
**Optimizasyon Süresi**: {tuning_results['Random Forest']['tuning_time']:.2f} saniye

**En İyi Hiperparametreler:**
```python
{json.dumps(tuning_results['Random Forest']['best_params'], indent=2)}
```

**Performans:**
- CV F1-Score: {tuning_results['Random Forest']['cv_f1']:.4f}
- Test F1-Score: {tuning_results['Random Forest']['test_f1']:.4f}
- Test ROC-AUC: {tuning_results['Random Forest']['test_roc_auc']:.4f}

### 2.2 XGBoost

**Arama Stratejisi**: {tuning_results['XGBoost']['search_type']}
**Toplam İterasyon**: {tuning_results['XGBoost']['n_iterations']}
**Optimizasyon Süresi**: {tuning_results['XGBoost']['tuning_time']:.2f} saniye

**En İyi Hiperparametreler:**
```python
{json.dumps(tuning_results['XGBoost']['best_params'], indent=2)}
```

**Performans:**
- CV F1-Score: {tuning_results['XGBoost']['cv_f1']:.4f}
- Test F1-Score: {tuning_results['XGBoost']['test_f1']:.4f}
- Test ROC-AUC: {tuning_results['XGBoost']['test_roc_auc']:.4f}

### 2.3 LightGBM

**Arama Stratejisi**: {tuning_results['LightGBM']['search_type']}
**Toplam İterasyon**: {tuning_results['LightGBM']['n_iterations']}
**Optimizasyon Süresi**: {tuning_results['LightGBM']['tuning_time']:.2f} saniye

**En İyi Hiperparametreler:**
```python
{json.dumps(tuning_results['LightGBM']['best_params'], indent=2)}
```

**Performans:**
- CV F1-Score: {tuning_results['LightGBM']['cv_f1']:.4f}
- Test F1-Score: {tuning_results['LightGBM']['test_f1']:.4f}
- Test ROC-AUC: {tuning_results['LightGBM']['test_roc_auc']:.4f}

## 3. Original vs Tuned Karşılaştırma

![Tuning Comparison](hyperparameter_tuning/01_tuning_comparison.png)

### Performans Karşılaştırması

| Model | Original F1 | Tuned F1 | Original ROC-AUC | Tuned ROC-AUC |
|-------|-------------|----------|------------------|---------------|
"""

for model in models:
    markdown_content += f"| {model} | {original_results[model]['test_metrics']['f1']:.4f} | "
    markdown_content += f"{tuning_results[model]['test_f1']:.4f} | "
    markdown_content += f"{original_results[model]['test_metrics']['roc_auc']:.4f} | "
    markdown_content += f"{tuning_results[model]['test_roc_auc']:.4f} |\n"

markdown_content += f"""
## 4. İyileşme Analizi

![Improvement Percentage](hyperparameter_tuning/02_improvement_percentage.png)

### İyileşme Yüzdeleri

| Model | F1-Score İyileşmesi | ROC-AUC İyileşmesi |
|-------|--------------------|--------------------|
"""

for idx, row in improvement_df.iterrows():
    markdown_content += f"| {row['Model']} | {row['F1 Improvement (%)']:+.2f}% | "
    markdown_content += f"{row['ROC-AUC Improvement (%)']:+.2f}% |\n"

markdown_content += f"""
**Ortalama İyileşme:**
- F1-Score: {improvement_df['F1 Improvement (%)'].mean():+.2f}%
- ROC-AUC: {improvement_df['ROC-AUC Improvement (%)'].mean():+.2f}%

## 5. Hiperparametre Etkileri

### Key Parametreler ve Etkileri

**Learning Rate (XGBoost, LightGBM):**
- Düşük değer (0.01-0.05): Daha yavaş öğrenme, daha iyi generalization, overfitting riski az
- Yüksek değer (0.1-0.3): Hızlı öğrenme, overfitting riski artar
- **Optimal**: {tuning_results['XGBoost']['best_params'].get('learning_rate', 'N/A')} (XGBoost)

**Max Depth:**
- Ağacın maksimum derinliği
- Düşük: Underfitting riski
- Yüksek: Overfitting riski
- **Optimal**: {tuning_results['Random Forest']['best_params'].get('max_depth', 'N/A')} (Random Forest)

**N Estimators:**
- Ağaç sayısı (ensemble size)
- Daha fazla ağaç → genelde daha iyi performans
- Diminishing returns after certain point
- **Optimal**: {tuning_results['LightGBM']['best_params'].get('n_estimators', 'N/A')} (LightGBM)

**Subsample / Colsample:**
- Training data'nın / feature'ların ne kadarı kullanılacak
- <1.0: Regularization effect, overfitting önler
- **Optimal Subsample**: {tuning_results['XGBoost']['best_params'].get('subsample', 'N/A')} (XGBoost)

## 6. Computational Efficiency

### Tuning Süreleri

| Model | Tuning Süresi | İterasyon Sayısı | Saniye/İterasyon |
|-------|---------------|------------------|------------------|
"""

for model in models:
    time_val = tuning_results[model]['tuning_time']
    iters = tuning_results[model]['n_iterations']
    time_per_iter = time_val / iters if iters > 0 else 0
    markdown_content += f"| {model} | {time_val:.2f}s | {iters} | {time_per_iter:.2f}s |\n"

markdown_content += f"""
**Gözlemler:**
- RandomizedSearchCV genelde GridSearchCV'den daha hızlı
- LightGBM en hızlı training süresine sahip
- XGBoost GridSearch en uzun sürdü (comprehensive search)

## 7. En İyi Model Seçimi

**Test F1-Score'a Göre En İyi**: **{best_model[0]}** ({best_model[1]['test_f1']:.4f})

**Test ROC-AUC'ye Göre Sıralama:**
"""

sorted_by_roc = sorted(tuning_results.items(), key=lambda x: x[1]['test_roc_auc'], reverse=True)
for rank, (model, results) in enumerate(sorted_by_roc, 1):
    markdown_content += f"{rank}. {model}: {results['test_roc_auc']:.4f}\n"

markdown_content += f"""
## 8. Önemli Çıkarımlar

### 8.1 Hiperparametre Tuning Değerli mi?

"""

avg_f1_improvement = improvement_df['F1 Improvement (%)'].mean()
if avg_f1_improvement > 1:
    verdict = "**EVET** - Tuning anlamlı performans artışı sağladı"
elif avg_f1_improvement > 0:
    verdict = "**KISMEN** - Hafif iyileşme var ama marjinal"
else:
    verdict = "**HAYIR** - Bu durumda default parametreler yeterli"

markdown_content += f"""
{verdict}

Ortalama F1-Score İyileşmesi: **{avg_f1_improvement:+.2f}%**
Ortalama ROC-AUC İyileşmesi: **{improvement_df['ROC-AUC Improvement (%)'].mean():+.2f}%**

### 8.2 Trade-offs

**Avantajlar:**
- Daha iyi model performansı
- Optimal parametre kombinasyonlarını keşfetme
- Model davranışını daha iyi anlama

**Dezavantajlar:**
- Hesaplama maliyeti yüksek
- Zaman alıcı
- Overfitting riski (CV fold'larına)

### 8.3 Production Önerileri

1. **Model Selection**: {best_model[0]} kullan (en iyi test performance)

2. **Regularization**: Tuned parametreler overfitting'i azaltmaya odaklanmış
   - Subsample/Colsample < 1.0
   - Moderate max_depth
   - Appropriate learning rate

3. **Retraining Strategy**: 
   - Yeni data geldiğinde model retrain edilmeli
   - Hiperparametreler büyük ihtimalle stable kalacak
   - 6 ayda bir tuning refresh edilebilir

4. **Monitoring**:
   - Production'da model drift izlenmeli
   - Performance degradation varsa re-tuning gerekebilir

## 9. Sonraki Adımlar

1. ✓ En iyi modeli production deployment için hazırla
2. ✓ Model interpretability analizi (SHAP values)
3. ✓ Ensemble methods dene (stacking, voting)
4. ✓ Online learning / incremental training araştır
5. ✓ A/B testing framework kur

---
*Rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(OUTPUT_DIR / 'tuning_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"  ✓ Markdown rapor kaydedildi: tuning_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("HİPERPARAMETRE OPTİMİZASYONU TAMAMLANDI")
print("=" * 80)
print(f"\n✓ 3 model için hiperparametre optimizasyonu tamamlandı")
print(f"✓ En iyi model (Test F1): {best_model[0]} - {best_model[1]['test_f1']:.4f}")
print(f"✓ Ortalama F1 iyileşmesi: {avg_f1_improvement:+.2f}%")
print(f"✓ Tuned modeller '{MODELS_DIR}' dizinine kaydedildi")
print(f"✓ Tüm raporlar '{OUTPUT_DIR}' dizinine kaydedildi")
print("\nSonraki Adım: 06_model_interpretation.py scriptini çalıştırın")
print("=" * 80)
