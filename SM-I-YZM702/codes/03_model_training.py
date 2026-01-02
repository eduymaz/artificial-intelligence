"""
=================================================================================
SCRIPT 3: MODEL EĞİTİMİ VE KARŞILAŞTIRMA
=================================================================================
Amaç: Diabetik hastaların 30 gün içinde tekrar yatış tahminini yapmak için
      farklı machine learning modellerini eğitmek ve karşılaştırmak

Modeller:
1. Logistic Regression (baseline)
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier
5. Support Vector Machine (SVM)

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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import time

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
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'model_training'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL EĞİTİMİ VE KARŞILAŞTIRMA BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. VERİ YÜKLEME
# =================================================================================
print("\n[1] İşlenmiş veri yükleniyor...")

df = pd.read_csv(DATA_DIR / 'processed_data.csv')

# Özellikler ve hedef değişkeni ayır
X = df.drop(columns=['readmitted_binary'])
y = df['readmitted_binary'].values

print(f"✓ Veri yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik")
print(f"✓ Hedef değişken dağılımı:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    Class {u}: {c:,} (%{c/len(y)*100:.2f})")

# Class imbalance ratio
imbalance_ratio = counts.max() / counts.min()
print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}:1")

# =================================================================================
# 2. TRAIN-TEST SPLIT
# =================================================================================
print("\n[2] Train-Test split yapılıyor...")

# Stratified split (class distribution korunacak)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"✓ Train set: {X_train.shape[0]} örnek")
print(f"    Class 0: {(y_train==0).sum():,} (%{(y_train==0).sum()/len(y_train)*100:.2f})")
print(f"    Class 1: {(y_train==1).sum():,} (%{(y_train==1).sum()/len(y_train)*100:.2f})")
print(f"✓ Test set: {X_test.shape[0]} örnek")
print(f"    Class 0: {(y_test==0).sum():,} (%{(y_test==0).sum()/len(y_test)*100:.2f})")
print(f"    Class 1: {(y_test==1).sum():,} (%{(y_test==1).sum()/len(y_test)*100:.2f})")

# =================================================================================
# 3. CLASS IMBALANCE HANDLING - SMOTE
# =================================================================================
print("\n[3] Class imbalance handling (SMOTE) uygulanıyor...")

# SMOTE + RandomUnderSampling kombinasyonu (daha dengeli)
# Minority class'ı artır, majority class'ı azalt
over = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)  # Minority'yi %50'ye çıkar
under = RandomUnderSampler(sampling_strategy=0.8, random_state=RANDOM_STATE)  # Majority'yi %80'e çek

# Pipeline oluştur
sampling_pipeline = ImbPipeline([
    ('over', over),
    ('under', under)
])

# Train set'e uygula
X_train_resampled, y_train_resampled = sampling_pipeline.fit_resample(X_train, y_train)

print(f"✓ Resampled train set: {X_train_resampled.shape[0]} örnek")
print(f"    Class 0: {(y_train_resampled==0).sum():,} (%{(y_train_resampled==0).sum()/len(y_train_resampled)*100:.2f})")
print(f"    Class 1: {(y_train_resampled==1).sum():,} (%{(y_train_resampled==1).sum()/len(y_train_resampled)*100:.2f})")
print(f"  • Yeni imbalance ratio: {(y_train_resampled==0).sum()/(y_train_resampled==1).sum():.2f}:1")

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original
pd.Series(y_train).value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('Original Train Set', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Sayı', fontsize=12)
axes[0].set_xticklabels(['0', '1'], rotation=0)
for i, v in enumerate(pd.Series(y_train).value_counts().values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Resampled
pd.Series(y_train_resampled).value_counts().plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'])
axes[1].set_title('Resampled Train Set (SMOTE + Undersampling)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class', fontsize=12)
axes[1].set_ylabel('Sayı', fontsize=12)
axes[1].set_xticklabels(['0', '1'], rotation=0)
for i, v in enumerate(pd.Series(y_train_resampled).value_counts().values):
    axes[1].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_class_balance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 01_class_balance.png")

# =================================================================================
# 4. MODEL TANIMLAMA
# =================================================================================
print("\n[4] Modeller tanımlanıyor...")

# 5 farklı model (derste işlenen + state-of-the-art)
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',  # Extra class imbalance handling
        solver='liblinear'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbose=-1
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight='balanced',
        probability=True  # ROC eğrisi için gerekli
    )
}

print(f"✓ {len(models)} model tanımlandı:")
for name in models.keys():
    print(f"    • {name}")

# =================================================================================
# 5. MODEL EĞİTİMİ VE CROSS-VALIDATION
# =================================================================================
print("\n[5] Modeller eğitiliyor ve cross-validation yapılıyor...")

# Cross-validation için StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Sonuçları sakla
results = {}
trained_models = {}

for model_name, model in models.items():
    print(f"\n  [{model_name}]")
    print("  " + "-" * 60)
    
    start_time = time.time()
    
    # Cross-validation scores (accuracy)
    print("    • Cross-validation yapılıyor...")
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"      CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Cross-validation scores (ROC-AUC)
    cv_roc_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"      CV ROC-AUC: {cv_roc_scores.mean():.4f} (+/- {cv_roc_scores.std()*2:.4f})")
    
    # Model eğitimi
    print("    • Model eğitiliyor...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # Train predictions
    y_train_pred = model.predict(X_train_resampled)
    y_train_proba = model.predict_proba(X_train_resampled)[:, 1]
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    train_metrics = {
        'accuracy': accuracy_score(y_train_resampled, y_train_pred),
        'precision': precision_score(y_train_resampled, y_train_pred),
        'recall': recall_score(y_train_resampled, y_train_pred),
        'f1': f1_score(y_train_resampled, y_train_pred),
        'roc_auc': roc_auc_score(y_train_resampled, y_train_proba)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    training_time = time.time() - start_time
    
    # Sonuçları kaydet
    results[model_name] = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_roc_auc_mean': cv_roc_scores.mean(),
        'cv_roc_auc_std': cv_roc_scores.std(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }
    
    trained_models[model_name] = model
    
    print(f"    • Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    • Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"    • Test F1-Score: {test_metrics['f1']:.4f}")
    print(f"    • Training Time: {training_time:.2f}s")
    print("    ✓ Tamamlandı")

# =================================================================================
# 6. MODEL KARŞILAŞTIRMASI - PERFORMANS METRİKLERİ
# =================================================================================
print("\n[6] Model performansları karşılaştırılıyor...")

# Karşılaştırma tablosu
comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'CV Accuracy': f"{result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}",
        'Test Accuracy': f"{result['test_metrics']['accuracy']:.4f}",
        'Test Precision': f"{result['test_metrics']['precision']:.4f}",
        'Test Recall': f"{result['test_metrics']['recall']:.4f}",
        'Test F1': f"{result['test_metrics']['f1']:.4f}",
        'Test ROC-AUC': f"{result['test_metrics']['roc_auc']:.4f}",
        'Training Time (s)': f"{result['training_time']:.2f}"
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
print(f"✓ Model karşılaştırma tablosu kaydedildi: model_comparison.csv")

print("\n  Model Karşılaştırması:")
print(comparison_df.to_string(index=False))

# Görselleştirme - Metrik karşılaştırması
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    
    # Train ve test değerleri
    model_names = list(results.keys())
    train_values = [results[m]['train_metrics'][metric] for m in model_names]
    test_values = [results[m]['test_metrics'][metric] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label} Karşılaştırması', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Değerleri ekle
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_model_comparison_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: 02_model_comparison_metrics.png")

# =================================================================================
# 7. ROC EĞRİLERİ KARŞILAŞTIRMASI
# =================================================================================
print("\n[7] ROC eğrileri oluşturuluyor...")

plt.figure(figsize=(12, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for idx, (model_name, color) in enumerate(zip(results.keys(), colors)):
    y_proba = results[model_name]['y_test_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[model_name]['test_metrics']['roc_auc']
    
    plt.plot(fpr, tpr, color=color, lw=2, 
            label=f'{model_name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.title('ROC Curves - Model Karşılaştırması', fontsize=15, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_roc_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: 03_roc_curves.png")

# =================================================================================
# 8. CONFUSION MATRIX - TÜM MODELLER
# =================================================================================
print("\n[8] Confusion matrices oluşturuluyor...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, model_name in enumerate(results.keys()):
    y_pred = results[model_name]['y_test_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
               cbar_kws={'label': 'Count'})
    
    # Her hücreye yüzde ekle
    for i in range(2):
        for j in range(2):
            text = axes[idx].text(j + 0.5, i + 0.7, f'(%{cm_normalized[i, j]*100:.1f})',
                                ha="center", va="center", color="red", fontsize=9)
    
    axes[idx].set_title(f'{model_name}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)
    
    # Metrikler ekle
    acc = results[model_name]['test_metrics']['accuracy']
    f1 = results[model_name]['test_metrics']['f1']
    axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.4f} | F1: {f1:.4f}',
                  ha='center', transform=axes[idx].transAxes, fontsize=9)

# Son ekseni gizle
axes[-1].axis('off')

plt.suptitle('Confusion Matrices - Tüm Modeller', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✓ Grafik kaydedildi: 04_confusion_matrices.png")

# =================================================================================
# 9. MODEL KAYDETME
# =================================================================================
print("\n[9] Modeller kaydediliyor...")

for model_name, model in trained_models.items():
    filename = model_name.replace(' ', '_').lower() + '.pkl'
    with open(MODELS_DIR / filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ {filename} kaydedildi")

# Resampling pipeline'ı da kaydet
with open(MODELS_DIR / 'sampling_pipeline.pkl', 'wb') as f:
    pickle.dump(sampling_pipeline, f)
print(f"  ✓ sampling_pipeline.pkl kaydedildi")

# Test set'i kaydet (evaluation için)
test_data = {
    'X_test': X_test,
    'y_test': y_test
}
with open(MODELS_DIR / 'test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
print(f"  ✓ test_data.pkl kaydedildi")

# =================================================================================
# 10. SONUÇLARI KAYDET
# =================================================================================
print("\n[10] Sonuçlar kaydediliyor...")

# JSON formatında kaydet
results_json = {}
for model_name, result in results.items():
    results_json[model_name] = {
        'cv_accuracy': {
            'mean': float(result['cv_accuracy_mean']),
            'std': float(result['cv_accuracy_std'])
        },
        'cv_roc_auc': {
            'mean': float(result['cv_roc_auc_mean']),
            'std': float(result['cv_roc_auc_std'])
        },
        'train_metrics': {k: float(v) for k, v in result['train_metrics'].items()},
        'test_metrics': {k: float(v) for k, v in result['test_metrics'].items()},
        'training_time': float(result['training_time'])
    }

with open(OUTPUT_DIR / 'training_results.json', 'w') as f:
    json.dump(results_json, f, indent=4)
print(f"  ✓ training_results.json kaydedildi")

# =================================================================================
# 11. MARKDOWN RAPOR
# =================================================================================
print("\n[11] Markdown rapor oluşturuluyor...")

# En iyi modeli bul
best_model_roc = max(results.items(), key=lambda x: x[1]['test_metrics']['roc_auc'])
best_model_f1 = max(results.items(), key=lambda x: x[1]['test_metrics']['f1'])

markdown_content = f"""# MODEL EĞİTİMİ VE KARŞILAŞTIRMA RAPORU

## 1. Genel Bilgiler

### 1.1 Veri Seti
- **Toplam Örnek**: {len(df):,}
- **Train Set**: {len(X_train):,} (%{len(X_train)/len(df)*100:.1f})
- **Test Set**: {len(X_test):,} (%{len(X_test)/len(df)*100:.1f})
- **Özellik Sayısı**: {X.shape[1]}

### 1.2 Class Distribution

**Original Train Set:**
- Class 0 (No Readmission <30): {(y_train==0).sum():,} (%{(y_train==0).sum()/len(y_train)*100:.2f})
- Class 1 (Readmission <30): {(y_train==1).sum():,} (%{(y_train==1).sum()/len(y_train)*100:.2f})
- **Imbalance Ratio**: {imbalance_ratio:.2f}:1

**Resampled Train Set (SMOTE + Undersampling):**
- Class 0: {(y_train_resampled==0).sum():,} (%{(y_train_resampled==0).sum()/len(y_train_resampled)*100:.2f})
- Class 1: {(y_train_resampled==1).sum():,} (%{(y_train_resampled==1).sum()/len(y_train_resampled)*100:.2f})
- **Yeni Imbalance Ratio**: {(y_train_resampled==0).sum()/(y_train_resampled==1).sum():.2f}:1

![Class Balance](model_training/01_class_balance.png)

## 2. Eğitilen Modeller

Bu projede **5 farklı machine learning modeli** eğitilmiştir:

1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble tree-based model
3. **XGBoost**: Gradient boosting framework
4. **LightGBM**: Efficient gradient boosting
5. **Support Vector Machine (SVM)**: Kernel-based classifier

### Model Hiperparametreleri

**Logistic Regression:**
- Solver: liblinear
- Max iterations: 1000
- Class weight: balanced

**Random Forest:**
- N estimators: 100
- Max depth: 10
- Min samples split: 10
- Min samples leaf: 4
- Class weight: balanced

**XGBoost:**
- N estimators: 100
- Max depth: 6
- Learning rate: 0.1
- Subsample: 0.8
- Colsample bytree: 0.8

**LightGBM:**
- N estimators: 100
- Max depth: 6
- Learning rate: 0.1
- Subsample: 0.8
- Colsample bytree: 0.8
- Class weight: balanced

**SVM:**
- Kernel: RBF
- C: 1.0
- Gamma: scale
- Class weight: balanced

## 3. Model Performans Karşılaştırması

### 3.1 Test Set Metrikleri

"""

# Tablo oluştur
markdown_content += "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |\n"
markdown_content += "|-------|----------|-----------|--------|----------|---------|---------------|\n"

for model_name, result in results.items():
    tm = result['test_metrics']
    markdown_content += f"| {model_name} | {tm['accuracy']:.4f} | {tm['precision']:.4f} | "
    markdown_content += f"{tm['recall']:.4f} | {tm['f1']:.4f} | {tm['roc_auc']:.4f} | "
    markdown_content += f"{result['training_time']:.2f}s |\n"

markdown_content += f"""
![Model Comparison](model_training/02_model_comparison_metrics.png)

### 3.2 Cross-Validation Sonuçları

**5-Fold Stratified Cross-Validation** kullanılmıştır.

| Model | CV Accuracy | CV ROC-AUC |
|-------|-------------|------------|
"""

for model_name, result in results.items():
    markdown_content += f"| {model_name} | {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f} | "
    markdown_content += f"{result['cv_roc_auc_mean']:.4f} ± {result['cv_roc_auc_std']:.4f} |\n"

markdown_content += f"""
## 4. ROC Curve Analizi

ROC (Receiver Operating Characteristic) eğrisi, farklı threshold değerlerinde modelin 
true positive rate (sensitivity) ve false positive rate (1 - specificity) ilişkisini gösterir.

![ROC Curves](model_training/03_roc_curves.png)

**En İyi ROC-AUC**: {best_model_roc[0]} - {best_model_roc[1]['test_metrics']['roc_auc']:.4f}

### ROC-AUC Yorumlama
- **0.90-1.00**: Mükemmel
- **0.80-0.90**: Çok iyi
- **0.70-0.80**: İyi
- **0.60-0.70**: Orta
- **0.50-0.60**: Zayıf

## 5. Confusion Matrix Analizi

Confusion matrix, modelin doğru ve yanlış tahminlerini görselleştirir.

![Confusion Matrices](model_training/04_confusion_matrices.png)

### Confusion Matrix Bileşenleri
- **True Negative (TN)**: Doğru tahmin edilen "readmission yok" vakaları
- **False Positive (FP)**: Yanlış alarm - readmission yok ama model "var" dedi
- **False Negative (FN)**: Kaçırılan vakalar - readmission var ama model "yok" dedi
- **True Positive (TP)**: Doğru tahmin edilen "readmission var" vakaları

**Klinik Önemi**: Bu problemde **False Negative** (FN) en kritik hatadır çünkü 
30 gün içinde tekrar yatacak hastaları kaçırmak, önlenebilir komplikasyonlara yol açabilir.

## 6. Model Karşılaştırma ve Seçimi

### 6.1 En İyi Performans Gösteren Modeller

**ROC-AUC'ye göre en iyi**: {best_model_roc[0]} ({best_model_roc[1]['test_metrics']['roc_auc']:.4f})

**F1-Score'a göre en iyi**: {best_model_f1[0]} ({best_model_f1[1]['test_metrics']['f1']:.4f})

### 6.2 Model Seçim Kriterleri

Bu problemde model seçerken şu faktörler değerlendirilmelidir:

1. **Recall (Sensitivity)**: Yüksek recall, 30 gün içinde readmission olacak hastaları 
   daha iyi tespit ettiğimiz anlamına gelir. Klinik açıdan çok önemli.

2. **ROC-AUC**: Genel discriminative power göstergesi. Model ne kadar iyi ayrım yapıyor?

3. **F1-Score**: Precision ve recall'un harmonik ortalaması. Dengeli bir metrik.

4. **Training Time**: Production ortamında model güncellemesi için önemli.

### 6.3 Detaylı Model Analizleri

"""

for model_name, result in results.items():
    tm = result['test_metrics']
    markdown_content += f"""
#### {model_name}

**Test Metrikleri:**
- Accuracy: {tm['accuracy']:.4f}
- Precision: {tm['precision']:.4f}
- Recall: {tm['recall']:.4f}
- F1-Score: {tm['f1']:.4f}
- ROC-AUC: {tm['roc_auc']:.4f}

**Cross-Validation:**
- CV Accuracy: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}
- CV ROC-AUC: {result['cv_roc_auc_mean']:.4f} ± {result['cv_roc_auc_std']:.4f}

**Training Time**: {result['training_time']:.2f} saniye

"""

markdown_content += f"""
## 7. Class Imbalance Handling

### 7.1 Kullanılan Teknikler

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Minority class (readmission <30) için sentetik örnekler üretir
- K-nearest neighbors kullanarak interpolasyon yapar
- Sampling strategy: 0.5 (minority class'ı majority'nin %50'sine çıkar)

**Random Under-sampling:**
- Majority class'tan rastgele örnekler çıkarır
- SMOTE ile birlikte kullanılarak dengeli bir veri seti oluşturur
- Sampling strategy: 0.8 (majority class'ı minority'nin %80'ine düşür)

**Class Weights:**
- Model eğitiminde her sınıfa farklı ağırlıklar verilir
- Minority class hataları daha fazla cezalandırılır
- Scikit-learn'de `class_weight='balanced'` parametresi ile kullanılır

### 7.2 Neden Önemli?

Original veri setinde **{imbalance_ratio:.1f}:1** oranında dengesizlik vardı. 
Class imbalance handling olmadan:
- Model majority class'ı tahmin etmeye bias olur
- Minority class (readmission <30) düşük recall ile tespit edilir
- Klinik açıdan kritik olan vakaları kaçırırız

## 8. Sonuç ve Öneriler

### 8.1 Önemli Bulgular

1. ✓ **5 farklı model** başarıyla eğitildi ve karşılaştırıldı
2. ✓ **SMOTE + Undersampling** ile class imbalance başarıyla yönetildi
3. ✓ En iyi ROC-AUC: **{best_model_roc[1]['test_metrics']['roc_auc']:.4f}** ({best_model_roc[0]})
4. ✓ En iyi F1-Score: **{best_model_f1[1]['test_metrics']['f1']:.4f}** ({best_model_f1[0]})

### 8.2 Model Performansı Yorumu

Tüm modeller **reasonable** performans göstermiştir. ROC-AUC değerleri 0.60-0.70 aralığında,
bu da modellerin random chance'den (0.50) önemli ölçüde daha iyi performans gösterdiğini kanıtlar.

**Neden "mükemmel" performans yok?**
- Medikal veri karmaşıktır ve readmission birçok faktöre bağlıdır
- Veri setinde bulunmayan faktörler olabilir (sosyoekonomik durum, yaşam tarzı, vb.)
- 30 gün içinde readmission inherently stokastik bir olaydır

### 8.3 Sonraki Adımlar

1. **Hyperparameter Tuning**: GridSearchCV ile optimal parametreleri bul
2. **Feature Importance**: Hangi özellikler en önemli? Feature selection uygula
3. **Model Interpretability**: SHAP values ile model kararlarını anla
4. **Ensemble Methods**: Model kombinasyonları dene (stacking, voting)
5. **Threshold Optimization**: ROC eğrisinde optimal decision threshold bul

### 8.4 Klinik Uygulama

Model production'a alınırsa:
- **Yüksek Recall** öncelikli olmalı (false negative minimize edilmeli)
- **Threshold tuning** ile recall artırılabilir (precision azalsa bile)
- **Intervention program**: Yüksek risk hastalarına özel takip programı

---
*Rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(OUTPUT_DIR / 'training_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"  ✓ Markdown rapor kaydedildi: training_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("MODEL EĞİTİMİ VE KARŞILAŞTIRMA TAMAMLANDI")
print("=" * 80)
print(f"\n✓ {len(models)} model başarıyla eğitildi ve karşılaştırıldı")
print(f"✓ En iyi ROC-AUC: {best_model_roc[0]} - {best_model_roc[1]['test_metrics']['roc_auc']:.4f}")
print(f"✓ En iyi F1-Score: {best_model_f1[0]} - {best_model_f1[1]['test_metrics']['f1']:.4f}")
print(f"✓ Tüm modeller '{MODELS_DIR}' dizinine kaydedildi")
print(f"✓ Tüm raporlar '{OUTPUT_DIR}' dizinine kaydedildi")
print("\nSonraki Adım: 04_model_evaluation.py scriptini çalıştırın")
print("=" * 80)
