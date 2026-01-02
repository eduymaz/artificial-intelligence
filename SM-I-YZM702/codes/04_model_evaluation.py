"""
=================================================================================
SCRIPT 4: DETAYLI MODEL DEĞERLENDİRME
=================================================================================
Amaç: Eğitilen modellerin kapsamlı değerlendirmesi, threshold optimizasyonu,
      feature importance analizi ve detaylı performans raporları

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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report,
                            confusion_matrix, roc_curve, precision_recall_curve,
                            average_precision_score)
from sklearn.calibration import calibration_curve
import shap

warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dizin yapısını oluştur
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'model_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DETAYLI MODEL DEĞERLENDİRME BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. MODELLERİ VE VERİYİ YÜKLE
# =================================================================================
print("\n[1] Modeller ve test verisi yükleniyor...")

# Test verisini yükle
with open(MODELS_DIR / 'test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    X_test = test_data['X_test']
    y_test = test_data['y_test']

print(f"✓ Test seti yüklendi: {len(X_test)} örnek")

# Modelleri yükle
models = {}
model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'xgboost.pkl', 
               'lightgbm.pkl', 'svm.pkl']

for model_file in model_files:
    model_path = MODELS_DIR / model_file
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            models[model_name] = pickle.load(f)
            print(f"  ✓ {model_name} yüklendi")

# =================================================================================
# 2. TAHMINLER OLUŞTUR
# =================================================================================
print("\n[2] Model tahminleri oluşturuluyor...")

predictions = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    predictions[model_name] = {
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    print(f"  ✓ {model_name} tahminleri oluşturuldu")

# =================================================================================
# 3. PRECISION-RECALL EĞRİLERİ
# =================================================================================
print("\n[3] Precision-Recall eğrileri oluşturuluyor...")

plt.figure(figsize=(12, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for (model_name, preds), color in zip(predictions.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, preds['y_proba'])
    avg_precision = average_precision_score(y_test, preds['y_proba'])
    
    plt.plot(recall, precision, color=color, lw=2,
            label=f'{model_name} (AP = {avg_precision:.4f})')

plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.title('Precision-Recall Curves', fontsize=15, fontweight='bold')
plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_precision_recall_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 01_precision_recall_curves.png")

# =================================================================================
# 4. THRESHOLD OPTİMİZASYONU
# =================================================================================
print("\n[4] Threshold optimizasyonu yapılıyor...")

threshold_results = {}

fig, axes = plt.subplots(len(models), 1, figsize=(14, 4*len(models)))
if len(models) == 1:
    axes = [axes]

for idx, (model_name, preds) in enumerate(predictions.items()):
    y_proba = preds['y_proba']
    
    # Farklı threshold değerleri için metrikleri hesapla
    thresholds = np.arange(0.1, 0.9, 0.05)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        accuracies.append(accuracy_score(y_test, y_pred_threshold))
        precisions.append(precision_score(y_test, y_pred_threshold, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_threshold))
        f1_scores.append(f1_score(y_test, y_pred_threshold))
    
    # Optimal threshold'ları bul
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    
    # Youden's J statistic için optimal threshold (ROC curve'den)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    optimal_j_idx = np.argmax(j_scores)
    optimal_j_threshold = roc_thresholds[optimal_j_idx]
    
    threshold_results[model_name] = {
        'optimal_f1_threshold': optimal_f1_threshold,
        'optimal_f1_score': f1_scores[optimal_f1_idx],
        'optimal_j_threshold': optimal_j_threshold,
        'optimal_j_score': j_scores[optimal_j_idx]
    }
    
    # Görselleştirme
    ax = axes[idx]
    ax.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2)
    ax.plot(thresholds, precisions, 's-', label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, '^-', label='Recall', linewidth=2)
    ax.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2)
    
    # Optimal threshold çizgisi
    ax.axvline(optimal_f1_threshold, color='red', linestyle='--', linewidth=2,
              label=f'Optimal F1 Threshold ({optimal_f1_threshold:.3f})')
    
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'{model_name} - Threshold vs Metrics', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_threshold_optimization.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 02_threshold_optimization.png")

# Threshold sonuçlarını yazdır
print("\n  Optimal Threshold Değerleri:")
for model_name, results in threshold_results.items():
    print(f"    {model_name}:")
    print(f"      • F1-optimal: {results['optimal_f1_threshold']:.3f} (F1: {results['optimal_f1_score']:.4f})")
    print(f"      • J-optimal: {results['optimal_j_threshold']:.3f} (J: {results['optimal_j_score']:.4f})")

# =================================================================================
# 5. CALIBRATION CURVES (Model Kalibrasyonu)
# =================================================================================
print("\n[5] Calibration curves oluşturuluyor...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (model_name, preds) in enumerate(predictions.items()):
    y_proba = preds['y_proba']
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='uniform'
    )
    
    # Plot
    axes[idx].plot(mean_predicted_value, fraction_of_positives, 's-',
                  linewidth=2, label=model_name)
    axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    axes[idx].set_xlabel('Mean Predicted Probability', fontsize=11)
    axes[idx].set_ylabel('Fraction of Positives', fontsize=11)
    axes[idx].set_title(f'{model_name} Calibration', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='best', fontsize=10)
    axes[idx].grid(alpha=0.3)
    axes[idx].set_xlim([0, 1])
    axes[idx].set_ylim([0, 1])

# Son ekseni gizle
axes[-1].axis('off')

plt.suptitle('Model Calibration Curves', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_calibration_curves.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 03_calibration_curves.png")

# =================================================================================
# 6. FEATURE IMPORTANCE ANALİZİ
# =================================================================================
print("\n[6] Feature importance analizi yapılıyor...")

# Feature names
with open(DATA_DIR.parent / 'processed' / 'feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

# Tree-based modeller için feature importance
tree_models = {k: v for k, v in models.items() 
              if any(x in k.lower() for x in ['forest', 'xgboost', 'lightgbm'])}

if tree_models:
    fig, axes = plt.subplots(len(tree_models), 1, figsize=(14, 6*len(tree_models)))
    if len(tree_models) == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(tree_models.items()):
        # Feature importance al
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # DataFrame oluştur
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)
            
            # Plot
            ax = axes[idx]
            bars = ax.barh(range(len(importance_df)), importance_df['importance'].values)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'].values)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'{model_name} - Top 20 Feature Importance', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Renk gradyanı
            colors = plt.cm.viridis(importance_df['importance'].values / 
                                   importance_df['importance'].max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Grafik kaydedildi: 04_feature_importance.png")

# Logistic Regression için coefficient importance
if 'Logistic Regression' in models:
    lr_model = models['Logistic Regression']
    coefficients = lr_model.coef_[0]
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False).head(20)
    
    plt.figure(figsize=(14, 8))
    colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
    plt.barh(range(len(coef_df)), coef_df['coefficient'].values, color=colors, alpha=0.7)
    plt.yticks(range(len(coef_df)), coef_df['feature'].values)
    plt.xlabel('Coefficient', fontsize=12)
    plt.title('Logistic Regression - Top 20 Feature Coefficients', 
             fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_logistic_coefficients.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Grafik kaydedildi: 05_logistic_coefficients.png")

# =================================================================================
# 7. CLASSIFICATION REPORTS
# =================================================================================
print("\n[7] Detaylı classification reports oluşturuluyor...")

classification_reports = {}

for model_name, preds in predictions.items():
    report = classification_report(y_test, preds['y_pred'], 
                                   target_names=['No Readmission', 'Readmission <30'],
                                   output_dict=True)
    classification_reports[model_name] = report
    
    # Text dosyası olarak kaydet
    text_report = classification_report(y_test, preds['y_pred'],
                                       target_names=['No Readmission', 'Readmission <30'])
    
    report_filename = model_name.replace(' ', '_').lower() + '_classification_report.txt'
    with open(OUTPUT_DIR / report_filename, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text_report)
    
    print(f"  ✓ {model_name} classification report kaydedildi")

# =================================================================================
# 8. ERROR ANALİZİ
# =================================================================================
print("\n[8] Error analizi yapılıyor...")

# Her model için hata türlerini analiz et
error_analysis = {}

for model_name, preds in predictions.items():
    y_pred = preds['y_pred']
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # False positive ve false negative oranları
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    error_analysis[model_name] = {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# False Positive Rate
model_names = list(error_analysis.keys())
fpr_values = [error_analysis[m]['false_positive_rate'] for m in model_names]
fnr_values = [error_analysis[m]['false_negative_rate'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, fpr_values, width, label='False Positive Rate', 
                   color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x + width/2, fnr_values, width, label='False Negative Rate',
                   color='#3498db', alpha=0.8)

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Error Rate', fontsize=12)
axes[0].set_title('False Positive vs False Negative Rates', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Değerleri ekle
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Error counts
fp_counts = [error_analysis[m]['false_positive'] for m in model_names]
fn_counts = [error_analysis[m]['false_negative'] for m in model_names]

bars3 = axes[1].bar(x - width/2, fp_counts, width, label='False Positives',
                   color='#e74c3c', alpha=0.8)
bars4 = axes[1].bar(x + width/2, fn_counts, width, label='False Negatives',
                   color='#3498db', alpha=0.8)

axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('False Positive vs False Negative Counts', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Değerleri ekle
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_error_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 06_error_analysis.png")

# =================================================================================
# 9. MODEL AGREEMENT ANALİZİ
# =================================================================================
print("\n[9] Model agreement analizi yapılıyor...")

# Tüm modellerin tahminlerini karşılaştır
all_predictions = np.array([preds['y_pred'] for preds in predictions.values()])

# Her örnek için kaç model "1" tahmin etti
agreement_scores = all_predictions.sum(axis=0)

# Agreement distribution
plt.figure(figsize=(12, 6))
agreement_counts = pd.Series(agreement_scores).value_counts().sort_index()

bars = plt.bar(agreement_counts.index, agreement_counts.values, color='#3498db', alpha=0.7)
plt.xlabel('Number of Models Predicting Readmission', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.title('Model Agreement Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(models) + 1))
plt.grid(axis='y', alpha=0.3)

# Değerleri ekle
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_model_agreement.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 07_model_agreement.png")

# High agreement örneklerini analiz et
high_agreement_positive = agreement_scores >= (len(models) * 0.8)  # %80+ modeller "1" dedi
high_agreement_negative = agreement_scores <= (len(models) * 0.2)  # %80+ modeller "0" dedi

print(f"\n  Model Agreement İstatistikleri:")
print(f"    • Yüksek pozitif agreement: {high_agreement_positive.sum()} örnek")
print(f"    • Yüksek negatif agreement: {high_agreement_negative.sum()} örnek")
print(f"    • Düşük agreement (tartışmalı): {(~high_agreement_positive & ~high_agreement_negative).sum()} örnek")

# =================================================================================
# 10. ÖZET RAPOR
# =================================================================================
print("\n[10] Özet değerlendirme raporu oluşturuluyor...")

# Tüm sonuçları topla
evaluation_summary = {
    'threshold_optimization': threshold_results,
    'error_analysis': error_analysis,
    'classification_reports': classification_reports,
    'model_agreement': {
        'high_positive_agreement': int(high_agreement_positive.sum()),
        'high_negative_agreement': int(high_agreement_negative.sum()),
        'low_agreement': int((~high_agreement_positive & ~high_agreement_negative).sum())
    }
}

# JSON olarak kaydet
with open(OUTPUT_DIR / 'evaluation_summary.json', 'w') as f:
    # NumPy int64'ları Python int'e çevir
    json.dump(evaluation_summary, f, indent=4, default=int)
print(f"  ✓ evaluation_summary.json kaydedildi")

# =================================================================================
# 11. MARKDOWN RAPOR
# =================================================================================
print("\n[11] Markdown rapor oluşturuluyor...")

# En düşük FNR'ye sahip model (klinik için en önemli)
best_model_fnr = min(error_analysis.items(), key=lambda x: x[1]['false_negative_rate'])

markdown_content = f"""# DETAYLI MODEL DEĞERLENDİRME RAPORU

## 1. Değerlendirme Genel Bakış

Bu rapor, eğitilen modellerin kapsamlı performans değerlendirmesini içermektedir:
- Precision-Recall analizi
- Threshold optimizasyonu
- Model kalibrasyonu
- Feature importance
- Error analizi
- Model agreement analizi

## 2. Precision-Recall Analizi

Precision-Recall eğrisi, özellikle **imbalanced datasets** için ROC eğrisinden daha informatif olabilir.

![Precision-Recall Curves](model_evaluation/01_precision_recall_curves.png)

**Average Precision (AP)** değerleri:

| Model | Average Precision |
|-------|-------------------|
"""

for model_name, preds in predictions.items():
    ap = average_precision_score(y_test, preds['y_proba'])
    markdown_content += f"| {model_name} | {ap:.4f} |\n"

markdown_content += f"""
**Yorumlama:**
- AP, precision-recall curve altındaki alanı temsil eder
- Yüksek AP, tüm threshold değerlerinde iyi precision ve recall dengesi demektir
- Imbalanced data için ROC-AUC'den daha güvenilir bir metrik

## 3. Threshold Optimizasyonu

Default threshold (0.5) her zaman optimal değildir. Her model için optimal threshold değerleri bulunmuştur.

![Threshold Optimization](model_evaluation/02_threshold_optimization.png)

### Optimal Threshold Değerleri

| Model | F1-Optimal Threshold | F1-Score | J-Optimal Threshold | J-Score |
|-------|---------------------|----------|---------------------|---------|
"""

for model_name, results in threshold_results.items():
    markdown_content += f"| {model_name} | {results['optimal_f1_threshold']:.3f} | "
    markdown_content += f"{results['optimal_f1_score']:.4f} | {results['optimal_j_threshold']:.3f} | "
    markdown_content += f"{results['optimal_j_score']:.4f} |\n"

markdown_content += """
**Threshold Seçim Stratejileri:**

1. **F1-Optimal**: Precision ve recall'ın harmonik ortalamasını maksimize eder
   - Dengeli performans istiyorsak
   
2. **Youden's J**: Sensitivity + Specificity - 1'i maksimize eder
   - ROC curve'den türetilir
   - Genel discriminative power için

3. **Klinik Threshold**: Domain knowledge'a dayalı
   - **High Recall için**: Düşük threshold (örn: 0.3)
     * Daha fazla pozitif tahmin → Daha az missed case
     * False alarm artabilir ama critical cases'i kaçırmayız
   
   - **High Precision için**: Yüksek threshold (örn: 0.7)
     * Sadece emin olduğumuz pozitif tahminler
     * Bazı cases kaçabilir ama tahminler güvenilir

**Önerimiz:** Bu problemde **False Negative minimize edilmeli** (30 gün içinde readmit olacak
hastaları kaçırmamak kritik). Bu yüzden threshold'u düşürüp **recall'ı artırmalıyız**.

## 4. Model Kalibrasyonu

Model kalibrasyonu, tahmin edilen olasılıkların ne kadar güvenilir olduğunu gösterir.

![Calibration Curves](model_evaluation/03_calibration_curves.png)

**İyi Kalibre Edilmiş Model:**
- Model "70% readmission olasılığı" diyorsa, gerçekten de vakaların %70'i readmit olmalı
- Calibration curve, diagonal line'a yakın olmalı

**Zayıf Kalibrasyon:**
- **Over-confident**: Curve diagonal'ın altında → Model olasılıkları olduğundan yüksek tahmin ediyor
- **Under-confident**: Curve diagonal'ın üstünde → Model olasılıkları olduğundan düşük tahmin ediyor

**Kullanım Alanları:**
- Risk stratification: Hastaları yüksek/orta/düşük risk gruplarına ayırmak
- Resource allocation: Limited resources'ları en yüksek risk hastalarına tahsis etmek

## 5. Feature Importance Analizi

### Tree-Based Models

![Feature Importance](model_evaluation/04_feature_importance.png)

**Yorumlama:**
- Yüksek importance → Model bu özellikleri sıkça kullanıyor ve ayırıcı buluyor
- Düşük importance → Model bu özellikleri çok kullanmıyor veya bilgi içermiyor

### Logistic Regression Coefficients

![Logistic Coefficients](model_evaluation/05_logistic_coefficients.png)

**Yorumlama:**
- **Pozitif coefficient**: Özellik artarsa readmission olasılığı artar
- **Negatif coefficient**: Özellik artarsa readmission olasılığı azalır
- **Büyük mutlak değer**: Güçlü etkisi var

## 6. Error Analizi

Error türlerinin detaylı analizi yapılmıştır.

![Error Analysis](model_evaluation/06_error_analysis.png)

### Hata Türleri ve Klinik Etkileri

**False Positive (FP):**
- Model "readmission olacak" dedi ama olmadı
- **Klinik Etki**: Gereksiz müdahale, ekstra takip, resource waste
- **Maliyet**: Orta (gereksiz işlemler)

**False Negative (FN):**
- Model "readmission olmaz" dedi ama oldu
- **Klinik Etki**: Readmission önlenemedi, hasta komplikasyon yaşadı
- **Maliyet**: Yüksek (hasta sağlığı + readmission maliyeti)

### Model Error Karşılaştırması

| Model | False Positive | False Negative | FP Rate | FN Rate |
|-------|----------------|----------------|---------|---------|
"""

for model_name, errors in error_analysis.items():
    markdown_content += f"| {model_name} | {errors['false_positive']} | {errors['false_negative']} | "
    markdown_content += f"{errors['false_positive_rate']:.4f} | {errors['false_negative_rate']:.4f} |\n"

markdown_content += f"""
**En Düşük False Negative Rate**: {best_model_fnr[0]} ({best_model_fnr[1]['false_negative_rate']:.4f})

**Öneri**: Klinik uygulamada **{best_model_fnr[0]}** modeli kullanılabilir çünkü en az 
critical case'i kaçırıyor (lowest FNR).

## 7. Model Agreement Analizi

Farklı modellerin tahminleri ne kadar uyumlu?

![Model Agreement](model_evaluation/07_model_agreement.png)

### Agreement İstatistikleri

- **Yüksek Pozitif Agreement**: {evaluation_summary['model_agreement']['high_positive_agreement']} örnek
  * Modellerin %80+'ı "readmission olacak" dedi
  * Bu örnekler yüksek risk → Priority intervention

- **Yüksek Negatif Agreement**: {evaluation_summary['model_agreement']['high_negative_agreement']} örnek
  * Modellerin %80+'ı "readmission olmaz" dedi
  * Bu örnekler düşük risk → Standard follow-up

- **Düşük Agreement (Tartışmalı)**: {evaluation_summary['model_agreement']['low_agreement']} örnek
  * Modeller anlaşamıyor
  * Bu örnekler belirsiz → Extra attention veya clinical judgment

**Ensemble Önerisi:**
- Yüksek agreement örneklerde confidence yüksek
- Düşük agreement örneklerde **voting** veya **stacking** ile consensus bulunabilir

## 8. Detaylı Classification Reports

Her model için detaylı classification report oluşturulmuştur:

"""

for model_name in predictions.keys():
    filename = model_name.replace(' ', '_').lower() + '_classification_report.txt'
    markdown_content += f"- [{model_name}](model_evaluation/{filename})\n"

markdown_content += """
## 9. Sonuç ve Öneriler

### 9.1 Model Seçimi

**En İyi Genel Performans**: ROC-AUC ve F1 metriklerine göre değerlendirilmeli

**Klinik Kullanım İçin**: En düşük False Negative Rate'e sahip model:
"""
markdown_content += f"- **{best_model_fnr[0]}** (FNR: {best_model_fnr[1]['false_negative_rate']:.4f})\n"

markdown_content += """
### 9.2 Threshold Ayarı

Production deployment için:
- **Aggressive screening**: Threshold = 0.3 (High recall, catch all possible cases)
- **Balanced approach**: Optimal F1 threshold kullan
- **Conservative**: Threshold = 0.7 (High precision, less false alarms)

**Önerimiz**: Başlangıç için **0.3-0.4 arası threshold** kullanılmalı. Bu:
- False negatives minimize eder
- Critical cases kaçırılmaz
- Resource planlaması yapılabilir

### 9.3 Model Deployment Stratejisi

**Option 1: Single Best Model**
- En düşük FNR'ye sahip model deploy edilir
- Basit, hızlı
- Tek model'e bağımlı

**Option 2: Ensemble (Voting)**
- Birden fazla modelin consensus'ü
- Daha robust, güvenilir
- Biraz daha karmaşık

**Option 3: Staged Approach**
- İlk screening: High recall model (catch everything)
- Second stage: High precision model (refine predictions)
- İki aşamalı değerlendirme

### 9.4 Klinik Implementasyon

1. **Risk Stratification**
   - Very High Risk (>0.7 probability): Immediate intervention
   - High Risk (0.5-0.7): Enhanced monitoring
   - Moderate Risk (0.3-0.5): Standard monitoring  
   - Low Risk (<0.3): Routine follow-up

2. **Intervention Programs**
   - Discharge planning improvement
   - Patient education enhancement
   - Medication reconciliation
   - Post-discharge follow-up calls

3. **Monitoring and Updating**
   - Model performance monthly review
   - Retrain with new data quarterly
   - Feature drift monitoring
   - Clinical feedback integration

### 9.5 Sınırlamalar

1. **Data Limitations**: Veri seti 1999-2008, güncel patterns farklı olabilir
2. **Missing Variables**: Sosyoekonomik faktörler, yaşam tarzı bilgileri eksik
3. **External Validity**: Farklı hastane sistemlerinde test edilmeli
4. **Temporal Trends**: Tedavi protokolleri değişiyor, model güncellemeli

---
*Rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(OUTPUT_DIR / 'evaluation_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"  ✓ Markdown rapor kaydedildi: evaluation_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("DETAYLI MODEL DEĞERLENDİRME TAMAMLANDI")
print("=" * 80)
print(f"\n✓ {len(models)} model kapsamlı olarak değerlendirildi")
print(f"✓ En düşük False Negative Rate: {best_model_fnr[0]} ({best_model_fnr[1]['false_negative_rate']:.4f})")
print(f"✓ Tüm raporlar '{OUTPUT_DIR}' dizinine kaydedildi")
print("\nSonraki Adım: 05_hyperparameter_tuning.py scriptini çalıştırın")
print("=" * 80)
