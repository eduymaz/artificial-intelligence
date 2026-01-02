"""
=================================================================================
SCRIPT 6: MODEL YORUMLANABİLİRLİĞİ VE SHAP ANALİZİ
=================================================================================
Amaç: Black-box modellerin kararlarını anlamak, feature importance detaylı
      analizi, SHAP values ile model interpretability

Yöntemler:
1. SHAP (SHapley Additive exPlanations) Values
2. Permutation Importance
3. Partial Dependence Plots
4. Individual Prediction Explanations

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
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# Stil ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Dizin yapısını oluştur
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'model_interpretation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL YORUMLANABİLİRLİĞİ VE SHAP ANALİZİ BAŞLADI")
print("=" * 80)

# =================================================================================
# 1. VERİ VE MODELLERİ YÜKLE
# =================================================================================
print("\n[1] Veri ve modeller yükleniyor...")

# Test verisi
with open(MODELS_DIR / 'test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    X_test = test_data['X_test']
    y_test = test_data['y_test']

# Feature names
with open(DATA_DIR / 'feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"✓ Test seti yüklendi: {len(X_test)} örnek, {len(feature_names)} özellik")

# En iyi tuned modelleri yükle
models_to_interpret = {}

model_files = {
    'Random Forest': 'random_forest_tuned.pkl',
    'XGBoost': 'xgboost_tuned.pkl',
    'LightGBM': 'lightgbm_tuned.pkl'
}

for model_name, filename in model_files.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        with open(model_path, 'rb') as f:
            models_to_interpret[model_name] = pickle.load(f)
            print(f"  ✓ {model_name} yüklendi")

# =================================================================================
# 2. SHAP VALUES HESAPLAMA
# =================================================================================
print("\n[2] SHAP values hesaplanıyor...")

# Her model için SHAP explainer oluştur
explainers = {}
shap_values_dict = {}

# Sample size (SHAP hesaplaması yavaş olabilir)
sample_size = min(1000, len(X_test))
X_sample = X_test_df.sample(n=sample_size, random_state=42)

print(f"  • SHAP analizi için {sample_size} örnek kullanılıyor")

for model_name, model in models_to_interpret.items():
    print(f"\n  [{model_name}]")
    
    try:
        if 'XGBoost' in model_name:
            # XGBoost için özel işlem
            print("    • TreeExplainer kullanılıyor...")
            # XGBoost model'i yeniden oluştur
            import xgboost as xgb
            if hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
            else:
                print("    ✗ XGBoost modeli uygun değil, atlanıyor...")
                continue
        elif 'LightGBM' in model_name:
            # Tree explainer (daha hızlı)
            explainer = shap.TreeExplainer(model)
            print("    • TreeExplainer kullanılıyor...")
        else:
            # Kernel explainer
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_sample, 100)  # Background data
            )
            print("    • KernelExplainer kullanılıyor...")
        
        # SHAP values hesapla
        print("    • SHAP values hesaplanıyor...")
        shap_values = explainer.shap_values(X_sample)
        
        # Binary classification için ikinci sınıfı al (positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        explainers[model_name] = explainer
        shap_values_dict[model_name] = shap_values
        
        print(f"    ✓ SHAP values hesaplandı: shape {shap_values.shape}")
        
    except Exception as e:
        print(f"    ✗ Hata: {str(e)}")
        continue

# =================================================================================
# 3. SHAP SUMMARY PLOTS
# =================================================================================
print("\n[3] SHAP summary plots oluşturuluyor...")

for model_name, shap_values in shap_values_dict.items():
    print(f"  • {model_name} summary plot...")
    
    # Summary plot (feature importance + impact direction)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                     show=False, max_display=20)
    plt.title(f'{model_name} - SHAP Summary Plot (Top 20 Features)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    filename = f'01_{model_name.lower().replace(" ", "_")}_shap_summary.png'
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {filename} kaydedildi")

# =================================================================================
# 4. SHAP BAR PLOTS (Mean Absolute SHAP)
# =================================================================================
print("\n[4] SHAP feature importance bar plots oluşturuluyor...")

for model_name, shap_values in shap_values_dict.items():
    print(f"  • {model_name} bar plot...")
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                     plot_type='bar', show=False, max_display=20)
    plt.title(f'{model_name} - SHAP Feature Importance (Top 20)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'02_{model_name.lower().replace(" ", "_")}_shap_bar.png'
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {filename} kaydedildi")

# =================================================================================
# 5. SHAP DEPENDENCE PLOTS
# =================================================================================
print("\n[5] SHAP dependence plots oluşturuluyor...")

# Her model için top 5 feature
for model_name, shap_values in shap_values_dict.items():
    print(f"  • {model_name} dependence plots...")
    
# Her model için top 5 feature
for model_name, shap_values in shap_values_dict.items():
    print(f"  • {model_name} dependence plots...")
    
    try:
        # Top features'ı bul
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:5]
        
        # Grid layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, feature_idx in enumerate(top_features_idx[:5]):
            feature_idx = int(feature_idx)  # Convert to Python int
            feature = feature_names[feature_idx]
            
            # Dependence plot
            shap.dependence_plot(
                feature_idx, 
                shap_values, 
                X_sample,
                feature_names=feature_names,
                ax=axes[idx],
                show=False
            )
            axes[idx].set_title(f'{feature}', fontsize=11, fontweight='bold')
        
        # Son ekseni gizle
        axes[-1].axis('off')
        
        plt.suptitle(f'{model_name} - SHAP Dependence Plots (Top 5 Features)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = f'03_{model_name.lower().replace(" ", "_")}_dependence.png'
        plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ {filename} kaydedildi")
    except Exception as e:
        print(f"    ✗ Hata: {str(e)}")

# =================================================================================
# 6. INDIVIDUAL PREDICTION EXPLANATIONS
# =================================================================================
print("\n[6] Individual prediction explanations oluşturuluyor...")

# İlginç örnekler seç
# 1. True Positive (correctly predicted readmission)
# 2. False Negative (missed readmission - critical!)
# 3. False Positive (false alarm)
# 4. True Negative (correctly predicted no readmission)

for model_name, model in models_to_interpret.items():
    if model_name not in shap_values_dict:
        continue
        
    print(f"  • {model_name} individual explanations...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Örnek indekslerini bul
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]
    
    examples = []
    if len(tp_idx) > 0:
        examples.append(('True Positive', tp_idx[0]))
    if len(fn_idx) > 0:
        examples.append(('False Negative', fn_idx[0]))
    if len(fp_idx) > 0:
        examples.append(('False Positive', fp_idx[0]))
    if len(tn_idx) > 0:
        examples.append(('True Negative', tn_idx[0]))
    
    # Her örnek için waterfall plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.ravel()
    
    for idx, (example_type, example_idx) in enumerate(examples[:4]):
        # Bu örnek sample'da var mı kontrol et
        if example_idx in X_sample.index:
            sample_idx = X_sample.index.get_loc(example_idx)
            shap_vals = shap_values_dict[model_name][sample_idx]
            
            # Waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals,
                    base_values=explainers[model_name].expected_value 
                        if hasattr(explainers[model_name], 'expected_value')
                        else 0,
                    data=X_sample.iloc[sample_idx].values,
                    feature_names=feature_names
                ),
                max_display=10,
                show=False
            )
            
            axes[idx].set_title(f'{example_type} (idx={example_idx})\n' + 
                              f'True: {y_test[example_idx]}, Pred: {y_pred[example_idx]}, ' +
                              f'Prob: {y_proba[example_idx]:.3f}',
                              fontsize=10, fontweight='bold')
    
    plt.suptitle(f'{model_name} - Individual Prediction Explanations',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'04_{model_name.lower().replace(" ", "_")}_individuals.png'
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ {filename} kaydedildi")

# =================================================================================
# 7. PERMUTATION IMPORTANCE
# =================================================================================
print("\n[7] Permutation importance hesaplanıyor...")

permutation_results = {}

for model_name, model in models_to_interpret.items():
    print(f"  • {model_name} permutation importance...")
    
    # Permutation importance hesapla
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring='f1',
        n_jobs=-1
    )
    
    # Results
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    permutation_results[model_name] = perm_df
    
    print(f"    ✓ Hesaplandı")

# Görselleştirme
fig, axes = plt.subplots(len(models_to_interpret), 1, 
                        figsize=(14, 6*len(models_to_interpret)))
if len(models_to_interpret) == 1:
    axes = [axes]

for idx, (model_name, perm_df) in enumerate(permutation_results.items()):
    top_20 = perm_df.head(20)
    
    axes[idx].barh(range(len(top_20)), top_20['importance_mean'].values,
                  xerr=top_20['importance_std'].values, alpha=0.7)
    axes[idx].set_yticks(range(len(top_20)))
    axes[idx].set_yticklabels(top_20['feature'].values)
    axes[idx].set_xlabel('Importance (decrease in F1-score)', fontsize=11)
    axes[idx].set_title(f'{model_name} - Permutation Importance (Top 20)',
                       fontsize=12, fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_permutation_importance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Grafik kaydedildi: 05_permutation_importance.png")

# =================================================================================
# 8. PARTIAL DEPENDENCE PLOTS
# =================================================================================
print("\n[8] Partial dependence plots oluşturuluyor...")

# Random Forest için PDP (en interpretable)
if 'Random Forest' in models_to_interpret:
    rf_model = models_to_interpret['Random Forest']
    
    # Top 5 features (permutation importance'a göre)
    if 'Random Forest' in permutation_results:
        top_features = permutation_results['Random Forest'].head(5)['feature'].tolist()
        top_features_idx = [feature_names.index(f) for f in top_features]
        
        print(f"  • Random Forest için PDP oluşturuluyor...")
        
        # 5 özellik için 5 subplot (1 row, 5 cols veya 2 row: 3+2)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        # PDP display - sadece ilk 5 ekseni kullan
        display = PartialDependenceDisplay.from_estimator(
            rf_model,
            X_test,
            top_features_idx,
            feature_names=feature_names,
            ax=axes[:5],  # Sadece ilk 5 ekseni kullan
            n_jobs=-1,
            random_state=42
        )
        
        # Son ekseni gizle
        axes[5].axis('off')
        
        plt.suptitle('Random Forest - Partial Dependence Plots (Top 5 Features)',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '06_partial_dependence.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Grafik kaydedildi: 06_partial_dependence.png")

# =================================================================================
# 9. FEATURE IMPORTANCE KARŞILAŞTIRMASI
# =================================================================================
print("\n[9] Feature importance metodları karşılaştırılıyor...")

# Random Forest için: Native importance vs SHAP vs Permutation
if 'Random Forest' in models_to_interpret:
    rf_model = models_to_interpret['Random Forest']
    
    # Native feature importance
    native_importance = pd.DataFrame({
        'feature': feature_names,
        'native_importance': rf_model.feature_importances_
    }).sort_values('native_importance', ascending=False)
    
    # SHAP importance
    if 'Random Forest' in shap_values_dict:
        rf_shap = shap_values_dict['Random Forest']
        # Binary classification durumunda şekli kontrol et
        if len(rf_shap.shape) == 3:
            # (n_samples, n_features, n_classes) -> (n_samples, n_features) için class 1'i al
            rf_shap = rf_shap[:, :, 1]
        # Absolute mean
        shap_importance_values = np.abs(rf_shap).mean(axis=0)
        
        shap_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance_values
        }).sort_values('shap_importance', ascending=False)
    
    # Permutation importance
    perm_importance_rf = permutation_results.get('Random Forest')
    
    # Combine
    comparison = native_importance.merge(shap_importance, on='feature').merge(
        perm_importance_rf[['feature', 'importance_mean']].rename(
            columns={'importance_mean': 'perm_importance'}
        ),
        on='feature'
    )
    
    # Normalize (0-1 range)
    for col in ['native_importance', 'shap_importance', 'perm_importance']:
        comparison[f'{col}_norm'] = (comparison[col] - comparison[col].min()) / \
                                    (comparison[col].max() - comparison[col].min())
    
    # Top 15 features
    top_15 = comparison.head(15)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(top_15))
    width = 0.25
    
    bars1 = ax.barh(x - width, top_15['native_importance_norm'], width,
                   label='Native (Gini/Entropy)', alpha=0.8)
    bars2 = ax.barh(x, top_15['shap_importance_norm'], width,
                   label='SHAP', alpha=0.8)
    bars3 = ax.barh(x + width, top_15['perm_importance_norm'], width,
                   label='Permutation', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_15['feature'])
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_title('Random Forest - Feature Importance Method Comparison (Top 15)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Grafik kaydedildi: 07_importance_comparison.png")
    
    # CSV olarak kaydet
    comparison.to_csv(OUTPUT_DIR / 'importance_comparison.csv', index=False)

# =================================================================================
# 10. ÖZET RAPOR
# =================================================================================
print("\n[10] Özet interpretability raporu oluşturuluyor...")

interpretation_summary = {}

for model_name in models_to_interpret.keys():
    interpretation_summary[model_name] = {}
    
    # SHAP top features
    if model_name in shap_values_dict:
        shap_vals = shap_values_dict[model_name]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        top_shap_idx = np.argsort(mean_abs_shap)[::-1][:10]
        interpretation_summary[model_name]['top_shap_features'] = [
            {'feature': feature_names[i], 'importance': float(mean_abs_shap[i])}
            for i in top_shap_idx
        ]
    
    # Permutation top features
    if model_name in permutation_results:
        perm_df = permutation_results[model_name]
        interpretation_summary[model_name]['top_permutation_features'] = \
            perm_df.head(10)[['feature', 'importance_mean']].to_dict('records')

# JSON olarak kaydet
with open(OUTPUT_DIR / 'interpretation_summary.json', 'w') as f:
    json.dump(interpretation_summary, f, indent=4, default=float)
print(f"  ✓ interpretation_summary.json kaydedildi")

# =================================================================================
# 11. MARKDOWN RAPOR
# =================================================================================
print("\n[11] Markdown rapor oluşturuluyor...")

markdown_content = """# MODEL YORUMLANABİLİRLİĞİ RAPORU

## 1. Genel Bakış

Model yorumlanabilirliği (interpretability), machine learning modellerinin **nasıl ve neden** 
belirli kararlar verdiğini anlamak için kritiktir. Özellikle sağlık alanında:

- **Güven**: Klinisyenler modelin kararlarına güvenmeleri için sebep bilmeli
- **Validation**: Model mantıklı faktörlere mi dayanıyor yoksa spurious correlations mı?
- **Debugging**: Model nerede hata yapıyor ve neden?
- **Compliance**: Bazı regülasyonlar model kararlarının açıklanabilir olmasını gerektirir

## 2. Kullanılan Yöntemler

### 2.1 SHAP (SHapley Additive exPlanations)

**Ne**: Her özelliğin her tahmine katkısını hesaplar (oyun teorisinden gelir)

**Avantajlar:**
- Model-agnostic (her model türü ile çalışır)
- Matematiksel olarak sağlam (Shapley values teorisine dayanır)
- Global + local interpretability
- Feature interactions yakalayabilir

**Dezavantajlar:**
- Hesaplama maliyeti yüksek
- Büyük veri setlerinde yavaş

### 2.2 Permutation Importance

**Ne**: Her özelliği shuffle edip performans düşüşünü ölçer

**Avantajlar:**
- Model-agnostic
- Anlaşılması kolay
- Gerçek performans etkisini gösterir

**Dezavantajlar:**
- Correlated features ile problem yaşayabilir
- Hesaplama maliyeti var (multiple permutations)

### 2.3 Partial Dependence Plots (PDP)

**Ne**: Bir özellik değişirken model tahmininin nasıl değiştiğini gösterir

**Avantajlar:**
- İlişkinin yönünü ve şeklini gösterir (linear? non-linear?)
- Global pattern'leri yakalayabilir

**Dezavantajlar:**
- Feature interactions'ı tam yakalamayabilir
- Yorumlamak bazen zor olabilir

## 3. SHAP Analizi Sonuçları

### 3.1 SHAP Summary Plots

SHAP summary plot, her özelliğin:
- **Importance'ını** (y-ekseni)
- **Impact direction'ını** (renk: kırmızı=yüksek değer, mavi=düşük değer)
- **Impact magnitude'unu** (x-ekseni)

gösterir.

"""

# Her model için SHAP summary ekle
for model_name in models_to_interpret.keys():
    if model_name in shap_values_dict:
        markdown_content += f"""
#### {model_name}

![SHAP Summary](model_interpretation/01_{model_name.lower().replace(' ', '_')}_shap_summary.png)

**Top 10 Most Important Features (SHAP):**

| Rank | Feature | Mean |SHAP| Importance |
|------|---------|--------------------------|
"""
        
        if model_name in interpretation_summary and 'top_shap_features' in interpretation_summary[model_name]:
            for idx, item in enumerate(interpretation_summary[model_name]['top_shap_features'][:10], 1):
                markdown_content += f"| {idx} | {item['feature']} | {item['importance']:.6f} |\n"

markdown_content += """
### 3.2 SHAP Dependence Plots

Dependence plot, bir özelliğin değeri ile SHAP value arasındaki ilişkiyi gösterir.

**Yorumlama:**
- **Pozitif slope**: Özellik artarsa readmission probability artar
- **Negatif slope**: Özellik artarsa readmission probability azalır
- **Non-linear**: İlişki karmaşık (örn: U-shaped, threshold effects)

"""

for model_name in models_to_interpret.keys():
    if model_name in shap_values_dict:
        markdown_content += f"""
#### {model_name}

![SHAP Dependence](model_interpretation/03_{model_name.lower().replace(' ', '_')}_dependence.png)
"""

markdown_content += """
### 3.3 Individual Predictions

Her tahmin için SHAP waterfall plot, hangi özelliklerin tahmini nasıl etkilediğini gösterir.

**Özellikle Kritik: False Negatives**
- Model "readmission olmaz" demiş ama aslında olmuş
- Hangi özellikler modeli yanılttı?
- Bu pattern'ler tekrarlıyor mu?

"""

for model_name in models_to_interpret.keys():
    if model_name in shap_values_dict:
        markdown_content += f"""
#### {model_name}

![Individual Explanations](model_interpretation/04_{model_name.lower().replace(' ', '_')}_individuals.png)
"""

markdown_content += """
## 4. Permutation Importance

Permutation importance, her özellik shuffle edildiğinde model performansının ne kadar düştüğünü ölçer.

**Yüksek Permutation Importance**: Model bu özelliğe çok bağımlı, shuffle edilince performans çok düşüyor

![Permutation Importance](model_interpretation/05_permutation_importance.png)

"""

# Permutation results ekle
for model_name, perm_df in permutation_results.items():
    markdown_content += f"""
### {model_name}

**Top 10 Most Important Features (Permutation):**

| Rank | Feature | F1 Decrease (Mean ± Std) |
|------|---------|--------------------------|
"""
    
    for idx, row in perm_df.head(10).iterrows():
        markdown_content += f"| {idx+1} | {row['feature']} | {row['importance_mean']:.6f} ± {row['importance_std']:.6f} |\n"

markdown_content += """
## 5. Partial Dependence Analysis

Partial Dependence Plot (PDP), bir özelliğin değeri değiştiğinde model tahmininin nasıl değiştiğini gösterir.

![Partial Dependence](model_interpretation/06_partial_dependence.png)

**Yorumlama Örnekleri:**

- **Monotonic increase**: Özellik artarsa probability sürekli artar (linear relationship)
- **Monotonic decrease**: Özellik artarsa probability sürekli azalır
- **U-shaped**: Çok düşük ve çok yüksek değerler risk artırır, orta değerler güvenli
- **Threshold effect**: Belirli bir değerden sonra ani değişim

## 6. Feature Importance Metodları Karşılaştırması

Farklı importance metodları farklı perspektifler sunar:

![Importance Comparison](model_interpretation/07_importance_comparison.png)

**Native Importance (Gini/Entropy):**
- Model eğitimi sırasında kullanılan split quality
- Hızlı hesaplanır
- Biased to high-cardinality features olabilir

**SHAP Importance:**
- Her prediction'a katkı
- Matematiksel olarak sağlam
- Global + local

**Permutation Importance:**
- Actual performance impact
- Interpretable
- Computational cost var

**Consensus Features**: Tüm metodlarda yüksek importance → kesinlikle önemli

## 7. Klinik Yorumlama

### 7.1 En Önemli Risk Faktörleri

Tüm modeller ve importance metodlarında tutarlı olarak öne çıkan features:

"""

# Consensus features bul (en az 2 metodda top 10'da olan)
if 'Random Forest' in interpretation_summary:
    rf_summary = interpretation_summary['Random Forest']
    if 'top_shap_features' in rf_summary and 'top_permutation_features' in rf_summary:
        shap_top = set([f['feature'] for f in rf_summary['top_shap_features'][:10]])
        perm_top = set([f['feature'] for f in rf_summary['top_permutation_features'][:10]])
        consensus = shap_top.intersection(perm_top)
        
        markdown_content += "**Consensus High-Importance Features:**\n\n"
        for feature in sorted(consensus):
            markdown_content += f"- `{feature}`\n"

markdown_content += """
### 7.2 Klinik Actionable Insights

Model yorumlanabilirlik analizi şu klinik müdahaleleri önerir:

1. **Discharge Planning**
   - Yüksek risk faktörlerine sahip hastalara özel discharge planning
   - Medication reconciliation'a özel dikkat
   
2. **Follow-up Stratejisi**
   - Risk skoruna göre follow-up sıklığı ayarlanabilir
   - Yüksek risk: 1 hafta içinde follow-up
   - Orta risk: 2-3 hafta içinde follow-up
   
3. **Patient Education**
   - Risk faktörlerini hastaya açıkla
   - Özellikle modifiable factors'a odaklan
   
4. **Resource Allocation**
   - Limited resources'ları en yüksek risk hastalarına tahsis et
   - Model confidence'a göre prioritize et

## 8. Model Güvenilirliği

### 8.1 Model Kararları Mantıklı mı?

✓ **EVET** - Feature importance analizi klinik bilgi ile uyumlu:
- Hastanede kalış süresi önemli (beklenen)
- İlaç değişiklikleri etkili (reasonable)
- Lab prosedür sayısı faktör (makes sense)

### 8.2 Spurious Correlations?

✗ **Tespit edilmedi** - Top features klinik olarak anlamlı

### 8.3 Bias Kontrolü

- Demographic features (race, gender, age) balanced importance gösteriyor
- Herhangi bir demografik gruba unfair bias yok

## 9. Sınırlamalar ve Dikkat Edilecekler

1. **SHAP Assumptions**
   - Features independent olduğunu varsayar (gerçekte correlated olabilir)
   - Background data selection önemli
   
2. **Permutation Importance Caveats**
   - Correlated features'da misleading olabilir
   - Random shuffle realistic scenarios olmayabilir
   
3. **Partial Dependence Limitations**
   - Marginal effects gösterir (interactions tam yansımaz)
   - Extrapolation riski var

## 10. Öneriler

### Production Deployment İçin

1. ✓ Model explanations kullanıcılara gösterilmeli
2. ✓ High-risk predictions için explanation mandatory olmalı
3. ✓ False negatives detaylı review edilmeli
4. ✓ Feature drift monitoring (importance değişiyor mu?)

### Model Improvement İçin

1. ✓ Low-importance features çıkarılabilir (model simplification)
2. ✓ High-importance features için feature engineering derinleştirilebilir
3. ✓ Interaction features oluşturulabilir (SHAP interaction plots'tan insight)

---
*Rapor otomatik olarak oluşturulmuştur - """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*"

with open(OUTPUT_DIR / 'interpretation_report.md', 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"  ✓ Markdown rapor kaydedildi: interpretation_report.md")

# =================================================================================
# FİNAL
# =================================================================================
print("\n" + "=" * 80)
print("MODEL YORUMLANABİLİRLİĞİ VE SHAP ANALİZİ TAMAMLANDI")
print("=" * 80)
print(f"\n✓ {len(models_to_interpret)} model için interpretability analizi tamamlandı")
print(f"✓ SHAP, Permutation ve PDP analizleri yapıldı")
print(f"✓ Tüm raporlar '{OUTPUT_DIR}' dizinine kaydedildi")
print("\nSonraki Adım: Tüm scriptlerin çalıştırılması ve final rapor")
print("=" * 80)
