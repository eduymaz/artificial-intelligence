# MODEL EĞİTİMİ VE KARŞILAŞTIRMA RAPORU

## 1. Genel Bilgiler

### 1.1 Veri Seti
- **Toplam Örnek**: 101,766
- **Train Set**: 81,412 (%80.0)
- **Test Set**: 20,354 (%20.0)
- **Özellik Sayısı**: 44

### 1.2 Class Distribution

**Original Train Set:**
- Class 0 (No Readmission <30): 72,326 (%88.84)
- Class 1 (Readmission <30): 9,086 (%11.16)
- **Imbalance Ratio**: 7.96:1

**Resampled Train Set (SMOTE + Undersampling):**
- Class 0: 45,203 (%55.56)
- Class 1: 36,163 (%44.44)
- **Yeni Imbalance Ratio**: 1.25:1

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

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| Logistic Regression | 0.6373 | 0.1676 | 0.5672 | 0.2587 | 0.6530 | 2.73s |
| Random Forest | 0.7989 | 0.2076 | 0.2849 | 0.2402 | 0.6507 | 5.85s |
| XGBoost | 0.8862 | 0.4096 | 0.0449 | 0.0810 | 0.6674 | 2.72s |
| LightGBM | 0.8851 | 0.4086 | 0.0669 | 0.1150 | 0.6743 | 6.47s |
| SVM | 0.6693 | 0.1660 | 0.4883 | 0.2478 | 0.6246 | 6896.93s |

![Model Comparison](model_training/02_model_comparison_metrics.png)

### 3.2 Cross-Validation Sonuçları

**5-Fold Stratified Cross-Validation** kullanılmıştır.

| Model | CV Accuracy | CV ROC-AUC |
|-------|-------------|------------|
| Logistic Regression | 0.6120 ± 0.0042 | 0.6518 ± 0.0062 |
| Random Forest | 0.8122 ± 0.0059 | 0.8837 ± 0.0036 |
| XGBoost | 0.8834 ± 0.0016 | 0.9149 ± 0.0009 |
| LightGBM | 0.8855 ± 0.0013 | 0.9159 ± 0.0014 |
| SVM | 0.7056 ± 0.0071 | 0.7800 ± 0.0054 |

## 4. ROC Curve Analizi

ROC (Receiver Operating Characteristic) eğrisi, farklı threshold değerlerinde modelin 
true positive rate (sensitivity) ve false positive rate (1 - specificity) ilişkisini gösterir.

![ROC Curves](model_training/03_roc_curves.png)

**En İyi ROC-AUC**: LightGBM - 0.6743

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

**ROC-AUC'ye göre en iyi**: LightGBM (0.6743)

**F1-Score'a göre en iyi**: Logistic Regression (0.2587)

### 6.2 Model Seçim Kriterleri

Bu problemde model seçerken şu faktörler değerlendirilmelidir:

1. **Recall (Sensitivity)**: Yüksek recall, 30 gün içinde readmission olacak hastaları 
   daha iyi tespit ettiğimiz anlamına gelir. Klinik açıdan çok önemli.

2. **ROC-AUC**: Genel discriminative power göstergesi. Model ne kadar iyi ayrım yapıyor?

3. **F1-Score**: Precision ve recall'un harmonik ortalaması. Dengeli bir metrik.

4. **Training Time**: Production ortamında model güncellemesi için önemli.

### 6.3 Detaylı Model Analizleri


#### Logistic Regression

**Test Metrikleri:**
- Accuracy: 0.6373
- Precision: 0.1676
- Recall: 0.5672
- F1-Score: 0.2587
- ROC-AUC: 0.6530

**Cross-Validation:**
- CV Accuracy: 0.6120 ± 0.0042
- CV ROC-AUC: 0.6518 ± 0.0062

**Training Time**: 2.73 saniye


#### Random Forest

**Test Metrikleri:**
- Accuracy: 0.7989
- Precision: 0.2076
- Recall: 0.2849
- F1-Score: 0.2402
- ROC-AUC: 0.6507

**Cross-Validation:**
- CV Accuracy: 0.8122 ± 0.0059
- CV ROC-AUC: 0.8837 ± 0.0036

**Training Time**: 5.85 saniye


#### XGBoost

**Test Metrikleri:**
- Accuracy: 0.8862
- Precision: 0.4096
- Recall: 0.0449
- F1-Score: 0.0810
- ROC-AUC: 0.6674

**Cross-Validation:**
- CV Accuracy: 0.8834 ± 0.0016
- CV ROC-AUC: 0.9149 ± 0.0009

**Training Time**: 2.72 saniye


#### LightGBM

**Test Metrikleri:**
- Accuracy: 0.8851
- Precision: 0.4086
- Recall: 0.0669
- F1-Score: 0.1150
- ROC-AUC: 0.6743

**Cross-Validation:**
- CV Accuracy: 0.8855 ± 0.0013
- CV ROC-AUC: 0.9159 ± 0.0014

**Training Time**: 6.47 saniye


#### SVM

**Test Metrikleri:**
- Accuracy: 0.6693
- Precision: 0.1660
- Recall: 0.4883
- F1-Score: 0.2478
- ROC-AUC: 0.6246

**Cross-Validation:**
- CV Accuracy: 0.7056 ± 0.0071
- CV ROC-AUC: 0.7800 ± 0.0054

**Training Time**: 6896.93 saniye


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

Original veri setinde **8.0:1** oranında dengesizlik vardı. 
Class imbalance handling olmadan:
- Model majority class'ı tahmin etmeye bias olur
- Minority class (readmission <30) düşük recall ile tespit edilir
- Klinik açıdan kritik olan vakaları kaçırırız

## 8. Sonuç ve Öneriler

### 8.1 Önemli Bulgular

1. ✓ **5 farklı model** başarıyla eğitildi ve karşılaştırıldı
2. ✓ **SMOTE + Undersampling** ile class imbalance başarıyla yönetildi
3. ✓ En iyi ROC-AUC: **0.6743** (LightGBM)
4. ✓ En iyi F1-Score: **0.2587** (Logistic Regression)

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
*Rapor otomatik olarak oluşturulmuştur - 2025-12-13 17:56:44*
