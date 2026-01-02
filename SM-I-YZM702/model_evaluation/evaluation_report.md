# DETAYLI MODEL DEĞERLENDİRME RAPORU

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
| Logistic Regression | 0.1981 |
| Random Forest | 0.1889 |
| Xgboost | 0.2178 |
| Lightgbm | 0.2235 |
| Svm | 0.1699 |

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
| Logistic Regression | 0.500 | 0.2587 | 0.455 | 0.2250 |
| Random Forest | 0.450 | 0.2636 | 0.397 | 0.2230 |
| Xgboost | 0.200 | 0.2715 | 0.209 | 0.2504 |
| Lightgbm | 0.300 | 0.2788 | 0.240 | 0.2527 |
| Svm | 0.350 | 0.2488 | 0.313 | 0.1994 |

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
| Logistic Regression | 6399 | 983 | 0.3539 | 0.4328 |
| Random Forest | 2469 | 1624 | 0.1365 | 0.7151 |
| Xgboost | 147 | 2169 | 0.0081 | 0.9551 |
| Lightgbm | 220 | 2119 | 0.0122 | 0.9331 |
| Svm | 5570 | 1162 | 0.3080 | 0.5117 |

**En Düşük False Negative Rate**: Logistic Regression (0.4328)

**Öneri**: Klinik uygulamada **Logistic Regression** modeli kullanılabilir çünkü en az 
critical case'i kaçırıyor (lowest FNR).

## 7. Model Agreement Analizi

Farklı modellerin tahminleri ne kadar uyumlu?

![Model Agreement](model_evaluation/07_model_agreement.png)

### Agreement İstatistikleri

- **Yüksek Pozitif Agreement**: 336 örnek
  * Modellerin %80+'ı "readmission olacak" dedi
  * Bu örnekler yüksek risk → Priority intervention

- **Yüksek Negatif Agreement**: 14699 örnek
  * Modellerin %80+'ı "readmission olmaz" dedi
  * Bu örnekler düşük risk → Standard follow-up

- **Düşük Agreement (Tartışmalı)**: 5319 örnek
  * Modeller anlaşamıyor
  * Bu örnekler belirsiz → Extra attention veya clinical judgment

**Ensemble Önerisi:**
- Yüksek agreement örneklerde confidence yüksek
- Düşük agreement örneklerde **voting** veya **stacking** ile consensus bulunabilir

## 8. Detaylı Classification Reports

Her model için detaylı classification report oluşturulmuştur:

- [Logistic Regression](model_evaluation/logistic_regression_classification_report.txt)
- [Random Forest](model_evaluation/random_forest_classification_report.txt)
- [Xgboost](model_evaluation/xgboost_classification_report.txt)
- [Lightgbm](model_evaluation/lightgbm_classification_report.txt)
- [Svm](model_evaluation/svm_classification_report.txt)

## 9. Sonuç ve Öneriler

### 9.1 Model Seçimi

**En İyi Genel Performans**: ROC-AUC ve F1 metriklerine göre değerlendirilmeli

**Klinik Kullanım İçin**: En düşük False Negative Rate'e sahip model:
- **Logistic Regression** (FNR: 0.4328)

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
