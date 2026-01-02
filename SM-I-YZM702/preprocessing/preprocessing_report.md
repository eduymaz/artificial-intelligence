# VERİ ÖN İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ RAPORU

## 1. Veri Temizleme

### 1.1 Duplike Satırlar
- **Çıkarılan Duplike Satır**: 0
- **Kalan Satır Sayısı**: 101,766

### 1.2 Gereksiz Sütunlar
Aşağıdaki sütunlar analiz için uygun olmadığından çıkarıldı:
- `encounter_id`
- `patient_nbr`
- `weight`
- `payer_code`
- `medical_specialty`
- `readmitted`

**Toplam Çıkarılan Sütun**: 6

## 2. Eksik Değer Yönetimi

### 2.1 Strateji
- **%50'den fazla eksik değere sahip sütunlar**: Veri setinden çıkarıldı
- **Kategorik değişkenler**: Mode (en sık değer) ile dolduruldu
- **Numerik değişkenler**: Median ile dolduruldu

### 2.2 Yüksek Eksik Değere Sahip Çıkarılan Sütunlar
- `max_glu_serum`
- `A1Cresult`

**Kalan Toplam Eksik Değer**: 0

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
- **Sınıf 0**: 90,409 (%88.84)
- **Sınıf 1**: 11,357 (%11.16)

**Class Imbalance Durumu**: Var - Model eğitiminde class_weight veya SMOTE kullanılacak

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

**Outlier İçeren Özellik Sayısı**: 29

### En Çok Outlier İçeren Özellikler

| Özellik | Outlier Sayısı | Yüzde |
|---------|----------------|-------|
| on_diabetes_med | 23403 | %23.00 |
| metformin | 19988 | %19.64 |
| number_outpatient | 16739 | %16.45 |
| glipizide | 12686 | %12.47 |
| number_emergency | 11383 | %11.19 |
| glyburide | 10650 | %10.47 |
| pioglitazone | 7328 | %7.20 |
| rosiglitazone | 6365 | %6.25 |
| glimepiride | 5191 | %5.10 |
| procedure_intensity | 2808 | %2.76 |

**Karar**: Outlier'lar bırakıldı. Medikal verilerde extreme değerler klinik olarak anlamlı olabilir ve 
model için önemli bilgi taşıyabilir. Robust scaler kullanımı ile etkileri azaltılacak.

## 7. Feature Scaling

**Metod**: StandardScaler (Z-score normalization)

$$
z = \frac{x - \mu}{\sigma}
$$

- Tüm özellikler ortalama=0, standart sapma=1 olacak şekilde ölçeklendirildi
- Model performansını artırmak ve gradient descent optimizasyonunu hızlandırmak için gerekli

![Ölçeklendirilmiş Özellikler](preprocessing/03_scaled_features_distribution.png)

## 8. Final Veri Seti

### 8.1 Boyutlar
- **Örnek Sayısı**: 101,766
- **Özellik Sayısı**: 44
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
*Rapor otomatik olarak oluşturulmuştur - 2025-12-13 15:58:31*
