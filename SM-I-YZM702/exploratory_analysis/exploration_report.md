# VERİ KEŞFİ VE İLK ANALİZ RAPORU

## 1. Veri Seti Genel Bilgileri

- **Toplam Örnek Sayısı**: 101,766
- **Toplam Özellik Sayısı**: 50
- **Bellek Kullanımı**: 192.87 MB
- **Duplike Satır Sayısı**: 0

## 2. Hedef Değişken (readmitted) Dağılımı

| Readmission | Hasta Sayısı | Yüzde |
|-------------|--------------|-------|
| NO | 54,864 | %53.91 |
| >30 | 35,545 | %34.93 |
| <30 | 11,357 | %11.16 |

![Hedef Değişken Dağılımı](exploratory_analysis/01_target_distribution.png)

**Yorum**: Veri seti dengeli bir dağılıma sahip. 
Bu durum model eğitiminde normal eğitim sürecine devam edilebilir.

## 3. Eksik Değerler

- **Eksik Değeri Olan Özellik Sayısı**: 7

### En Çok Eksik Değere Sahip Özellikler

| Özellik | Eksik Sayı | Yüzde |
|---------|------------|-------|
| weight | 98,569 | %96.86 |
| medical_specialty | 49,949 | %49.08 |
| payer_code | 40,256 | %39.56 |
| race | 2,273 | %2.23 |
| diag_3 | 1,423 | %1.40 |
| diag_2 | 358 | %0.35 |
| diag_1 | 21 | %0.02 |

![Eksik Değerler](exploratory_analysis/02_missing_values.png)

**Yorum**: Bazı özellikler çok yüksek oranda eksik değere sahip (>50%). Bu özellikler veri ön işleme aşamasında dikkatli ele alınmalı veya çıkarılmalı.

## 4. Numerik Özellikler

- **Toplam Numerik Özellik Sayısı**: 13

![Numerik Dağılımlar](exploratory_analysis/03_numeric_distributions.png)

![Numerik vs Target](exploratory_analysis/05_numeric_vs_target.png)

## 5. Kategorik Özellikler

- **Toplam Kategorik Özellik Sayısı**: 37

![Kategorik Dağılımlar](exploratory_analysis/04_categorical_distributions.png)

![Kategorik vs Target](exploratory_analysis/06_categorical_vs_target.png)

## 6. Korelasyon Analizi

![Korelasyon Matrisi](exploratory_analysis/07_correlation_matrix.png)

**Yüksek Korelasyonlu Özellik Çiftleri (|r| > 0.7)**: 0

## 7. İlaç Kullanım Analizi

![İlaç Kullanımı](exploratory_analysis/08_medication_usage.png)

### En Çok Kullanılan İlaçlar

| İlaç | Kullanım Yüzdesi |
|------|------------------|
| insulin | %53.44 |
| metformin | %19.64 |
| glipizide | %12.47 |
| glyburide | %10.47 |
| pioglitazone | %7.20 |
| rosiglitazone | %6.25 |
| glimepiride | %5.10 |
| repaglinide | %1.51 |
| nateglinide | %0.69 |
| acarbose | %0.30 |

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
