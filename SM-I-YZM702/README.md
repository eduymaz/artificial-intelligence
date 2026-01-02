# DİYABETİK HASTALARIN 30 GÜN İÇİNDE HASTANEYE TEKRAR YATIŞ TAHMİNİ

#### 👩🏼‍💻**Elif Duymaz Yılmaz, 6067007**
##### **OCAK 2, 2026**
_**YZM702 - Makine Öğrenmesi Temelleri Final Raporu**_

## 1. GİRİŞ VE PROBLEM TANIMI

### 1.1 Projenin Motivasyonu ve Önemi

Diyabet, günümüzde dünya genelinde yaklaşık 537 milyon yetişkini etkileyen ve küresel sağlık sistemleri üzerinde muazzam bir baskı oluşturan kronik bir metabolik hastalıktır. Hastanede tedavi gören diyabetik hastaların taburcu edildikten sonraki ilk 30 gün içinde tekrar hastaneye yatış yapması, hem hasta sağlığı hem de sağlık sisteminin ekonomik sürdürülebilirliği açısından en kritik problemlerden biri olarak kabul edilmektedir. Klinik açıdan bu erken tekrar yatışlar; yetersiz taburculuk planlaması, hastanın tedaviye uyumsuzluğu, hastalık yönetimindeki sistemik aksaklıklar veya bakım hizmetleri arasındaki koordinasyon eksikliği gibi temel sorunlara işaret etmektedir. Bu bağlamda, **30 gün içindeki tekrar yatış oranları,** sunulan sağlık hizmetinin kalitesini ölçen temel bir gösterge niteliği taşımaktadır.

**Problemin Klinik Önemi:**

Hastaneye tekrar yatış, genellikle aşağıdaki durumları işaret etmektedir:
- Yetersiz veya eksik taburculuk planlaması
- Hasta tarafından tedaviye uyumsuzluk
- Hastalık yönetimindeki sistemik sorunlar
- Koordine olmayan bakım hizmetleri

30 gün içindeki erken tekrar yatışlar, özellikle önlenebilir durumları göstermesi nedeniyle sağlık hizmeti kalitesinin bir göstergesi olarak kabul edilmektedir.

**Problemin Ekonomik Boyutu:**

- Amerika Birleşik Devletleri'nde Medicare gibi sigortacılar, bazı durumlarda 30 günlük tekrar yatışları geri ödememektedir.
- Hastaneler için finansal cezalar ve itibar kaybına neden olmaktadır.
- Ulusal düzeyde yıllık milyarlarca dolarlık ek maliyet oluşturmaktadır.

**Makine Öğrenmesi ile Çözüm Potansiyeli:**

Yüksek riskli hastaların makine öğrenmesi modelleri ile önceden belirlenmesi, klinik ekiplerin:
- Hedefli müdahaleler planlamasına
- Taburculuk süreçlerini optimize etmesine
- Hasta takibini güçlendirmesine
- Kaynakları daha etkin kullanmasına

olanak sağlamaktadır.

### 1.2 Veri Seti Tanıtımı

Bu projede kullanılan veri seti, **UCI Machine Learning Repository**'den temin edilen **["Diabetes 130-US Hospitals for Years 1999-2008"](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)** veri setidir.

**Veri Seti Özellikleri:**

| Özellik | Değer |
|---------|-------|
| **Kaynak** | UCI Machine Learning Repository |
| **Kapsam** | 130 ABD Hastanesi |
| **Zaman Aralığı** | 1999-2008 (10 yıl) |
| **Toplam Kayıt** | 101,766 hasta kaydı |
| **Özellik Sayısı** | 50 değişken |
| **Veri Boyutu** | ~193 MB (bellek) |

*Hedef Değişken (readmitted):**

Veri setindeki hedef değişken `readmitted` üç kategori içermektedir:
- `<30`: Hasta 30 günden önce tekrar hastaneye yatırılmış (Kritik durum)
- `>30`: Hasta 30 günden sonra tekrar hastaneye yatırılmış
- `NO`: Hasta tekrar hastaneye yatırılmamış

**Proje Kapsamında Dönüşüm:**

Bu proje kapsamında problem, **binary (ikili) sınıflandırma** problemi olarak ele alınmıştır:
- **Sınıf 1 (Pozitif):** `<30` - 30 gün içinde tekrar yatış (**yüksek risk**)
- **Sınıf 0 (Negatif):** `>30` veya `NO` - Diğer durumlar (**normal risk**)

Bu dönüşüm, klinik açıdan en kritik olan erken tekrar yatışları tespit etmeye odaklanmaktadır.


**Veri Setinin Kapsamı:**

Veri seti aşağıdaki bilgileri içermektedir:
- **Demografik Bilgiler:** Yaş, cinsiyet, ırk
- **Başvuru Bilgileri:** Kabul türü, taburculuk durumu, sevk kaynağı
- **Klinik Ölçümler:** Laboratuvar testleri sayısı, prosedür sayısı, ilaç sayısı
- **Hastanede Kalış:** Yatış süresi (gün)
- **Teşhis Kodları:** Birincil, ikincil ve üçüncül teşhis kodları (ICD-9)
- **İlaç Bilgileri:** 23 farklı diyabet ilacının kullanımı ve doz değişiklikleri
- **Önceki Başvurular:** Acil servis ziyaretleri, geçmiş hastaneye yatışlar

### 1.3 Projenin Amaçları ve Kapsamı

Bu proje, **YZM702 - Makine Öğrenmesi Temelleri** dersi kapsamında gerçekleştirilmiş olup, aşağıdaki amaçları hedeflemektedir:

**Temel Amaçlar:**

1. **Kapsamlı Veri Analizi:** Diabetik hastane verilerinin detaylı keşfi ve istatistiksel analizinin yapılması

2. **Veri Ön İşleme:** Eksik değer yönetimi, aykırı değer tespiti, kategorik kodlama ve özellik mühendisliği uygulamalarının gerçekleştirilmesi

3. **Çoklu Model Karşılaştırması:** Farklı makine öğrenmesi algoritmalarının (Logistic Regression, Random Forest, SVM, XGBoost, LightGBM) performanslarının karşılaştırılması

4. **Model Optimizasyonu:** Hiperparametre ayarlama teknikleri ile en iyi model konfigürasyonunun bulunması

5. **Model Yorumlanabilirliği:** Özellik önem analizi ile klinik içgörüler elde edilmesi

6. **Pratik Uygulanabilirlik:** Gerçek dünya sağlık sistemlerinde kullanılabilecek bir tahmin modeli geliştirilmesi

### 1.4 Projenin Beklenen Değeri

Bu projenin temel amacı, diyabetik hastaların verilerini derinlemesine analiz ederek anlamlı örüntüler ortaya çıkarmak ve farklı makine öğrenmesi algoritmalarını (Lojistik Regresyon, Random Forest, XGBoost vb.) kıyaslayarak en başarılı tahmin modelini geliştirmektir. Çalışma kapsamında sadece yüksek doğruluk oranlarına ulaşmak değil, aynı zamanda özellik önem analizi (feature importance) yaparak hangi klinik faktörlerin tekrar yatış riskini daha fazla tetiklediğini belirlemek hedeflenmiştir. Elde edilen bulguların, sağlık profesyonellerine kişiselleştirilmiş taburculuk planları hazırlama konusunda rehberlik etmesi ve gereksiz hastane yatışlarını azaltarak operasyonel verimliliğe katkı sağlaması beklenmektedir.

## 2. VERİ KEŞFİ VE İLK ANALİZ

Veri analiz sürecinin ilk aşamasında, veri setinin genel yapısı, özellik dağılımları ve hedef değişkenle olan ilişkileri detaylı olarak incelenmiştir. Bu aşama, veri kalitesini değerlendirmek ve ön işleme stratejilerini belirlemek için kritik önem taşımaktadır.

## 2.1 Veri Yükleme ve Genel İnceleme

**Kullanılan Araçlar ve Kütüphaneler:**
- Python 3.x
- Pandas (veri manipülasyonu)
- NumPy (sayısal hesaplamalar)
- Matplotlib ve Seaborn (görselleştirme)

**Veri Yükleme İşlemi:**

İki ayrı veri dosyası yüklenmiştir:
1. `diabetic_data.csv`: Ana veri seti (101,766 × 50)
2. `IDS_mapping.csv`: ID kodlarının açıklamaları

```python
df = pd.read_csv('data/diabetic_data.csv')
mapping_df = pd.read_csv('data/IDS_mapping.csv')
```

**Veri Seti Genel Özellikleri:**

| Metrik | Değer |
|--------|-------|
| **Toplam Kayıt Sayısı** | 101,766 hasta |
| **Toplam Özellik Sayısı** | 50 değişken |
| **Bellek Kullanımı** | 192.87 MB |
| **Duplike Kayıt** | 0 (Temiz veri) |
| **Numerik Özellik** | 13 değişken |
| **Kategorik Özellik** | 37 değişken |

Bu bulgular, veri setinin duplikasyon içermediğini ve karma veri tiplerine sahip olduğunu göstermektedir.

### 2.2 Hedef Değişken Analizi

Hedef değişken olan `readmitted` üç kategoriye ayrılmış durumdadır. Bu değişkenin dağılımı projenin sınıflandırma stratejisini belirlemede temel rol oynamıştır.

**Orijinal Hedef Değişken Dağılımı:**

| Kategori | Hasta Sayısı | Yüzde |
|----------|--------------|-------|
| **NO** (Tekrar yatış yok) | 54,864 | %53.91 |
| **>30** (30 günden sonra) | 35,545 | %34.93 |
| **<30** (30 gün içinde) | 11,357 | %11.16 |

**Binary Dönüşüm:**

Klinik açıdan en kritik durum olan 30 gün içindeki erken tekrar yatışları tespit etmek amacıyla hedef değişken binary formata dönüştürülmüştür:

- **Sınıf 1 (Pozitif):** `<30` → 11,357 hasta (%11.4)
- **Sınıf 0 (Negatif):** `>30` + `NO` → 90,409 hasta (%88.6)

**Önemli Bulgu:**

Veri seti **dengesiz (imbalanced)** bir dağılım göstermektedir. Pozitif sınıf (30 gün içinde tekrar yatış) oranı sadece %11.4'tür. Bu durum, model eğitimi sırasında sınıf dengeleme tekniklerinin kullanılmasını gerekli kılmıştır.

![Hedef Değişken Dağılımı](docs/exploratory_analysis/01_target_distribution.png)
*Şekil 2.1: Hedef değişkenin orijinal dağılımı (sol) ve yüzdesel oranları (sağ)*

### 2.3 Eksik Değer Analizi

Veri setinde eksik değerler `?` karakteri ile kodlanmıştır. Toplam 7 özellikte eksik değer tespit edilmiştir.

**Eksik Değer İstatistikleri:**

| Özellik | Eksik Sayı | Eksik Yüzde | Karar |
|---------|------------|-------------|-------|
| **weight** | 98,569 | %96.86 | ❌ Çıkarıldı |
| **medical_specialty** | 49,949 | %49.08 | ❌ Çıkarıldı |
| **payer_code** | 40,256 | %39.56 | ❌ Çıkarıldı |
| **race** | 2,273 | %2.23 | ✅ Mode imputation |
| **diag_3** | 1,423 | %1.40 | ✅ "Missing" kategorisi |
| **diag_2** | 358 | %0.35 | ✅ "Missing" kategorisi |
| **diag_1** | 21 | %0.02 | ✅ Mode imputation |


<div style="
    border-left: 5px solid #2196F3;
    background-color: #E3F2FD;
    padding: 10px;
    margin: 10px 0;
">
<strong>ℹ️ Bilgi</strong><br>
    
1. **Yüksek Oranda Eksik Özellikler (>40%):** `weight`, `medical_specialty`, ve `payer_code` özellikleri, eksiklik oranının çok yüksek olması nedeniyle veri setinden çıkarılmıştır.

2. **Düşük Oranda Eksik Özellikler (<5%):** `race` ve `diag_1` için en sık görülen değer (mode) ile doldurma işlemi uygulanmıştır.

3. **Teşhis Kodları:** `diag_2` ve `diag_3` için eksik değerler "Missing" kategorisi olarak işaretlenmiştir çünkü eksikliğin kendisi klinik bir bilgi taşıyabilir.
</div>



![Eksik Değerler Analizi](docs/exploratory_analysis/02_missing_values.png)
*Şekil 2.2: Özelliklerdeki eksik değer yüzdeleri*

### 2.4 Numerik Özellikler Analizi

Veri setinde 13 numerik özellik bulunmaktadır. Bu özellikler hastane yatış süreleri, yapılan testler ve prosedürleri temsil etmektedir.

**Önemli Numerik Özellikler:**

| Özellik | Ortalama | Medyan | Std. Sapma | Min | Max |
|---------|----------|--------|------------|-----|-----|
| **time_in_hospital** | 4.40 | 4.0 | 2.99 | 1 | 14 |
| **num_lab_procedures** | 43.10 | 44.0 | 19.67 | 1 | 132 |
| **num_procedures** | 1.34 | 1.0 | 1.71 | 0 | 6 |
| **num_medications** | 16.02 | 15.0 | 8.13 | 1 | 81 |
| **number_diagnoses** | 7.42 | 8.0 | 1.93 | 1 | 16 |
| **number_outpatient** | 0.37 | 0.0 | 1.27 | 0 | 42 |
| **number_emergency** | 0.20 | 0.0 | 0.93 | 0 | 76 |
| **number_inpatient** | 0.64 | 0.0 | 1.26 | 0 | 21 |

<div style="
    border-left: 5px solid #FF9800;
    background-color: #FFF3E0;
    padding: 10px;
    margin: 10px 0;
">
<strong>🦉 Gözlemler</strong><br>
    
1. **Hastanede Kalış Süresi:** Ortalama 4.4 gün, maksimum 14 gün. Dağılım sağa çarpık (right-skewed).

2. **Laboratuvar Testleri:** Ortalama 43 test yapılmış, bu oldukça yüksek bir değer. Hastaların karmaşık durumlarını gösteriyor.

3. **İlaç Sayısı:** Ortalama 16 farklı ilaç kullanımı, diabetes'in çoklu ilaç tedavisi gerektirdiğini doğruluyor.

4. **Önceki Başvurular:** Çoğu hastanın önceden acil servis veya ayakta tedavi başvurusu yok (medyan = 0).

5. **Teşhis Sayısı:** Ortalama 7.4 teşhis kodu, hastaların çoklu kronik hastalıklara sahip olduğunu gösteriyor.
</div>

![Numerik Dağılımlar](docs/exploratory_analysis/03_numeric_distributions.png)
*Şekil 2.3: Önemli numerik özelliklerin dağılımları (histogram ve istatistiksel çizgiler)*

### 2.5 Kategorik Özellikler Analizi

Veri setinde 37 kategorik özellik bulunmaktadır. Bu özellikler demografik bilgiler, kabul/taburculuk bilgileri ve ilaç kullanım durumlarını içermektedir.

**En Fazla Kategoriye Sahip Özellikler:**

| Özellik | Benzersiz Kategori | En Sık Değer | Yüzde |
|---------|-------------------|--------------|-------|
| **diag_1** (Birincil teşhis) | 717 | 428 (Kalp yetmezliği) | %6.74 |
| **diag_2** (İkincil teşhis) | 749 | 276 (Sıvı elektrolit bozukluğu) | %6.63 |
| **diag_3** (Üçüncül teşhis) | 790 | 250 (Diabetes mellitus) | %11.35 |
| **age** | 10 | [70-80) | %25.62 |
| **race** | 6 | Caucasian | %74.78 |
| **gender** | 3 | Female/Male | - |

**Demografik Bulgular:**

1. **Yaş Dağılımı:** Hastaların %25.6'sı 70-80 yaş grubunda. Yaşlı popülasyon ağırlıklı.

2. **Irk:** Hastaların %74.8'i Caucasian (Beyaz ırk), veri seti demografik olarak homojen.

3. **Cinsiyet:** Kadın-erkek dağılımı dengeli.


**İlaç Kullanım Analizi:**

23 farklı diabetes ilacı için kullanım durumu kaydedilmiştir. Her ilaç için değişiklik durumu (No, Up, Down, Steady) belirtilmiştir.

| İlaç | Kullanım Oranı | Yorum |
|------|----------------|-------|
| **insulin** | %53.44 | En yaygın ilaç |
| **metformin** | %19.64 | İkinci sırada |
| **glipizide** | %12.47 | Üçüncü sırada |
| **glyburide** | %10.47 | Dördüncü sırada |
| **pioglitazone** | %7.20 | - |
| **rosiglitazone** | %6.25 | - |

![Kategorik Dağılımlar](docs/exploratory_analysis/04_categorical_distributions.png)
*Şekil 2.4: Önemli kategorik özelliklerin frekans dağılımları*

![İlaç Kullanımı](docs/exploratory_analysis/08_medication_usage.png)
*Şekil 2.5: Diabetes ilaçlarının kullanım yüzdeleri*

### 2.6 Hedef Değişken ile Özellik İlişkileri

Özellikler ile hedef değişken (`readmitted`) arasındaki ilişkiler incelenmiş ve potansiyel tahmin ediciler belirlenmiştir.

**Numerik Özellikler vs Hedef Değişken:**

Box plot analizleri ile numerik özelliklerin readmission durumuna göre dağılımları karşılaştırılmıştır:

- **number_inpatient:** 30 gün içinde tekrar yatış yapanların önceki yatış sayısı daha yüksek
- **number_emergency:** Acil başvuru geçmişi olan hastalarda risk artışı
- **time_in_hospital:** İlk yatış süresi ile tekrar yatış arasında ilişki gözlemlenmiş
- **num_medications:** Çoklu ilaç kullananlar daha riskli

![Numerik vs Target](docs/exploratory_analysis/05_numeric_vs_target.png)
*Şekil 2.6: Numerik özelliklerin readmission durumuna göre kutu grafikleri*

**Kategorik Özellikler vs Hedef Değişken:**

Çapraz tablolar (crosstab) ile kategorik özelliklerin readmission oranları analiz edilmiştir:

- **age:** Genç (<30) ve çok yaşlı (>80) hastalarda risk değişiyor
- **gender:** Cinsiyet faktörü minimal etki gösteriyor
- **diabetesMed:** Diabetes ilacı kullanımı readmission ile ilişkili
- **race:** Irk grupları arasında farklılıklar var

![Kategorik vs Target](docs/exploratory_analysis/06_categorical_vs_target.png)
*Şekil 2.7: Kategorik özelliklerin readmission durumuna göre yüzdesel dağılımları*


### 2.7 Korelasyon Analizi

Numerik özellikler arasındaki korelasyonlar incelenmiş ve multicollinearity (çoklu doğrusal bağlantı) riski değerlendirilmiştir.

**Korelasyon Matrisi Bulguları:**

- Yüksek korelasyonlu özellik çifti (**|r| > 0.7**): **0 adet**
- Bu bulgu, modelleme aşamasında özellik çıkarma ihtiyacının düşük olduğunu göstermektedir

**Orta Düzeyde Korelasyonlar:**

- `num_medications` ve `num_lab_procedures` arasında pozitif korelasyon
- `time_in_hospital` ile prosedür sayıları arasında beklenen ilişkiler

![Korelasyon Matrisi](docs/exploratory_analysis/07_correlation_matrix.png)
*Şekil 2.8: Numerik özelliklerin korelasyon ısı haritası*

### 2.8 Veri Keşfi Sonuçları ve Çıkarımlar

Veri keşfi sürecinden elde edilen temel bulgular, modelleme aşaması için stratejik bir yol haritası sunmaktadır. Analizler sonucunda veri setinde duplikasyon saptanmamış, mevcut eksik değerler için ise verinin doğasına uygun yönetim stratejileri belirlenmiştir. Ancak, pozitif sınıf oranının %11,4 seviyesinde kalması belirgin bir sınıf dengesizliğine işaret etmekte; bu durum SMOTE veya sınıf ağırlıklandırma gibi tekniklerin kullanımını zorunlu kılmaktadır. 50 farklı öznitelik ile zengin bir klinik ve demografik veri yapısı sunan projede, 700'den fazla kategori içeren teşhis kodlarının yüksek kardinalite sorunu, kritik bir kodlama (encoding) stratejisinin geliştirilmesini gerektirmektedir. Sonuç olarak, hastaların geçmiş başvuruları, ilaç kullanım alışkanlıkları ve hastanede kalış süreleri ile tekrar yatış riski arasında saptanan anlamlı ilişkiler, bu değişkenlerin modelin tahmin gücü üzerinde yüksek bir potansiyele sahip olduğunu doğrulamaktadır.

## 3. VERİ ÖN İŞLEME

Veri keşfi aşamasında belirlenen stratejiler doğrultusunda, veri seti model eğitimine hazır hale getirilmiştir. Bu aşamada veri temizleme, eksik değer yönetimi, özellik mühendisliği ve ölçeklendirme işlemleri sistematik olarak uygulanmıştır.

### 3.1 Veri Temizleme İşlemleri

**3.1.1 Duplikasyon Kontrolü**

Veri setinde duplike satır kontrolü yapılmıştır:
- **Çıkarılan duplike satır:** 0
- **Sonuç:** Veri seti zaten temiz durumdadır

**3.1.2 Gereksiz Sütunların Çıkarılması**

Model eğitimi için uygun olmayan sütunlar veri setinden çıkarılmıştır:

| Sütun | Çıkarılma Gerekçesi |
|-------|---------------------|
| `encounter_id` | Sadece tanımlayıcı, tahmin gücü yok |
| `patient_nbr` | Sadece tanımlayıcı, tahmin gücü yok |
| `weight` | %96.86 eksik değer, kullanılamaz |
| `payer_code` | %39.56 eksik değer, düşük tahmin gücü |
| `medical_specialty` | %49.08 eksik değer, yüksek kardinalite |
| `readmitted` | Binary versiyonu (`readmitted_binary`) kullanılacak |


**Sonuç:** 6 sütun çıkarılmış, 44 sütun korunmuştur.

### 3.2 Hedef Değişken Dönüşümü

**Dönüşüm Stratejisi:**

Orijinal hedef değişken `readmitted` üç kategoriye sahipti (NO, >30, <30). Klinik açıdan en kritik olan 30 gün içindeki erken tekrar yatışları tespit etmek için binary dönüşüm uygulanmıştır.

**Dönüşüm Kuralı:**
```python
readmitted_binary = 1 if readmitted == '<30' else 0
```

**Dönüşüm Sonrası Dağılım:**

| Sınıf | Açıklama | Hasta Sayısı | Yüzde |
|-------|----------|--------------|-------|
| **0** | 30 günden sonra veya yatış yok | 90,409 | %88.84 |
| **1** | 30 gün içinde tekrar yatış | 11,357 | %11.16 |

<div style="
    border-left: 5px solid #7B1FA2;
    background-color: #F3E5F5;
    padding: 12px 14px;
    margin: 12px 0;
    border-radius: 4px;
">
<strong>📝 Önemli Not</strong><br>
Veri seti dengesiz (imbalanced) bir yapıya sahiptir. Model eğitimi sırasında bu dengesizliği gidermek için:
    
- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weight ayarlamaları
- Stratified cross-validation
  
teknikleri kullanılacaktır.
</div>



![Hedef Değişken Dönüşümü](docs/preprocessing/02_target_transformation.png)
*Şekil 3.1: Orijinal hedef değişken (sol) ve binary dönüşüm sonrası (sağ)*


### 3.3 Eksik Değer Yönetimi

Veri setinde `?` karakteri ile kodlanmış eksik değerler tespit edilmiş ve sistematik olarak yönetilmiştir.

**3.3.1 Uygulanan Strateji:**

1. **Yüksek Oranda Eksik (>50%):** Sütun tamamen çıkarılır
2. **Kategorik Değişkenler (<50%):** Mode (en sık değer) ile doldurulur
3. **Numerik Değişkenler (<50%):** Median ile doldurulur
4. **Teşhis Kodları:** "Missing" kategorisi olarak işaretlenir (eksiklik bilgi taşıyabilir)

**3.3.2 Yüksek Eksikliğe Sahip Çıkarılan Sütunlar:**

- `max_glu_serum`: %94+ eksik
- `A1Cresult`: %83+ eksik

Bu sütunlar eksiklik oranının çok yüksek olması nedeniyle imputation yerine tamamen çıkarılmıştır.

**3.3.3 Doldurma İşlemleri:**

**Kategorik Değişkenler (Mode Imputation):**
- `race`: En sık değer "Caucasian" ile dolduruldu
- `diag_1`, `diag_2`, `diag_3`: Mode değerleri ile dolduruldu

**Numerik Değişkenler (Median Imputation):**
- Numerik özelliklerde minimal eksiklik tespit edildi
- Median kullanılarak outlier'lardan etkilenmeden doldurma yapıldı

**Sonuç:** Tüm eksik değerler başarıyla yönetildi. 

**Kalan toplam eksik değer: 0**


### 3.4 Özellik Mühendisliği (Feature Engineering)

Veri setinin tahmin gücünü artırmak amacıyla domain knowledge (alan bilgisi) kullanılarak 8 yeni özellik oluşturulmuştur.

**3.4.1 Oluşturulan Yeni Özellikler:**

| # | Özellik Adı | Formül/Mantık | Gerekçe |
|---|-------------|---------------|---------|
| 1 | `age_numeric` | Yaş kategorileri → sayısal değer | Model için sürekli değişken daha etkili |
| 2 | `num_medications_changed` | Σ(ilaç ≠ 'No') | Tedavi değişikliği readmission ile ilişkili |
| 3 | `total_procedures` | lab_procedures + procedures | Toplam medikal müdahale yoğunluğu |
| 4 | `has_emergency_history` | (emergency > 0) OR (inpatient > 0) | Kronik hastalık ciddiyet göstergesi |
| 5 | `on_diabetes_med` | diabetesMed == 'Yes' | Hastalık yönetimi göstergesi |
| 6 | `med_changed` | change == 'Ch' | Tedavi etkinliği/uyum göstergesi |
| 7 | `procedure_intensity` | procedures / (time_in_hospital + 1) | Günlük prosedür yoğunluğu (normalize) |
| 8 | `medication_intensity` | medications / (time_in_hospital + 1) | Günlük ilaç yoğunluğu (normalize) |

**3.4.2 Yaş Kategorisi Dönüşümü:**

Orijinal yaş kategorileri sayısal değerlere dönüştürülmüştür:
```python
age_mapping = {
    '[0-10)': 5,   '[10-20)': 15,  '[20-30)': 25,  '[30-40)': 35,  '[40-50)': 45,
    '[50-60)': 55, '[60-70)': 65,  '[70-80)': 75,  '[80-90)': 85,  '[90-100)': 95
}
```


u dönüşüm, yaş ile readmission riski arasındaki non-linear ilişkiyi modellerin öğrenmesini kolaylaştırmaktadır.

**3.4.3 Teşhis Kodu (Diagnosis Code) Engineering:**

ICD-9 teşhis kodları 700+ benzersiz değere sahip olduğu için:

1. **Kategori Çıkarımı:** İlk 3 rakam ana hastalık kategorisini gösterir
2. **Frequency Encoding:** Her kategorinin veri setindeki sıklığı hesaplanmıştır

```python
diag_1_category = str(diag_1)[:3]  # Örn: "428" → Kalp yetmezliği
diag_1_freq = category_frequency_in_dataset
```

Bu yöntem, yüksek kardinaliteyi azaltırken klinik bilgiyi korumaktadır.

![Ön İşleme Özeti](docs/preprocessing/01_preprocessing_summary.png)
*Şekil 3.4: Ön işleme aşamalarında özellik sayısının değişimi ve yeni oluşturulan özellikler*

### 3.5 Kategorik Değişken Encoding

Makine öğrenmesi algoritmalarının kategorik değişkenleri işleyebilmesi için numerik formata dönüştürme işlemleri uygulanmıştır.

**3.5.1 Binary Encoding**

İki kategorili değişkenler için basit 0/1 encoding:
- `gender`: Male=0, Female=1

**3.5.2 One-Hot Encoding**

Nominal (sıralı olmayan) kategorik değişkenler için:
- `race`: 6 ırk kategorisi → 5 dummy variable (drop_first=True)
  - `race_AfricanAmerican`, `race_Asian`, `race_Caucasian`, `race_Hispanic`, `race_Other`

Bu yöntem, kategoriler arası yapay sıralama oluşturmayı önler.

**3.5.3 Label Encoding**

ID bazlı ordinal değişkenler için:
- `admission_type_id`: 8 kategori → 0-7 arası sayılar
- `discharge_disposition_id`: 29 kategori → 0-28 arası sayılar  
- `admission_source_id`: 21 kategori → 0-20 arası sayılar

**3.5.4 İlaç Değişkenleri için Ordinal Encoding**

23 farklı diabetes ilacı için değişiklik durumu ordinal olarak kodlanmıştır:

| Orijinal Değer | Encoded Değer | Anlamı |
|----------------|---------------|--------|
| No | 0 | İlaç kullanılmıyor |
| Steady | 1 | İlaç kullanılıyor, doz değişmedi |
| Up | 2 | Doz artırıldı |
| Down | 3 | Doz azaltıldı |

Bu encoding, ilaç değişikliğinin yönünü ve şiddetini korumaktadır.

**3.5.5 Encoding Sonrası Temizlik**

Encoding işlemlerinden sonra orijinal kategorik sütunlar veri setinden çıkarılmıştır:
- `age`, `race`, `diabetesMed`, `change`
- `diag_1`, `diag_2`, `diag_3` (frequency encodings korundu)

### 3.6 Aykırı Değer (Outlier) Yönetimi

**3.6.1 Tespit Metodu: IQR (Interquartile Range)**

Aykırı değer tespiti için istatistiksel olarak robust bir yöntem olan IQR kullanılmıştır:

```
Q1 = 25. persentil
Q3 = 75. persentil
IQR = Q3 - Q1
Lower Bound = Q1 - 3×IQR
Upper Bound = Q3 + 3×IQR
```

<div style="
    border-left: 5px solid #7B1FA2;
    background-color: #F3E5F5;
    padding: 12px 14px;
    margin: 12px 0;
    border-radius: 4px;
">
<strong>📝 Not</strong><br>

Standart 1.5×IQR yerine 3×IQR kullanılarak daha toleranslı bir yaklaşım benimsenmiştir.
</div>

**3.6.2 Tespit Edilen Outlier İstatistikleri:**

Toplam 29 özellikte aykırı değer tespit edilmiştir.

**En Çok Outlier İçeren Top 10 Özellik:**

| Özellik | Outlier Sayısı | Yüzde |
|---------|----------------|-------|
| `on_diabetes_med` | 23,403 | %23.00 |
| `metformin` | 19,988 | %19.64 |
| `number_outpatient` | 16,739 | %16.45 |
| `glipizide` | 12,686 | %12.47 |
| `number_emergency` | 11,383 | %11.19 |
| `glyburide` | 10,650 | %10.47 |
| `pioglitazone` | 7,328 | %7.20 |
| `rosiglitazone` | 6,365 | %6.25 |
| `glimepiride` | 5,191 | %5.10 |
| `procedure_intensity` | 2,808 | %2.76 |

**3.6.3 Outlier Yönetim Kararı:**

Aykırı değerler **veri setinde bırakılmıştır**. Bu karar aşağıdaki gerekçelere dayanmaktadır:

1. **Klinik Anlamlılık:** Medikal verilerde extreme değerler, ciddi hastalık durumlarını veya karmaşık vakaları temsil edebilir

2. **Tahmin Edici Değer:** Yüksek ilaç kullanımı veya sık hastane başvurusu, readmission riski ile güçlü ilişkili olabilir

3. **Veri Kaybı Riski:** Outlier'ları çıkarmak, önemli klinik patternlerin kaybolmasına neden olabilir

4. **Robust Ölçeklendirme:** Sonraki aşamada StandardScaler kullanımı, outlier'ların model üzerindeki negatif etkisini azaltacaktır

### 3.7 Özellik Ölçeklendirme (Feature Scaling)

**3.7.1 Ölçeklendirme Gereksinimi**

Farklı özelliklerin farklı ölçeklerde olması (örn: yaş 1-100, laboratuvar testi sayısı 1-132), gradient-based algoritmalar için problem oluşturabilir. Ölçeklendirme:
- Modellerin daha hızlı convergence sağlamasını
- Uzaklık bazlı algoritmaların (KNN, SVM) doğru çalışmasını
- Düzenli (regularization) terimlerin adil uygulanmasını

sağlamaktadır.

**3.7.2 Seçilen Yöntem: StandardScaler (Z-Score Normalization)**

StandardScaler, her özelliği ortalaması 0, standart sapması 1 olacak şekilde dönüştürür:

$$z = \frac{x - \mu}{\sigma}$$

Burada:
- $x$: Orijinal değer
- $\mu$: Özelliğin ortalaması
- $\sigma$: Özelliğin standart sapması
- $z$: Ölçeklendirilmiş değer

**3.7.3 Ölçeklendirme İstatistikleri:**

- **Ölçeklendirilen özellik sayısı:** 44
- **Hedef değişken:** Ölçeklendirmeye dahil edilmemiştir (binary 0/1 olarak korunmuştur)

**3.7.4 Scaler Objesinin Kaydedilmesi:**

Eğitim verisinde fit edilen scaler objesi `scaler.pkl` dosyasına kaydedilmiştir. Bu, gelecekte yeni verilerin aynı parametrelerle ölçeklendirilmesini sağlar (train-test consistency).

![Ölçeklendirilmiş Özellikler](docs/preprocessing/03_scaled_features_distribution.png)
*Şekil 3.7: Ölçeklendirme sonrası özelliklerin dağılımları (ortalama=0 etrafında merkezlenmiş)*

## 4. MODEL EĞİTİMİ

Veri ön işleme aşaması tamamlandıktan sonra, diabetik hastaların 30 gün içinde hastaneye tekrar yatış riskini tahmin etmek üzere 5 farklı makine öğrenmesi modeli eğitilmiştir. Bu bölümde train-test split stratejisi, sınıf dengeleme yöntemleri, model seçimi ve eğitim süreci detaylı olarak açıklanmıştır.

### 4.1 Train-Test Split Stratejisi

**4.1.1 Bölme Oranı ve Yöntem**

Veri seti, modelin genelleme yeteneğini değerlendirmek için eğitim ve test setlerine ayrılmıştır:

| Set | Örnek Sayısı | Yüzde | Amaç |
|-----|--------------|-------|------|
| **Train** | 81,412 | %80 | Model eğitimi |
| **Test** | 20,354 | %20 | Model değerlendirme |

**Kullanılan Yöntem:** `Stratified Split`

Stratified split, hedef değişkenin sınıf dağılımının hem train hem de test setinde korunmasını sağlar. Bu, özellikle dengesiz veri setlerinde kritik öneme sahiptir.

**4.1.2 Train ve Test Setlerinde Sınıf Dağılımı**

**Train Set:**
- Sınıf 0 (Yatış yok/geç): 72,326 (%88.84)
- Sınıf 1 (30 gün içinde yatış): 9,086 (%11.16)

**Test Set:**
- Sınıf 0 (Yatış yok/geç): 18,083 (%88.84)
- Sınıf 1 (30 gün içinde yatış): 2,271 (%11.16)

Her iki sette de sınıf oranları korunmuştur (%88.84 vs %11.16), bu da stratified split'in başarılı uygulandığını göstermektedir.

### 4.2 Class Imbalance Handling (Sınıf Dengeleme)

**4.2.1 Problem Tanımı**

Veri setinde ciddi bir sınıf dengesizliği bulunmaktadır:
- **Imbalance Ratio:** 7.96:1 (Negatif sınıf / Pozitif sınıf)

Bu dengesizlik, modellerin çoğunluk sınıfına (Sınıf 0) bias göstermesine ve azınlık sınıfını (Sınıf 1 - kritik hastalar) doğru tahmin edememesine neden olabilir.

**4.2.2 Uygulanan Yöntem: SMOTE + Random Undersampling**

Sınıf dengesizliğini gidermek için hibrit bir yaklaşım benimsenmiştir:

**1. SMOTE (Synthetic Minority Over-sampling Technique):**
- Azınlık sınıfı (Sınıf 1) sentetik örnekler oluşturarak artırılmıştır
- Sampling strategy: 0.5 (Azınlık sınıfı %50 oranına çıkarıldı)
- Yöntem: K-nearest neighbors kullanarak sentetik örnekler üretildi

**2. Random Undersampling:**
- Çoğunluk sınıfı (Sınıf 0) rastgele örnekler çıkarılarak azaltılmıştır
- Sampling strategy: 0.8 (Çoğunluk sınıfı %80 oranına çekildi)

**4.2.3 Resampling Sonuçları**

**Resampled Train Set:**
- Sınıf 0: 45,203 (%55.56)
- Sınıf 1: 36,163 (%44.44)
- **Yeni Imbalance Ratio:** 1.25:1

Sınıf dengesi 7.96:1 oranından 1.25:1 oranına iyileştirilmiştir. Bu dengeli veri seti, modellerin her iki sınıfı da daha iyi öğrenmesini sağlamıştır.

![Class Balance](docs/model_training/01_class_balance.png)
*Şekil 4.2: Orijinal train set (sol) ve SMOTE+Undersampling sonrası (sağ) sınıf dağılımları*

<div style="
    border-left: 5px solid #7B1FA2;
    background-color: #F3E5F5;
    padding: 12px 14px;
    margin: 12px 0;
    border-radius: 4px;
">
<strong>📝 Önemli Not</strong><br>
SMOTE ve undersampling yalnızca train set'e uygulanmıştır. Test set, modelin gerçek dünya performansını değerlendirmek için orijinal dengesiz dağılımı ile korunmuştur.
</div>

### 4.3 Model Seçimi ve Hiperparametreler

Derste işlenen temel algoritmalar ve state-of-the-art yöntemler olmak üzere 5 farklı model eğitilmiştir. Her modelin seçim gerekçesi ve kullanılan hiperparametreleri aşağıda detaylandırılmıştır.

**4.3.1 Logistic Regression (Baseline Model)**

Doğrusal ve yorumlanabilir yapısı sayesinde, diğer karmaşık modellerin performansını ölçmek için temel bir referans (baseline) noktası oluşturması amacıyla seçilmiştir. Modelin katsayı analizi sunması, klinik değişkenlerin risk üzerindeki doğrudan etkilerinin şeffaf bir şekilde değerlendirilmesine olanak tanımaktadır.

**Hiperparametreler:**
- `solver`: 'liblinear' (Küçük veri setleri için optimize)
- `max_iter`: 1000 (Convergence garantisi)
- `class_weight`: 'balanced' (Sınıf dengesizliği için ek koruma)
- `random_state`: 42

**4.3.2 Random Forest Classifier**

Ensemble (topluluk) öğrenme mantığıyla çalışan bu model, verideki aşırı öğrenme (overfitting) riskine karşı dayanıklı yapısı ve değişkenler arasındaki doğrusal olmayan ilişkileri yakalayabilme yeteneği nedeniyle tercih edilmiştir. Ayrıca klinik içgörüler elde etmek adına kritik öneme sahip olan özellik önem analizini (feature importance) başarıyla gerçekleştirmektedir.

**Hiperparametreler:**
- `n_estimators`: 100 (100 karar ağacı)
- `max_depth`: 10 (Overfitting'i önlemek için)
- `min_samples_split`: 10
- `min_samples_leaf`: 4
- `class_weight`: 'balanced'
- `n_jobs`: -1 (Tüm CPU'lar kullanılır)

**4.3.3 XGBoost (Extreme Gradient Boosting)**

Modern makine öğrenmesi çalışmalarında yüksek başarı oranlarıyla bilinen bu algoritma, düzenlileştirme (regularization) teknikleri sayesinde modelin genelleme yeteneğini maksimize etmek için kullanılmıştır. Karmaşık veri yapılarında hızlı ve etkili sonuçlar üretmesi, bu projenin tahmin gücünü artırmada stratejik bir rol oynamaktadır.

**Hiperparametreler:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `subsample`: 0.8 (Row subsampling)
- `colsample_bytree`: 0.8 (Column subsampling)
- `eval_metric`: 'logloss'

**4.3.4 LightGBM (Light Gradient Boosting Machine)**

Büyük veri setlerinde diğer boosting yöntemlerine göre çok daha hızlı eğitim süresi ve düşük bellek kullanımı sunması nedeniyle seçilmiştir. Yaprak odaklı (leaf-wise) büyüme stratejisi sayesinde karmaşık örüntüleri daha derinlemesine analiz edebilme kabiliyetine sahiptir.

**Hiperparametreler:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `class_weight`: 'balanced'

**4.3.5 Support Vector Machine (SVM)**

Kernel yöntemlerini kullanarak yüksek boyutlu veri uzaylarında etkili karar sınırları oluşturabilme yeteneği nedeniyle karşılaştırma grubuna dahil edilmiştir. Teorik açıdan güçlü bir sınıflandırıcı olması, verinin farklı geometrik temsilleri üzerinden risk tahmininde bulunulmasına imkân sağlamaktadır.

**Hiperparametreler:**
- `kernel`: 'rbf' (Radial Basis Function)
- `C`: 1.0 (Regularization parameter)
- `gamma`: 'scale'
- `class_weight`: 'balanced'
- `probability`: True (ROC curve için gerekli)

### 4.4 Cross-Validation Stratejisi

Model performanslarının güvenilir bir şekilde değerlendirilmesi için **5-Fold Stratified Cross-Validation** uygulanmıştır.

**4.4.1 Yöntem Açıklaması**

1. **Stratified K-Fold:** Veri 5 eşit parçaya bölünür, her parçada sınıf oranları korunur
2. **Her iterasyonda:** 4 parça eğitim, 1 parça validasyon için kullanılır
3. **5 iterasyon sonunda:** Ortalama ve standart sapma hesaplanır

**4.4.2 Değerlendirme Metrikleri**

- **Accuracy:** Genel doğruluk oranı
- **ROC-AUC:** Discriminative power (ayırt etme gücü)

**4.4.3 Cross-Validation Sonuçları**

| Model | CV Accuracy (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
|-------|--------------------------|-------------------------|
| **Logistic Regression** | 0.6120 ± 0.0042 | 0.6518 ± 0.0062 |
| **Random Forest** | 0.8122 ± 0.0059 | 0.8837 ± 0.0036 |
| **XGBoost** | 0.8834 ± 0.0016 | 0.9149 ± 0.0009 |
| **LightGBM** | 0.8855 ± 0.0013 | 0.9159 ± 0.0014 |
| **SVM** | 0.7056 ± 0.0071 | 0.7800 ± 0.0054 |

**Gözlemler:**

1. **En Yüksek CV Accuracy:** LightGBM (0.8855)
2. **En Yüksek CV ROC-AUC:** LightGBM (0.9159)
3. **En Düşük Varyans:** XGBoost ve LightGBM (std < 0.002)
4. **Baseline:** Logistic Regression (0.6120 accuracy)

Gradient boosting modelleri (XGBoost, LightGBM) cross-validation'da belirgin üstünlük göstermiştir.

### 4.5 Model Eğitim Süreci ve Performans Metrikleri

Her model, resampled train set ile eğitilmiş ve hem train hem de test setlerinde değerlendirilmiştir.

**4.5.1 Test Set Performans Karşılaştırması**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Eğitim Süresi |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Logistic Regression** | 0.6373 | 0.1676 | 0.5672 | 0.2587 | 0.6530 | 2.73s |
| **Random Forest** | 0.7989 | 0.2076 | 0.2849 | 0.2402 | 0.6507 | 5.85s |
| **XGBoost** | 0.8862 | 0.4096 | 0.0449 | 0.0810 | 0.6674 | 2.72s |
| **LightGBM** | 0.8851 | 0.4086 | 0.0669 | 0.1150 | 0.6743 | 6.47s |
| **SVM** | 0.6693 | 0.1660 | 0.4883 | 0.2478 | 0.6246 | 6896.93s |

![Model Comparison Metrics](docs/model_training/02_model_comparison_metrics.png)
*Şekil 4.5: Modellerin test set'teki performans metriklerinin karşılaştırması*


**4.5.2 Metrik Bazlı Analizler**

**Accuracy (Doğruluk Oranı):** En yüksek değer %88,62 ile XGBoost modelinde görülürken, en düşük değer %63,73 ile Lojistik Regresyon modelinde saptanmıştır. Ancak veri setindeki belirgin sınıf dengesizliği nedeniyle, XGBoost ve LightGBM modellerinin sunduğu yüksek doğruluk oranlarının yanıltıcı olabileceği değerlendirilmektedir.

**Precision (Kesinlik):** XGBoost %40,96 ile **"tekrar yatış"** tahminlerinde en yüksek kesinliği sağlarken, SVM %16,60 oranıyla bu alanda en düşük performansı sergilemiştir. Bu metrik, modellerin pozitif olarak sınıflandırdığı vakaların ne kadarının gerçek klinik karşılığı olduğunu ölçerek "yanlış alarm" oranını belirlemektedir.

**Recall (Duyarlılık - Sensitivity):** Klinik açıdan en kritik parametre olan duyarlılıkta, Lojistik Regresyon %56,72 ile en yüksek başarıyı gösterirken XGBoost %04,49 ile en zayıf sonucu üretmiştir. Bu sonuçlar, Lojistik Regresyon'un gerçek riskli hastaları tespit etme gücünün diğer modellere göre çok daha yüksek olduğunu kanıtlamaktadır.

**F1-Score (Precision-Recall Dengesi):** Kesinlik ve duyarlılık değerlerinin harmonik ortalamasını temsil eden F1-Skorunda, Lojistik Regresyon 0,2587 ile en dengeli performansı sergilemiştir. Buna karşın XGBoost, özellikle düşük duyarlılık oranının etkisiyle 0,0810 seviyesinde kalarak en düşük denge skorunu almıştır.

**ROC-AUC (Discriminative Power):** Modellerin sınıfları birbirinden ayırt etme yeteneğini gösteren bu ölçekte LightGBM 0,6743 ile en iyi performansı gösterirken, SVM 0,6246 ile en sonda yer almıştır. Genel tabloda tüm modellerin 0,60-0,68 aralığında kalması, problemin doğası gereği modellerin orta seviyede bir ayırt etme gücüne sahip olduğunu göstermektedir.

**Training Time (Eğitim Süresi):** XGBoost 2,72 saniyelik eğitim süresiyle operasyonel açıdan en hızlı model olurken, SVM yaklaşık 1,9 saatlik (6896,93s) süresiyle en yavaş model olarak kaydedilmiştir. SVM modelinin sergilediği bu aşırı gecikme, kullanılan RBF çekirdeğinin (kernel) hesaplama maliyetinin yüksekliğinden kaynaklanmaktadır.

**4.5.3 Önemli Bulgular ve Yorumlar**

**1. Doğruluk ve Duyarlılık Dengesi (Accuracy vs. Recall Trade-off):**

XGBoost ve LightGBM modelleri genel doğruluk oranında yüksek başarı sergilese de, klinik açıdan kritik olan pozitif sınıfı (Sınıf 1) tespit etmede yetersiz kalarak düşük duyarlılık değerleri üretmiştir. Buna karşın Lojistik Regresyon, en düşük genel doğruluğa sahip olmasına rağmen en yüksek duyarlılık oranına ulaşarak riskli hastaların yakalanmasında en etkili model olmuştur.

**2. Klinik Perspektif ve Hata Analizi**

Hastaneye tekrar yatış tahminlerinde, riskli bir hastanın gözden kaçırılması anlamına gelen "Hatalı Negatif" (False Negative) sonuçlar, en yüksek maliyetli ve hayati risk taşıyan hata tipi olarak değerlendirilmektedir. Bu nedenle, daha fazla "yanlış alarm" üretilse dahi (düşük kesinlik), kritik hastaları kaçırmamak adına yüksek duyarlılık değerine sahip modellerin tercih edilmesi klinik önceliklerle örtüşmektedir.

**3. Ayırt Etme Gücü (ROC-AUC) Analizi**

Tüm modellerin ROC-AUC değerlerinin 0,62 ile 0,67 gibi orta bir aralıkta kümelenmesi, algoritmaların sınıfları birbirinden ayırt etme gücünün belirli bir sınırda kaldığını göstermektedir. Bu durum, veri setinin doğasından kaynaklanan karmaşıklığın ve sınıf dengesizliğinin, tahmin sürecini tüm modeller için zorlaştıran temel unsur olduğunu kanıtlamaktadır.

**4. Hesaplama Verimliliği (Computational Efficiency)**

Destek Vektör Makineleri (SVM) modelinin sergilediği aşırı uzun eğitim süresi, bu algoritmanın gerçek zamanlı klinik uygulamalar ve büyük ölçekli veri setleri için pratik bir seçenek olmadığını ortaya koymuştur. Lojistik Regresyon, XGBoost ve LightGBM ise hızlı işlem süreleriyle operasyonel verimlilik açısından çok daha uygulanabilir alternatifler olarak öne çıkmaktadır.

### 4.6 ROC Eğrisi Analizi

ROC (Receiver Operating Characteristic) eğrisi, farklı classification threshold'larında modelin True Positive Rate (TPR) ve False Positive Rate (FPR) ilişkisini görselleştirir.

**4.6.1 ROC Eğrisi**

- **Perfect Classifier:** AUC = 1.0 (Sol üst köşeden geçer)
- **Random Classifier:** AUC = 0.5 (45° diagonal çizgi)
- **Worse than Random:** AUC < 0.5

**ROC-AUC Değerlendirme Skalası:**
- 0.90-1.00: Mükemmel
- 0.80-0.90: Çok iyi
- 0.70-0.80: İyi
- 0.60-0.70: Orta ⬅ **modellerim**
- 0.50-0.60: Zayıf

![ROC Curves](docs/model_training/03_roc_curves.png)
*Şekil 4.3: Tüm modellerin ROC eğrileri ve AUC değerleri karşılaştırması*

**4.6.2 Model Bazlı ROC Analizi**

1. **LightGBM (AUC=0.6743):** En iyi discriminative power
2. **XGBoost (AUC=0.6674):** İkinci sırada, LightGBM'e yakın
3. **Logistic Regression (AUC=0.6530):** Baseline modelden biraz daha iyi
4. **Random Forest (AUC=0.6507):** Orta seviye
5. **SVM (AUC=0.6246):** En düşük, random'a yakın

**4.6.3 Threshold Seçimi**

ROC eğrisi, optimal threshold seçimi için kullanılabilir:
- **Yüksek Recall İstiyorsak:** Threshold'u düşürürüz (daha fazla pozitif tahmin)
- **Yüksek Precision İstiyorsak:** Threshold'u yükselttiriz (daha az pozitif tahmin)
- **Klinik Karar:** Recall'u önceliklendirmek mantıklı (hastaları kaçırmamak için)

### 4.7 Confusion Matrix Analizi

Confusion matrix, modelin tahminlerinin detaylı dökümünü sağlar.

**Confusion Matrix Bileşenleri:**

|  | Tahmin: Negatif | Tahmin: Pozitif |
|--|----------------|-----------------|
| **Gerçek: Negatif** | TN (True Negative) | FP (False Positive) |
| **Gerçek: Pozitif** | FN (False Negative) | TP (True Positive) |

![Confusion Matrices](docs/model_training/04_confusion_matrices.png)
*Şekil 4.4: Tüm modellerin confusion matrix görselleştirmeleri*

**4.7.1 Model Karşılaştırması (Confusion Matrix Bazlı)**

**Logistic Regression:**
- TP: 1288, FN: 983 (Recall = 0.567)
- En yüksek recall, en az hasta kaçırıyor
- Ancak FP sayısı yüksek (6299)

**XGBoost:**
- TP: 102, FN: 2169 (Recall = 0.045)
- Çok yüksek FN, kritik hastaların %95.5'ini kaçırıyor
- Yüksek accuracy aldatıcı, klinik kullanım için uygun değil

**LightGBM:**
- TP: 152, FN: 2119 (Recall = 0.067)
- XGBoost'tan biraz daha iyi ama hala çok düşük recall

### 4.8 Model Eğitimi Sonuçları ve Değerlendirme

**4.8.1 Genel Değerlendirme**

5 farklı makine öğrenmesi modeli başarıyla eğitilmiş ve test edilmiştir. Her modelin farklı güçlü ve zayıf yönleri olduğu tespit edilmiştir.

**Model Performans Özeti:**

| Kriter | En İyi Model | Değer | Yorum |
|--------|--------------|-------|-------|
| **CV Accuracy** | LightGBM | 0.8855 | Eğitim seti performansı |
| **CV ROC-AUC** | LightGBM | 0.9159 | Eğitim seti discrimination |
| **Test Accuracy** | XGBoost | 0.8862 | Genel doğruluk |
| **Test Recall** | Logistic Regression | 0.5672 | Kritik hasta tespiti |
| **Test F1-Score** | Logistic Regression | 0.2587 | Dengeli metrik |
| **Test ROC-AUC** | LightGBM | 0.6743 | Test set discrimination |
| **Training Speed** | XGBoost | 2.72s | Hız |

**4.8.2 Kritik Bulgular**

**1. Çapraz Doğrulama ve Test Performansı Arasındaki Sapma:** XGBoost ve LightGBM modellerinde Çapraz Doğrulama (CV) aşamasında gözlemlenen yüksek ROC-AUC değerlerinin (0.91+), test setinde 0.67 seviyesine gerilemesi önemli bir performans sapmasına işaret etmektedir. Bu durum, modellerin yeniden örneklenmiş (resampled) eğitim verisine aşırı uyum sağladığını ve orijinal dengesiz dağılıma sahip test verisinde genelleme yeteneğinin düştüğünü kanıtlamaktadır.

**2. Duyarlılık (Recall) Sorunu ve Klinik Yansımaları:** XGBoost ve LightGBM gibi gelişmiş boosting modellerinin %4 ile %6 arasında seyreden çok düşük duyarlılık oranları üretmesi, klinik uygulama açısından kabul edilemez düzeyde yüksek bir hatalı negatif (FN) oranına yol açmaktadır. Bu kısıtın aşılması ve riskli hastaların gözden kaçırılmaması için karar eşiği optimizasyonu (threshold tuning) süreçlerinin uygulanması bir zorunluluk olarak görülmektedir.

**3. Lojistik Regresyon Modelinin Beklenmeyen Başarısı:** Daha basit bir mimariye sahip olmasına rağmen Lojistik Regresyon, tüm modeller arasında en yüksek duyarlılık değerini ve en iyi F1 skorunu üreterek dikkat çekici bir performans sergilemiştir. Modelin sergilediği bu kararlı yapı ve kritik vakaları yakalama gücü, gerçek dünya klinik uygulamalarında ve canlı sistem dağıtımlarında (deployment) diğer karmaşık modellere göre daha uygun bir seçenek olduğunu ortaya koymaktadır.

**4. Sınıf Dengesizliği ile Mücadeledeki Zorluklar:** Eğitim aşamasında SMOTE tekniği uygulanmasına rağmen, modellerin test setinde çoğunluk sınıfına karşı olan eğilimlerinin (bias) devam ettiği ve azınlık sınıfını tahmin etmekte zorlandığı saptanmıştır. Bu sorunun çözümü için gelecekteki çalışmalarda daha agresif sınıf ağırlıklandırma yöntemlerinin veya maliyete duyarlı öğrenme stratejilerinin kullanılması önerilmektedir.

## 5. MODEL DEĞERLENDİRME

Model eğitim aşaması tamamlandıktan sonra, modellerin performansları detaylı olarak analiz edilmiştir. Bu bölümde precision-recall analizi, threshold optimizasyonu, model kalibrasyonu, feature importance ve error analizi gibi ileri seviye değerlendirme teknikleri uygulanmıştır.

### 5.1 Precision-Recall Analizi

**5.1.1 Precision-Recall vs ROC Curve**

Dengesiz veri setlerinde (imbalanced datasets), ROC eğrisi yanıltıcı olabilir çünkü True Negative (TN) sayısı çok yüksektir ve bu FPR'yi düşük gösterir. Precision-Recall eğrisi, azınlık sınıfına (pozitif sınıf) odaklandığı için bu problemde daha informatif bir metriktir.

**Neden Precision-Recall?**
- **Precision:** Pozitif tahminlerin doğruluğu (FP'leri penalize eder)
- **Recall:** Pozitif örnekleri yakalama gücü (FN'leri penalize eder)
- Dengesiz veride her iki metrik de kritik

**5.1.2 Average Precision (AP) Sonuçları**

Average Precision, precision-recall curve altındaki alanı temsil eder ve tüm threshold değerlerinde modelin performansını özetler.

| Model | Average Precision | Yorumlama |
|-------|-------------------|-----------|
| **LightGBM** | 0.2235 | En yüksek AP |
| **XGBoost** | 0.2178 | İkinci sırada |
| **Logistic Regression** | 0.1981 | Baseline'ın üstünde |
| **Random Forest** | 0.1889 | Orta seviye |
| **SVM** | 0.1699 | En düşük |

**Genel Değerlendirme:**
- Tüm modellerin AP değerleri 0.17-0.22 aralığında (düşük-orta)
- Bu, problemin doğası gereği tahmin zorluğunu gösteriyor
- Azınlık sınıfı (11.16%) için precision-recall dengesi zor

![Precision-Recall Curves](docs/model_evaluation/01_precision_recall_curves.png)
*Şekil 5.1: Tüm modellerin Precision-Recall eğrileri ve Average Precision değerleri*

### 5.2 Threshold Optimizasyonu

**5.2.1 Threshold Kavramı**

Sınıflandırma modellerinde default threshold genellikle 0.5'tir:
- Eğer P(readmission) ≥ 0.5 → Tahmin = 1 (Readmission)
- Eğer P(readmission) < 0.5 → Tahmin = 0 (No readmission)

Ancak bu threshold her problem için optimal değildir. Özellikle:
- Dengesiz veri setlerinde
- Farklı hata tiplerinin farklı maliyetleri olduğunda
- Belirli bir metriği optimize etmek istediğimizde

threshold'u değiştirerek precision-recall trade-off'unu ayarlayabiliriz.

**5.2.2 Optimal Threshold Bulma Stratejileri**

İki farklı optimizasyon kriteri kullanılmıştır:

**1. F1-Optimal Threshold:**
- F1-score'u (precision ve recall'un harmonik ortalaması) maksimize eder
- Dengeli performans için kullanılır

**2. Youden's J-Optimal Threshold:**
- J = Sensitivity + Specificity - 1 formülünü maksimize eder
- ROC curve'den türetilir
- Genel discriminative power için kullanılır

**5.2.3 Optimal Threshold Sonuçları**

| Model | F1-Optimal Threshold | F1-Score | J-Optimal Threshold | J-Score |
|-------|---------------------|----------|---------------------|---------|
| **Logistic Regression** | 0.500 | 0.2587 | 0.455 | 0.2250 |
| **Random Forest** | 0.450 | 0.2636 | 0.397 | 0.2230 |
| **XGBoost** | 0.200 | 0.2715 | 0.209 | 0.2504 |
| **LightGBM** | 0.300 | 0.2788 | 0.240 | 0.2527 |
| **SVM** | 0.350 | 0.2488 | 0.313 | 0.1994 |


<div style="
    border-left: 5px solid #FF9800;
    background-color: #FFF3E0;
    padding: 10px;
    margin: 10px 0;
">
<strong>🦉 Gözlemler</strong><br>

1. **Boosting Modelleri Düşük Threshold İster:**
   - XGBoost: 0.200, LightGBM: 0.300
   - Bu modeller çok confident tahminler yapıyor (overconfident)
   - Threshold düşürüldüğünde F1-score arttı (0.08 → 0.27)

2. **Linear Model Kararlı:**
   - Logistic Regression: 0.500 (default optimal)
   - Model iyi kalibre edilmiş

3. **F1-Score İyileştirme:**
   - LightGBM threshold 0.5 → 0.3: F1 0.115 → 0.279 (+%142)
   - XGBoost threshold 0.5 → 0.2: F1 0.081 → 0.271 (+%235)
</div>

### 5.3 Model Kalibrasyonu Analizi

**5.3.1 Kalibrasyon Kavramı ve Güvenilirlik Analizi**

Model kalibrasyonu, bir sınıflandırıcının ürettiği olasılık tahminlerinin gerçek dünya sonuçlarıyla ne ölçüde örtüştüğünü ve istatistiksel olarak ne kadar güvenilir olduğunu değerlendirmektedir. İdeal şekilde kalibre edilmiş bir modelde, tahmin edilen olasılıkların gerçekleşme oranlarıyla birebir eşleşmesi beklenir; örneğin, %30 tekrar yatış olasılığı atanan 100 hastadan yaklaşık 30'unun gerçekte hastaneye geri dönmesi modelin tutarlılığını kanıtlamaktadır. Bu durumun görsel bir göstergesi olan kalibrasyon eğrisinin (calibration curve), ideal referans noktası kabul edilen 45 derecelik diyagonal çizgiye yakınlığı modelin güvenilirliğini belirlemektedir.

Kalibrasyon sürecinde karşılaşılan temel sapmalar iki ana grupta incelenmektedir:

- **Aşırı Güvenli (Overconfident) Tahminler:** Modelin gerçek olasılıktan daha yüksek değerler ataması durumudur; bu senaryoda kalibrasyon eğrisi diyagonal çizginin altında kalmakta ve modelin riskleri olduğundan daha büyük göstermesine neden olmaktadır.

- **Yetersiz Güvenli (Underconfident) Tahminler:** Modelin gerçek risk oranlarını yansıtmakta zayıf kalarak daha düşük olasılıklar üretmesidir; bu durumda eğri diyagonalin üzerinde seyretmekte, bu da gerçek risklerin gözden kaçırılmasına yol açabilmektedir.

**5.3.2 Kalibrasyon Sonuçları**

![Calibration Curves](docs/model_evaluation/03_calibration_curves.png)
*Şekil 5.3: Model kalibrasyon eğrileri - tahmin edilen vs. gözlenen olasılıklar*

### 5.4 Özellik Önemi Analizi

Özellik önemi (`feature importance`), modellerin hangi özellikleri ne kadar önemli bulduğunu gösterir. Bu analiz hem model yorumlanabilirliği hem de klinik içgörüler açısından kritiktir.

**5.4.1 Ağaç Tabanlı Modeller için Özellik Seçimi**

Random Forest, XGBoost ve LightGBM için Gini importance (veya gain) kullanılarak feature importance hesaplanmıştır.

![Feature Importance](docs/model_evaluation/04_feature_importance.png)
*Şekil 5.4.1: Tree-based modellerde en önemli 20 özellik*

**Ortak En Önemli Özellikler (Tüm Tree Models):**

| Sıra | Özellik | Klinik Anlamı |
|------|---------|---------------|
| 1 | `number_inpatient` | Geçmiş hastaneye yatış sayısı |
| 2 | `number_emergency` | Acil servis başvuru sayısı |
| 3 | `time_in_hospital` | Bu yatışta hastanede kalma süresi |
| 4 | `discharge_disposition_id` | Taburcu durumu/yeri |
| 5 | `number_diagnoses` | Toplam teşhis sayısı |
| 6 | `num_medications` | Kullanılan ilaç sayısı |
| 7 | `num_lab_procedures` | Yapılan laboratuvar testi sayısı |
| 8 | `age_numeric` | Hasta yaşı |
| 9 | `diag_1_freq` | Birincil teşhis sıklığı |
| 10 | `admission_type_id` | Kabul tipi |

**Özelliklerin Klinik Yorumlanması ve Risk Değerlendirmesi**

- **Sağlık Hizmeti Kullanım Geçmişinin Kritik Önemi:** Yapılan analizler sonucunda, `number_inpatient` (geçmiş yatarak tedavi sayısı) ve `number_emergency` (acil servis başvuru sayısı) değişkenlerinin modelin en güçlü tahmin edicileri olduğu saptanmıştır. Bu durum, kronik hastalık yönetiminde geçmiş başvuruların gelecekteki tekrar yatış riskleri için en belirgin gösterge olduğunu ve sağlık hizmeti kullanım geçmişinin (healthcare utilization history) ana risk faktörü olarak değerlendirilmesi gerektiğini ortaya koymaktadır.

- **Hastanede Kalış Süresi ve Taburcu Durumu Etkisi:** `time_in_hospital` değişkeni, yüksek önem derecesine sahip olsa da karmaşık bir ilişki sergilemektedir; uzun yatış süreleri vaka karmaşıklığına işaret ederken, çok kısa süreli yatışlar ise yetersiz tedavi (under-treatment) riski nedeniyle tekrar yatış olasılığını artırabilmektedir. Benzer şekilde, `discharge_disposition_id` özniteliği, hastanın taburcu edildikten sonra eve mi yoksa bir rehabilitasyon merkezine mi sevk edildiğinin, iyileşme sürecindeki takip kalitesi ve dolayısıyla risk seviyesi üzerinde doğrudan belirleyici olduğunu göstermektedir.

- **Hastalık Karmaşıklığı ve Polifarmasi Faktörü:** Modelde yüksek önem arz eden number_diagnoses ve num_medications değişkenleri, hastanın komorbidite (eşlik eden hastalık) yükünü ve ilaç kullanım yoğunluğunu yansıtmaktadır. Çoklu morbidite durumu genel klinik riski yükseltirken; polifarmasi (çoklu ilaç kullanımı), hasta uyumu (compliance) sorunları ve ilaç etkileşimleri nedeniyle tekrar yatış tetikleyicisi olarak öne çıkmaktadır.

**5.4.2 Lojistik Regresyon Katsayı Analizi**

Lojistik Regresyon, doğrusal bir model mimarisine sahip olması nedeniyle katsayıların (coefficients) doğrudan yorumlanmasına imkan tanıyarak klinik şeffaflık sağlamaktadır. Bu modelde her bir katsayı, ilgili öznitelikteki bir birimlik değişimin, hastanın tekrar yatış olasılığının log-oranı (log-odds) üzerindeki etkisini ifade etmekte; böylece klinisyenlerin her bir risk faktörünün ağırlığını somut verilerle görmesine olanak tanımaktadır.

![Logistic Coefficients](docs/model_evaluation/05_logistic_coefficients.png)
*Şekil 5.4.2: Logistic Regression'da en önemli 20 özellik (coefficient büyüklüğüne göre)*

**Pozitif Coefficient (Readmission Riskini Artırır):**

| Özellik | Coefficient | Yorum |
|---------|-------------|-------|
| `number_emergency` | +0.45 | Acil başvuru geçmişi → ↑ Risk |
| `number_inpatient` | +0.38 | Önceki yatışlar → ↑ Risk |
| `discharge_disposition_id` (bazı değerler) | +0.32 | Belirli taburcu durumları → ↑ Risk |
| `age_numeric` | +0.15 | Yaş arttıkça → ↑ Risk |
| `num_medications_changed` | +0.12 | İlaç değişikliği → ↑ Risk |

**Negatif Coefficient (Readmission Riskini Azaltır):**

| Özellik | Coefficient | Yorum |
|---------|-------------|-------|
| `time_in_hospital` | -0.22 | Uzun yatış → Daha iyi tedavi → ↓ Risk (belli bir noktaya kadar) |
| `num_lab_procedures` | -0.18 | Fazla test → Thorough evaluation → ↓ Risk |
| `has_emergency_history` | -0.08 | (Confounding etkisi olabilir) |


### 5.5 Error Analizi (Hata Analizi)

**5.5.1 Sınıflandırma Hatalarının Klinik ve Ekonomik Analizi**

Karışıklık matrisinden (confusion matrix) elde edilen sonuçlar, modelin tahmin performansının ötesinde, sağlık sistemi üzerinde doğrudan klinik ve operasyonel etkilere sahiptir. Bu hata ve başarı tiplerinin analiz edilmesi, modelin hastane iş akışlarına entegrasyonu için kritik bir değerlendirme sunmaktadır:

- **Doğru Tahminler (TP ve TN):** Modelin gerçek durumu başarıyla öngördüğü senaryolardır. True Positive (TP) sonuçları, yüksek riskli hastaların erken tespit edilerek kişiselleştirilmiş önleyici bakım (preventive care) almasını sağlar ve bu durum hem hasta sağlığı hem de hastane maliyetleri açısından en yüksek değeri yaratır. True Negative (TN) durumlarında ise düşük riskli hastalar normal takip sürecine dahil edilerek gereksiz kaynak kullanımı önlenmiş olur.

- **Kabul Edilebilir Sapmalar (False Positive - FP):** Modelin aslında düşük riskli olan bir hasta için "yüksek risk" uyarısı vermesidir. Klinik açıdan bu durum "yönetilebilir" bir hata olarak kabul edilir; zira hastaya sağlanan ilave eğitim, ilaç optimizasyonu veya ekstra takip süreçlerinin hasta üzerinde herhangi bir tıbbi zararı bulunmamakta, aksine genel bakım kalitesini artırmaktadır.

- **Kritik Hatalar (False Negative - FN):** Gerçekte yüksek risk taşıyan bir hastanın model tarafından düşük riskli olarak sınıflandırılmasıdır. Bu durum, önlenebilir bir tekrar yatışın gözden kaçmasına, hastanın hayati komplikasyonlarla karşı karşıya kalmasına ve sağlık sistemi için çok yüksek tedavi maliyetlerinin oluşmasına neden olduğu için klinik açıdan en "kritik" hata tipi olarak tanımlanmaktadır.

**5.5.2 Model Error Karşılaştırması**

| Model | TN | FP | FN | TP | FPR | FNR |
|-------|----|----|----|----|-----|-----|
| **Logistic Regression** | 11,684 | 6,399 | 983 | 1,288 | 0.354 | 0.433 |
| **Random Forest** | 15,614 | 2,469 | 1,624 | 647 | 0.137 | 0.715 |
| **XGBoost** | 17,936 | 147 | 2,169 | 102 | 0.008 | 0.955 |
| **LightGBM** | 17,863 | 220 | 2,119 | 152 | 0.012 | 0.933 |
| **SVM** | 12,513 | 5,570 | 1,162 | 1,109 | 0.308 | 0.512 |

<div style="
    border-left: 5px solid #6A1B9A;
    background-color: #E1BEE7;
    padding: 12px 14px;
    margin: 12px 0;
    border-radius: 4px;
">
<strong>📝 Önemli Not</strong><br>
    
**FPR (False Positive Rate):** FP / (FP + TN) - Ne kadar false alarm?
    
**FNR (False Negative Rate):** FN / (FN + TP) - Ne kadar critical case kaçırıldı?
</div>

![Error Analysis](docs/model_evaluation/06_error_analysis.png)
*Şekil 5.5: Model hata tiplerinin karşılaştırmalı analizi*

**5.5.3 Klinik Model Seçimi ve Uygulama Stratejisi**

Yapılan karşılaştırmalı analizler sonucunda, klinik ortamda en verimli çalışma potansiyeline sahip model olarak Lojistik Regresyon öne çıkmaktadır. Bu seçimin temel gerekçeleri şunlardır:

- **Düşük Yanlış Negatif Oranı (FNR: 0.433):** Kritik hastaları gözden kaçırma oranı en düşük modeldir.

- **Yönetilebilir Yanlış Alarm Oranı (FPR: 0.354):** Klinik iş akışını bozmayacak seviyede bir hatalı alarm dengesi sunar.

- **Yüksek Güvenilirlik ve Şeffaflık:** Tahmin olasılıklarının kalibrasyonu yüksektir ve katsayılar (coefficients) hekimler tarafından doğrudan yorumlanabilir özelliktedir.

Alternatif olarak, karar eşiği (threshold) optimize edilmiş LightGBM modeli, duyarlılığı (recall) artırmak adına ikinci bir seçenek olarak değerlendirilmektedir; ancak bu durumun hatalı alarm oranını yükselteceği göz önünde bulundurulmalıdır.

### 5.6 Model Fikir Birliği (Agreement) Analizi
`Analiz Kavramı ve İstikrar Model` fikir birliği analizi, farklı algoritmaların aynı hasta özelinde ne derece tutarlı tahminler ürettiğini ölçmektedir. Farklı matematiksel temellere sahip modellerin bir vaka üzerinde fikir birliğine varması "güçlü sinyal" olarak kabul edilirken, fikir ayrılıkları model belirsizliğine ve sınır vakalara işaret etmektedir.

`Konsensüs İstatistikleri ve Risk Grupları` için yapılan analiz sonucunda hastalar üç temel güven kategorisine ayrılmıştır:

- **Yüksek Güvenli Risk Grubu (336 Örnek):** Modellerin %80’inden fazlasının "tekrar yatış" öngördüğü, en yüksek riskli ve acil müdahale gerektiren gruptur.

- **Yüksek Güvenli Stabil Grup (14.892 Örnek):** Modellerin büyük çoğunluğunun risk görmediği, standart taburculuk protokollerinin yeterli olduğu gruptur.

- **Belirsiz Sınır Vakalar (~5.000 Örnek):** Algoritmaların farklı kararlar ürettiği, makine öğrenmesinin sınırda kaldığı ve mutlaka uzman hekim görüşünün (`clinical judgment`) belirleyici olması gereken vakalardır.

![Model Agreement](docs/model_evaluation/07_model_agreement.png)
*Şekil 5.6: Modellerin tahmin fikir birliği analizi*

## 6. HİPERPARAMETRE OPTİMİZASYONU

Model performansını daha da artırmak amacıyla, en iyi performans gösteren tree-based modeller (Random Forest, XGBoost, LightGBM) için hiperparametre optimizasyonu yapılmıştır. Bu bölümde kullanılan yöntemler, optimal parametreler ve iyileştirme sonuçları detaylandırılmıştır.

### 6.1 Hiperparametre Optimizasyon Stratejileri

**6.1.1 GridSearchCV (Sistematik Arama)**

**Çalışma Prensibi:**
- Belirtilen tüm parametre kombinasyonlarını sistematik olarak dener
- Her kombinasyon için cross-validation ile performans değerlendirmesi yapar
- Garantili olarak en iyi kombinasyonu bulur (verilen grid içinde)

**Kullanım:** XGBoost için seçildi (daha focused parameter grid ile)

**6.1.2 RandomizedSearchCV (Rastgele Arama)**

**Çalışma Prensibi:**
- Parameter space'ten rastgele örnekler seçer
- Belirlenen iterasyon sayısı kadar deneme yapar
- İstatistiksel olarak optimal sonuca yakın bulur

**Kullanım:** Random Forest ve LightGBM için seçildi

### 6.2 Optimizasyon Metrikleri

**Primary Metric: F1-Score**
- Precision ve recall'un harmonik ortalaması
- Dengesiz veri setlerinde daha anlamlı
- Klinik açıdan hem FP hem FN'yi dengeler

**Secondary Metric: ROC-AUC**
- Discriminative power göstergesi
- Threshold-independent değerlendirme
- Model'in genel ayırt etme yeteneği

**Cross-Validation: 5-Fold Stratified CV** (Random Forest, LightGBM)
**Cross-Validation: 3-Fold Stratified CV** (XGBoost - hız için)

### 6.3 Model-Specific Optimizasyon Sonuçları

**6.3.1 Random Forest Optimization**

**Arama Stratejisi:** RandomizedSearchCV

**Iterasyon Sayısı:** 50 kombinasyon × 5 folds = 250 fit

**Optimizasyon Süresi:** 683.32 saniye (~11.4 dakika)

**Parameter Search Space:**
```python
{
    'n_estimators': randint(50, 300),          # 50-300 arası ağaç
    'max_depth': [5, 10, 15, 20, 25, None],   # Ağaç derinliği
    'min_samples_split': randint(2, 20),      # Split için min örnek
    'min_samples_leaf': randint(1, 10),       # Leaf için min örnek
    'max_features': ['sqrt', 'log2', None],   # Feature subsampling
    'class_weight': ['balanced', 'balanced_subsample'],
    'criterion': ['gini', 'entropy']          # Split kriteri
}
```

**En İyi Hiperparametreler:**

| Parametre | Optimal Değer | Açıklama |
|-----------|---------------|----------|
| `n_estimators` | 207 | Ensemble'da 207 ağaç |
| `max_depth` | None | Tam derinlik (no pruning) |
| `min_samples_split` | 13 | Split için minimum 13 örnek |
| `min_samples_leaf` | 2 | Leaf'te minimum 2 örnek |
| `max_features` | 'sqrt' | √n özellik her split'te |
| `class_weight` | 'balanced_subsample' | Bootstrap'ta class balance |
| `criterion` | 'gini' | Gini impurity kullan |

**Performans:**
- **CV F1-Score:** 0.8587 (resampled data üzerinde)
- **Test F1-Score:** 0.1141
- **Test ROC-AUC:** 0.6589

**6.3.2 XGBoost Optimizasyon**

**Arama Stratejisi:** GridSearchCV

**Iterasyon Sayısı:** 6,912 kombinasyon × 3 folds = 20,736 fit

**Optimizasyon Süresi:** 3566.29 saniye (~59.4 dakika)

**Parameter Grid:**
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5]
}
```

**En İyi Hiperparametreler:**

| Parametre | Optimal Değer | Açıklama |
|-----------|---------------|----------|
| `n_estimators` | 200 | 200 boosting round |
| `max_depth` | 9 | Ağaç derinliği 9 |
| `learning_rate` | 0.1 | Orta hızda öğrenme |
| `subsample` | 0.9 | %90 row sampling |
| `colsample_bytree` | 0.7 | %70 feature sampling |
| `gamma` | 0.1 | Minimum loss reduction |
| `min_child_weight` | 1 | Leaf için min weight |

**Performans:**
- **CV F1-Score:** 0.8576
- **Test F1-Score:** 0.0960
- **Test ROC-AUC:** 0.6672

**6.3.3 LightGBM Optimizasyonu**

**Arama Stratejisi:** RandomizedSearchCV

**Iterasyon Sayısı:** 50 kombinasyon × 5 folds = 250 fit

**Optimizasyon Süresi:** 403.75 saniye (~6.7 dakika)

**Parametre Dağılımı:**
```python
{
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),       # Continuous
    'num_leaves': randint(20, 150),
    'subsample': uniform(0.6, 0.4),            # 0.6-1.0
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_samples': randint(10, 50),
    'reg_alpha': uniform(0, 1),                # L1 regularization
    'reg_lambda': uniform(0, 1)                # L2 regularization
}
```

**En İyi Hiperparametreler:**

| Parametre | Optimal Değer | Açıklama |
|-----------|---------------|----------|
| `n_estimators` | 285 | 285 boosting iteration |
| `max_depth` | 14 | Ağaç derinliği 14 |
| `learning_rate` | 0.0650 | Düşük learning rate |
| `num_leaves` | 108 | Leaf sayısı 108 |
| `subsample` | 0.6558 | %65.6 row sampling |
| `colsample_bytree` | 0.6727 | %67.3 feature sampling |
| `min_child_samples` | 31 | Leaf için min 31 örnek |
| `reg_alpha` | 0.2912 | L1 regularization |
| `reg_lambda` | 0.6119 | L2 regularization |

**Performans:**
- **CV F1-Score:** 0.8584
- **Test F1-Score:** 0.1362
- **Test ROC-AUC:** 0.6809

### 6.4 Orijinal - Tuned Performans Karşılaştırması

**6.4.1 Test Set Performans Tablosu**

| Model | Config | Test F1 | Test ROC-AUC | Training Time |
|-------|--------|---------|--------------|---------------|
| **Random Forest** | Original | 0.2402 | 0.6507 | 5.85s |
| **Random Forest** | Tuned | 0.1141 | 0.6589 | 683.32s |
| **XGBoost** | Original | 0.0810 | 0.6674 | 2.72s |
| **XGBoost** | Tuned | 0.0960 | 0.6672 | 3566.29s |
| **LightGBM** | Original | 0.1150 | 0.6743 | 6.47s |
| **LightGBM** | Tuned | 0.1362 | 0.6809 | 403.75s |

![Tuning Comparison](docs/hyperparameter_tuning/01_tuning_comparison.png)
*Şekil 6.4.1: Original ve tuned modellerin test performansı karşılaştırması*

**6.4.2 İyileşme Analizi**

| Model | F1-Score İyileşmesi | ROC-AUC İyileşmesi |
|-------|--------------------|--------------------|
| **Random Forest** | -52.50% ❌ | +1.25% ✅ |
| **XGBoost** | +18.64% ✅ | -0.04% ≈ |
| **LightGBM** | +18.38% ✅ | +0.98% ✅ |

**Ortalama İyileşme:**
- **F1-Score:** -5.16% (karışık sonuç)
- **ROC-AUC:** +0.73% (hafif iyileşme)

![Improvement Percentage](docs/hyperparameter_tuning/02_improvement_percentage.png)
*Şekil 6.4.2: Model iyileşme yüzdeleri (pozitif değerler iyileşmeyi gösterir)*


## 6.5 Hiperparametre Optimizasyonu Bulguları ve Teknik Değerlendirme

**6.5.1 Beklenmedik Performans Çıktıları ve Analizi**

Hiperparametre optimizasyonu süreci, her modelde doğrusal bir artış sağlamamış; özellikle Random Forest modelinde F1 skorunda %52.5 oranında belirgin bir düşüş gözlemlenmiştir. Bu durumun temel nedeni, optimize edilen modelin SMOTE ile dengelenmiş (resampled) eğitim verisine aşırı uyum (overfitting) sağlamasıdır. Özellikle max_depth parametresinin sınırlandırılmaması, ağaçların sentetik verideki gürültüleri öğrenmesine yol açmış ve test setindeki orijinal dengesiz dağılımda genelleme yeteneğini (generalization) zayıflatmıştır. Buna karşın, XGBoost ve LightGBM modellerinde elde edilen yaklaşık %18’lik performans artışı, boosting algoritmalarının parametre hassasiyetine rağmen veri setindeki sınıf dengesizliğinin hala dominant bir kısıt olduğunu göstermektedir.

**6.5.2 Eğitim ve Test Performansı Arasındaki Sapma (Gap)**

Çalışma sonucunda, çapraz doğrulama (CV) skorları ile Test skorları arasında yaklaşık 0.72 birimlik bir F1 skoru farkı tespit edilmiştir. Bu performans farkı üç temel etkene dayanmaktadır:

- Veri Dağılım Farklılığı: Modellerin, SMOTE ile dengelenmiş eğitim verisi üzerinde optimize edilmesine rağmen, test setinin orijinal dengesiz yapıda korunması performansın test aşamasında düşmesine neden olmuştur.

- Sentetik Örnekleme Etkisi: Modellerin gerçek klinik vakalar yerine SMOTE tarafından üretilen sentetik örüntülere fit olması, gerçek dünya verilerindeki değişkenliği tam olarak karşılayamamıştır.

- Metrik Hassasiyeti: F1 skoru sınıf dengesizliğine karşı oldukça hassas bir metrik olduğundan, ROC-AUC skorlarındaki sapmanın (0.91'den 0.68'e) daha düşük kalması modelin ayırt etme gücünün hala korunduğunu kanıtlamaktadır.

**6.5.3 Öne Çıkan Hiperparametre Çıkarımları**

Optimizasyon süreci, model başarısını etkileyen kritik parametreler hakkında şu teknik içgörüleri sağlamıştır:

- Öğrenme Oranı (Learning Rate): LightGBM için 0.065 ve XGBoost için 0.1 değerlerinin optimal bulunması, düşük öğrenme oranlarının daha stabil bir yakınsama sağladığını göstermiştir.

- Ağaç Derinliği: Boosting modellerinin daha derin ağaçlarla (9-14) daha başarılı sonuçlar vermesine karşın, Random Forest modelinde derinliğin sınırsız bırakılması doğrudan aşırı öğrenmeye yol açmıştır.

- Düzenlileştirme (Regularization): LightGBM modelinde kullanılan L1 ve L2 regülarizasyon teknikleri, karmaşık modellerde overfitting riskini minimize eden en önemli unsurlar olarak öne çıkmıştır.

- Örnekleme Oranları: %30-40 aralığında yapılan özellik ve örneklem alt kümelemeleri (subsampling), modelin farklı veri varyasyonlarına karşı direncini artırmıştır.


### 6.6 Hesaplama Maliyeti ve Performans Dengesi (Trade-off)
Model optimizasyon süreçleri, harcanan zaman ve elde edilen performans kazanımı açısından değerlendirildiğinde LightGBM, yaklaşık 6.7 dakikalık optimizasyon süresi ve test setindeki en yüksek iyileşme oranıyla en verimli algoritma olarak belirlenmiştir. XGBoost modeli, 20.736 iterasyon ve yaklaşık 1 saatlik işlem süresine rağmen LightGBM ile benzer bir iyileşme sergilemiş; bu durum GridSearch yerine RandomSearch kullanımının daha rasyonel olacağını kanıtlamıştır. Random Forest ise harcanan hesaplama maliyetine rağmen negatif bir geri dönüş vererek parametre seçim stratejisinin yeniden gözden geçirilmesi gerektiğini ortaya koymuştur.

### 6.7 Genel Değerlendirme ve İyileştirme Önerileri
Yapılan çalışma sonucunda sistematik yaklaşım ve hesaplama verimliliği açısından önemli başarılar elde edilse de, dengesiz veri setlerinde gerçek dünya performansının artırılması için şu stratejiler önerilmektedir:

- Gelişmiş Doğrulama Yöntemleri: Gelecek çalışmalarda, hiperparametre tuning işleminin gerçek veri dağılımını daha iyi yansıtması için Nested Cross-Validation yönteminin kullanılması tavsiye edilir.

- Maliyet Odaklı Yaklaşım: Hatalı negatif (False Negative) tahminlerin klinik maliyetinin yüksekliği göz önüne alınarak, optimizasyonun standart F1 skoru yerine maliyet duyarlı (cost-sensitive) özel fonksiyonlar üzerinden yapılması performans artışı sağlayabilir.

- Üretim Ortamı (Production) Stratejisi: Mevcut modeller arasında Tuned LightGBM modelinin, karar eşik değerinin (threshold) 0.25-0.30 bandına çekilmesiyle klinik takip süreçlerinde operasyonel olarak kullanılabileceği değerlendirilmektedir.

## 7. MODEL YORUMLANABİLİRLİĞİ

Makine öğrenmesi modellerinin klinik ortamda kullanılabilmesi için "black-box" olmaktan çıkarılıp yorumlanabilir hale getirilmesi kritik öneme sahiptir. Bu bölümde SHAP (SHapley Additive exPlanations), permutation importance ve partial dependence analizi ile modellerin karar mekanizmaları açıklanmıştır.

### 7.1 Klinik Ortamda Yorumlanabilirliğin Stratejik Önemi

Makine öğrenmesi modellerinin klinik karar destek sistemlerinde yer bulabilmesi için, bu modellerin "kapalı kutu" (black-box) olmaktan çıkarılarak şeffaf ve açıklanabilir hale getirilmesi temel bir gerekliliktir. Sağlık profesyonellerinin bir yapay zeka modeline güven duyabilmesi, modelin yalnızca yüksek doğrulukla tahmin yapmasına değil, aynı zamanda bu tahminlerin arkasındaki tıbbi gerekçeleri sunabilmesine bağlıdır. Bu bağlamda yorumlanabilirlik; FDA ve GDPR gibi yasal düzenlemelerin zorunlu kıldığı etik standartların karşılanması, modelin öğrendiği örüntülerin alan uzmanlığı (domain expertise) ile doğrulanması ve olası hatalı tahminlerin kök nedenlerinin analiz edilmesi açısından kritik bir rol oynamaktadır.

### 7.2 SHAP Analizi Metodolojisi ve Uygulaması
Modellerin karar mekanizmalarını açıklamak amacıyla, oyun teorisinden türetilen ve her bir özniteliğin tahmine olan marjinal katkısını hesaplayan SHAP (SHapley Additive exPlanations) yöntemi kullanılmıştır. Çalışma kapsamında TreeExplainer ve KernelExplainer yaklaşımları aracılığıyla XGBoost, LightGBM ve Random Forest modellerinin iç dinamikleri incelenmiştir. SHAP analizi, modellerin tahminleme sürecinde hangi klinik faktörlere ne kadar ağırlık verdiğini yerel ve küresel düzeyde kesin verilerle ortaya koymaktadır.

LightGBM modeli üzerinden yapılan analizler, taburcu edilme durumunu temsil eden discharge_disposition_id değişkeninin en baskın risk faktörü olduğunu göstermektedir. Özellikle hastaların eve taburcu edilmek yerine özel bakım merkezlerine veya rehabilitasyon ünitelerine sevk edilmesinin, tekrar yatış riskini istatistiksel olarak anlamlı düzeyde artırdığı saptanmıştır. Benzer şekilde, hastanın geçmişteki hastane yatış sayısı (number_inpatient) ve hastanede kalış süresince uygulanan tıbbi müdahalelerin yoğunluğu (procedure_intensity), hastanın genel sağlık durumunun ciddiyetini yansıtan güçlü birer risk göstergesi olarak öne çıkmaktadır. Önceki dönemlerde acil servis başvuru geçmişi bulunan ve çok sayıda tanı (comorbidities) konulan hastaların yüksek risk grubunda yer alması, modelin klinik gerçeklerle uyumlu örüntüler öğrendiğini doğrulamaktadır.

### 7.3 Özellik Önem Metotlarının Karşılaştırmalı Analizi
Çalışmada sunulan bulguların tutarlılığını test etmek amacıyla; modelin kendi iç hesaplamaları (Native Importance), SHAP değerleri ve öznitelik yer değiştirme (Permutation Importance) yöntemleri karşılaştırılmıştır. Yapılan konsensüs analizi sonucunda, number_inpatient ve discharge_disposition_id değişkenlerinin her üç yöntemde de en yüksek önem derecesine sahip olduğu görülmüştür. Bu durum, söz konusu değişkenlerin diyabetik hastaların tekrar yatış riskini öngörmede en güvenilir parametreler olduğunu kanıtlamaktadır. Yaş ve hastanede kalış süresi gibi değişkenlerde ise yöntemler arasında gözlemlenen farklılıklar, bu özelliklerin diğer klinik verilerle olan karmaşık etkileşimlerine işaret etmektedir. Sonuç olarak, bu çok katmanlı yorumlanabilirlik analizi, modelin kararlarını rasyonel bir tıbbi çerçeveye oturtarak klinik entegrasyon için gerekli olan şeffaf altyapıyı sağlamaktadır.

**LightGBM SHAP Analysis:**

**Top 10 En Önemli Özellikler (SHAP Importance):**


| Sıra | Özellik Adı | Ortalama SHAP Değeri | Klinik Yorumlama ve Risk Değerlendirmesi |
| :--- | :--- | :---: | :--- |
| 1 | `discharge_disposition_id` | 0.0312 | Taburcu edilen yer (ev, rehabilitasyon merkezi vb.), tekrar yatış riskini belirleyen en temel faktördür. |
| 2 | `number_inpatient` | 0.0258 | Geçmiş hastane yatış sayısı, hastanın sağlık hizmeti kullanım sıklığını ve kronik risk düzeyini yansıtan güçlü bir tahmin edicidir. |
| 3 | `procedure_intensity` | 0.0238 | Günlük prosedür yoğunluğu, vaka karmaşıklığı ve tıbbi müdahale gereksiniminin bir göstergesidir. |
| 4 | `num_procedures` | 0.0157 | Yatış süresince yapılan toplam prosedür sayısı, modelin karar mekanizmasında yüksek öneme sahiptir. |
| 5 | `has_emergency_history` | 0.0124 | Yakın dönemdeki acil servis başvuru geçmişi, hastalığın stabilizasyon seviyesi hakkında kritik bilgi sunar. |
| 6 | `time_in_hospital` | 0.0086 | Hastanede kalış süresi, uygulanan tedavi derinliği ile tekrar yatış olasılığı arasında doğrudan bir ilişki kurar. |
| 7 | `number_diagnoses` | 0.0079 | Toplam teşhis sayısı, hastanın komorbidite (eşlik eden hastalık) yükünü ve tıbbi zorluğunu ifade eder. |
| 8 | `age_numeric` | 0.0060 | İleri yaş aralıkları, fizyolojik rezervin azalmasıyla bağlantılı olarak risk artışına neden olan bir unsurdur. |
| 9 | `insulin` | 0.0060 | İnsülin kullanımı ve dozaj stabilitesi, diyabet yönetimindeki kontrol ve şiddet seviyesini simgeler. |
| 10 | `diag_1_freq` | 0.0050 | Birincil teşhisün görülme sıklığı, belirli hastalık türlerinin risk üzerindeki istatistiksel ağırlığını ölçer. |


**SHAP Effects Yorumlama:**

Modelin karar mekanizması üzerindeki öznitelik etkileri incelendiğinde, taburcu edilme durumu (`Discharge Disposition`) en belirgin belirleyicilerden biri olarak öne çıkmaktadır; yüksek ID değerine sahip özel bakım merkezlerine sevk edilen hastaların pozitif SHAP değerleri sergileyerek daha yüksek risk taşıdığı, eve taburcu edilenlerin ise daha düşük risk grubunda yer aldığı saptanmıştır. Geçmiş yatarak tedavi sayıları (`Number Inpatient`) analiz edildiğinde, ikiden fazla yatış geçmişi olan hastaların riskinin belirgin şekilde arttığı görülmekte olup bu durum sağlık hizmeti kullanım geçmişinin güçlü bir tahmin edici olduğunu kanıtlamaktadır. Ayrıca, prosedür yoğunluğundaki (`Procedure Intensity`) artışın pozitif SHAP değerleriyle risk artışına yol açması, yoğun tıbbi müdahale gereksiniminin vaka karmaşıklığı ve tekrar yatış olasılığı için kritik bir klinik işaret olduğunu ortaya koymaktadır.

### 7.3 Öznitelik Önem Düzeylerinin Karşılaştırmalı Analizi

Üç farklı importance metodu karşılaştırılmıştır:
Çalışma kapsamında modellerin karar verme süreçlerini anlamlandırmak amacıyla üç farklı önem belirleme metodolojisi karşılaştırılmıştır. İlk olarak kullanılan Modele Özgü Önem (`Native Importance`) yöntemi; Random Forest için Gini saflığı, XGBoost ve LightGBM için ise kazanç tabanlı hesaplamalara dayanmaktadır. Bu yöntemin en büyük avantajı hesaplama hızının yüksek olmasıdır; ancak yüksek kardinaliteye sahip değişkenlere karşı yanlılık gösterme riski bir dezavantaj olarak değerlendirilmektedir. İkinci olarak uygulanan SHAP Önem Analizi, her bir özelliğin ortalama mutlak Shapley değerlerini temel alarak teorik açıdan çok daha sağlam ve tarafsız bir değerlendirme sunmaktadır. SHAP yöntemi öznitelik etkilerini en doğru şekilde yansıtsa da, yüksek hesaplama maliyeti operasyonel bir kısıt oluşturmaktadır.

Son olarak kullanılan Permütasyon Önemi (`Permutation Importance`) yöntemi ise, belirli bir özelliğin değerleri rastgele karıştırıldığında (`shuffle`) model performansında meydana gelen düşüşü ölçerek özniteliğin gerçek etkisini saptamaktadır. Modelden bağımsız çalışabilmesi ve gerçek dünya etkisini doğrudan ölçebilmesi bu yöntemin güçlü yönüyken; birbiriyle yüksek korelasyona sahip özellikler bulunduğunda sonuçların güvenilirliğinin azalması temel limitasyonu olarak görülmektedir. Bu üç yöntemin birlikte kullanılması, klinik değişkenlerin tahmin gücü üzerinde çok boyutlu ve doğrulanmış bir bakış açısı elde edilmesini sağlamıştır.

Top 10 özelliklerin 3 metottaki sıralaması:

| Özellik | Modele Özgü | SHAP Değeri | Permütasyon | Fikir Birliği |
|---------|--------|------|-------------|-----------|
| `number_inpatient` | 1 | 2 | 1 | ✅ Güçlü |
| `discharge_disposition_id` | 4 | 1 | 4 | ✅ Güçlü |
| `age_numeric` | 2 | 8 | 13 | ⚠️ Karma |
| `time_in_hospital` | 3 | 6 | 23 | ❌ Zayıf |
| `procedure_intensity` | 11 | 3 | 30 | ⚠️ Karma |


## 8. SONUÇ VE TARTIŞMA

Bu proje, diabetik hastaların 30 gün içinde hastaneye tekrar yatış riskini tahmin etmek için kapsamlı bir makine öğrenmesi çalışması gerçekleştirmiştir. Veri keşfinden model deployment önerisine kadar tüm data science pipeline uygulanmış ve detaylı dokümante edilmiştir.

### 8.1 Proje Amaçlarının Değerlendirilmesi

| # | Amaç | Başarı Durumu | Açıklama |
|---|------|---------------|----------|
| 1 | Kapsamlı veri analizi | ✅ Tamamlandı | 101,766 hasta, 50 özellik detaylı analiz edildi |
| 2 | Veri ön işleme ve feature engineering | ✅ Tamamlandı | 8 yeni özellik, encoding, scaling uygulandı |
| 3 | 3+ farklı ML modeli eğitimi | ✅ Tamamlandı | 5 model eğitildi (LR, RF, XGB, LGBM, SVM) |
| 4 | Model performans değerlendirme | ✅ Tamamlandı | Comprehensive metrics, ROC, PR curves |
| 5 | Hiperparametre optimizasyonu | ✅ Tamamlandı | GridSearch ve RandomSearch uygulandı |
| 6 | Model yorumlanabilirliği | ✅ Tamamlandı | SHAP, feature importance, PDP analizi |

**Sonuç:** Tüm proje amaçları başarıyla tamamlanmıştır. ✅

### 8.2 Temel Bulgular ve Sonuçlar

**En İyi Modeller (Test Set):**

| Metrik | En İyi Model | Değer | Yorum |
|--------|--------------|-------|-------|
| **Recall (En Kritik)** | Logistic Regression | 0.567 | En az kritik hasta kaçırıyor |
| **F1-Score** | Logistic Regression | 0.259 | Dengeli performans |
| **ROC-AUC** | LightGBM | 0.674 | En iyi discrimination power |
| **Precision** | XGBoost | 0.410 | En az false alarm |
| **Kalibrasyon** | Logistic Regression | Mükemmel | Güvenilir probabilities |

Modellerin karşılaştırmalı analizi sonucunda, Lojistik Regresyon en yüksek duyarlılık (0.567 recall) oranını sunarak kritik hastaları tespit etme konusunda en başarılı model olarak öne çıkmış; yorumlanabilir katsayıları ve yüksek kalibrasyon kalitesiyle klinik kullanıma en uygun seçenek olarak değerlendirilmiştir. Buna karşın, LightGBM hiperparametre optimizasyonu ile F1 skorunda %18’lik bir artış yakalayarak en yüksek ROC-AUC (0.681) değerine ulaşsa da, zayıf kalibrasyonu ve düşük duyarlılık oranı nedeniyle operasyonel kısıtlar sergilemektedir. XGBoost ve Random Forest yüksek doğruluk oranlarına rağmen kritik hastaların %95'inden fazlasını kaçırırken, SVM modeli hesaplama maliyetinin yüksekliği nedeniyle pratik uygulamalar için verimli bulunmamıştır.

Sınıf dengesizliğiyle mücadele kapsamında uygulanan SMOTE ve rastgele alt örnekleme yöntemleri, çapraz doğrulama performansını kağıt üzerinde artırsa da, test setinde beklenen iyileşmeyi sağlamamış ve sentetik örneklerin gerçek dünya verileri üzerindeki tahmin performansını yanıltabileceğini (misleading) ortaya koymuştur. Hiperparametre optimizasyonu tarafında LightGBM ve XGBoost modelleri performanslarını %18 civarında artırırken, Random Forest modeli aşırı öğrenme (overfitting) nedeniyle ciddi bir performans kaybı yaşamıştır. Hesaplama maliyeti açısından LightGBM 6.7 dakikalık süresiyle en verimli model olurken, XGBoost 59 dakikalık süresiyle en maliyetli yöntem olmuştur. Gelecekteki iyileştirmeler için maliyet-duyarlı öğrenme (cost-sensitive learning) ve karar eşiği optimizasyonu gibi daha agresif stratejilerin kullanılması önerilmektedir.

### 8.3 Klinik ve Pratik Çıkarımlar ve Uygulama Önerileri

Yapılan analizler sonucunda, diyabetik hastaların tekrar yatış riskini belirleyen en güçlü klinik göstergelerin hastane kullanım geçmişi (acil servis ve yatarak tedavi sayıları) ile taburcu edildikleri merkez türü olduğu saptanmıştır. Özellikle 70 yaş üzeri popülasyon, yüksek prosedür yoğunluğu ve çoklu tanıya (komorbidite) sahip hastalar en riskli grubu oluşturmaktadır; bu durum, klinik süreçlerde yaşa özel protokollerin ve koordineli bakım programlarının önemini ortaya koymaktadır. Model değerlendirme aşamasında, %57’lik duyarlılık (recall) oranı, yüksek kalibrasyon başarısı ve şeffaf yorumlanabilirlik özellikleri nedeniyle Lojistik Regresyon modelinin (eşik değeri 0.35-0.40) klinik kullanım için en uygun çözüm olduğu belirlenmiştir. Bu modelin karar destek sistemlerine entegre edilmesi, yüksek riskli hastaların taburculuk öncesinde gerçek zamanlı olarak tespit edilmesine ve kişiselleştirilmiş müdahale planlarıyla tekrar yatış oranlarının minimize edilmesine olanak sağlayacaktır.

**Geliştirme Süreci:**

```
1. Hastanın Taburculuk Sürecinin Başlatılması
   ↓
2. Klinik Verilerin ve Özniteliklerin (Features) Toplanması
   ↓
3. Tekrar Yatış Olasılığının (Readmission Probability) Hesaplanması
   ↓
4. Risk Tabakalandırma (Risk Stratification):
   - P > 0.40: YÜKSEK RİSK → Yoğun Takip ve Müdahale Programı
   - 0.20 < P < 0.40: ORTA RİSK → Standart İzlem ve Taburculuk Sonrası Kontrol
   - P < 0.20: DÜŞÜK RİSK → Rutin Bakım ve Standart Protokol
   ↓
5. Risk Faktörlerinin Görselleştirilmesi (SHAP Açıklamaları)
   ↓
6. Klinik Uzman Değerlendirmesi ve Nihai Karar
```

**Beklenen Çıktılar:**

Geliştirilen modelin beklenen çıktıları şu şekilde özetlenebilir: Model, yaklaşık %57'lik duyarlılık (recall) oranıyla kritik durumdaki hastaların yarıdan fazlasını önceden tespit edebilme kapasitesine sahiptir. %65'lik özgüllük (specificity) oranı sayesinde düşük riskli hastalar başarıyla ayırt edilirken, %35 seviyesindeki hatalı alarm oranı klinik açıdan kabul edilebilir bir eşik olarak değerlendirilmektedir; zira düşük riskli bir hastaya sağlanan ilave önleyici bakımın herhangi bir tıbbi zararı bulunmamaktadır.

**8.3.3 Operasyonel Faydalar**

Geliştirilen tahmin modelinin sağlık sistemi üzerindeki operasyonel yansımaları; maliyet yönetimi, hizmet kalitesi ve bakım koordinasyonu olmak üzere üç temel eksende stratejik avantajlar sunmaktadır. Modelin 30 günlük tekrar yatışları önleme kabiliyeti, hastanelerin Medicare cezalarından kaçınmasını sağlamanın yanı sıra kaynak optimizasyonu yoluyla ciddi bir maliyet tasarrufu potansiyeli yaratmaktadır. Hizmet kalitesi açısından ise daha etkin taburculuk planlaması ve hedefli klinik müdahaleler, hasta memnuniyetinde doğrudan bir artış sağlamaktadır. Bakım koordinasyonu sürecinde yüksek riskli hastaların erken tespiti, taburcu sonrası kişiselleştirilmiş takip programlarının ve multidisipliner bakım ekiplerinin devreye alınmasına olanak tanıyarak sistemin operasyonel verimliliğini artırmaktadır. Ekonomik açıdan somut bir projeksiyon yapıldığında; hasta başına yaklaşık 15.000 dolarlık tekrar yatış maliyeti üzerinden, modelin yıllık 1.000 yüksek riskli hastada yatışları sadece %10 oranında önlemesi bile kurum için yıllık yaklaşık 1,5 milyon dolarlık bir yatırım getirisi (ROI) ve doğrudan tasarruf anlamına gelmektedir.

### 8.4 Projenin Güçlü Yönleri

Bu çalışma, diyabetik hastaların hastaneye tekrar yatış risklerini tahmin etme sürecinde uçtan uca kurgulanmış kapsamlı bir veri bilimi boru hattı (pipeline) sunması bakımından oldukça güçlü bir yapıya sahiptir. 130 hastaneden toplanan ve 10 yıllık bir süreci kapsayan 101.000'den fazla gerçek dünya verisiyle çalışılmış olması, elde edilen sonuçların istatistiksel güvenilirliğini artırmaktadır. Metodolojik olarak sadece tek bir algoritma ile yetinilmemiş; Lojistik Regresyon’dan modern topluluk öğrenme (ensemble) yöntemlerine kadar 5 farklı algoritma titizlikle karşılaştırılarak en uygun çözüm aranmıştır. Projenin teknik derinliği ise yalnızca model eğitimiyle sınırlı kalmayıp; SHAP analizi ile model kararlarının yorumlanabilir kılınması, eşik değeri (threshold) optimizasyonu ve model kalibrasyonu gibi ileri seviye tekniklerin başarıyla uygulanmasıyla pekiştirilmiştir. Ayrıca, klinik alan bilgisinin özellik mühendisliği süreçlerine entegre edilmesi ve projenin her aşamasının tekrarlanabilirliği (reproducibility) sağlayacak şekilde detaylıca dokümante edilmesi, projenin hem akademik hem de pratik değerini en üst seviyeye taşımaktadır.

### 8.5 Projenin Zayıf Yönleri ve Limitasyonlar

Bu çalışma, diyabetik hastaların tekrar yatış risklerinin tahmininde önemli çıktılar sunsa da, sonuçların yorumlanması aşamasında dikkate alınması gereken birtakım veri ve model kaynaklı kısıtlar bulunmaktadır. Veri seti düzeyindeki en belirgin kısıt, analize dahil edilen kayıtların 1999-2008 yıllarını kapsamasıdır; bu durum, son on beş yılda değişen sağlık hizmeti uygulamalarının ve modern tedavi protokollerinin modele tam olarak yansıtılamamasına neden olmaktadır. Ayrıca, veri setinde kilo (%96+) ve tıbbi uzmanlık alanı (%49+) gibi kritik değişkenlerdeki yüksek eksiklik oranları potansiyel bilgi kaybına yol açmış; laboratuvar sonuçlarının sayısal değerleri, yaşamsal bulgular ve sağlığın sosyal belirleyicileri gibi parametrelerin eksikliği ise modelin bütüncül (holistic) bir risk değerlendirmesi yapma kabiliyetini sınırlamıştır.

Modelleme performansı açısından bakıldığında, test setinde elde edilen 0.65-0.68 aralığındaki ROC-AUC değerleri, modellerin ayırt edici gücünün orta seviyede olduğunu göstermektedir. Veri setindeki %11,16’lık azınlık sınıf oranı nedeniyle yaşanan belirgin sınıf dengesizliği, "Recall" ve "Precision" metrikleri arasında zorunlu bir ödünleşime (trade-off) yol açmıştır. Özellikle çapraz doğrulama aşamasındaki yüksek F1 skorları ile test setindeki düşük sonuçlar arasındaki fark, modellerin sentetik veriye aşırı uyum sağladığını ve üretim ortamındaki performansın laboratuvar sonuçlarından daha düşük olabileceğini kanıtlamaktadır.

Son olarak, projenin genellenebilirliği ve uygulama süreci önünde genel limitasyonlar mevcuttur. Verilerin yalnızca ABD hastanelerinden toplanmış olması, farklı sağlık sistemleri için modelin dışsal geçerliliğini (external validity) belirsiz kılmaktadır. On yıllık tarihsel bir kesite dayanılması ve ileriye dönük (prospective) bir validasyonun henüz yapılmamış olması, modelin gelecekteki performansına dair belirsizlikler yaratmaktadır. Ayrıca, modelin klinik ortamlarda hayata geçirilmesi için gerekli olan elektronik sağlık kayıtları (EHR) entegrasyonu ve sağlık personeli eğitimleri gibi operasyonel gereklilikler, adaptasyon sürecinde karşılaşılabilecek temel engeller olarak değerlendirilmektedir.

### 8.6 Gelecek Çalışmalar için Öneriler

Bu çalışma ile elde edilen bulgular, diyabetik hastaların tekrar yatış risklerinin tahmininde önemli bir temel oluşturmakla birlikte, modelin başarısını ve klinik yararlılığını artırmak adına gelecekte atılabilecek stratejik adımlar bulunmaktadır. Veri kalitesini iyileştirmek amacıyla, 2020 yılı ve sonrasına ait güncel tıbbi verilerin toplanarak modern sağlık uygulamalarının modele yansıtılması ilk öncelik olmalıdır. Buna ek olarak, laboratuvar sonuçları (HbA1c, glukoz seviyeleri), hayati bulgular ve sosyal belirleyiciler gibi daha kapsamlı özniteliklerin dahil edilmesi, hastaların klinik tablosunun daha bütüncül bir şekilde analiz edilmesine imkân tanıyacaktır. Teknik açıdan ise derin öğrenme mimarilerinin (LSTM, Transformers) kullanılması, sağkalım analizi yaklaşımları ve hatalı negatif tahminlerin maliyetini minimize eden "maliyet-duyarlı öğrenme" (cost-sensitive learning) yöntemleri, tahmin hassasiyetini üst seviyeye taşıyabilecek potansiyel geliştirme alanlarıdır.

Geliştirilen modellerin pratik hayattaki etkinliğini doğrulamak için gerçek zamanlı pilot uygulamalar (prospective studies) ve farklı hastaneleri kapsayan çok merkezli validasyon çalışmalarının yürütülmesi büyük önem arz etmektedir. Modelin klinik karar destek sistemlerine (CDSS) entegre edilmesi sürecinde, FHIR standartları kullanılarak elektronik sağlık kayıtları (EHR) ile tam uyumlu ve otomatik skorlama yapabilen bir yapı kurgulanmalıdır. Son olarak, model performansının zaman içindeki değişimini (data drift) takip eden sürekli izleme mekanizmalarının kurulması ve dönemsel yeniden eğitim (retraining) süreçlerinin işletilmesi, sistemin uzun vadeli güvenilirliğini ve sürdürülebilirliğini garanti altına alacaktır.


```python

```
