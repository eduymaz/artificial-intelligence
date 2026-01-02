# HİPERPARAMETRE OPTİMİZASYONU RAPORU

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

**Arama Stratejisi**: RandomizedSearchCV
**Toplam İterasyon**: 250
**Optimizasyon Süresi**: 683.32 saniye

**En İyi Hiperparametreler:**
```python
{
  "class_weight": "balanced_subsample",
  "criterion": "gini",
  "max_depth": null,
  "max_features": "sqrt",
  "min_samples_leaf": 2,
  "min_samples_split": 13,
  "n_estimators": 207
}
```

**Performans:**
- CV F1-Score: 0.8587
- Test F1-Score: 0.1141
- Test ROC-AUC: 0.6589

### 2.2 XGBoost

**Arama Stratejisi**: GridSearchCV
**Toplam İterasyon**: 20736
**Optimizasyon Süresi**: 3566.29 saniye

**En İyi Hiperparametreler:**
```python
{
  "colsample_bytree": 0.7,
  "gamma": 0.1,
  "learning_rate": 0.1,
  "max_depth": 9,
  "min_child_weight": 1,
  "n_estimators": 200,
  "subsample": 0.9
}
```

**Performans:**
- CV F1-Score: 0.8576
- Test F1-Score: 0.0960
- Test ROC-AUC: 0.6672

### 2.3 LightGBM

**Arama Stratejisi**: RandomizedSearchCV
**Toplam İterasyon**: 250
**Optimizasyon Süresi**: 403.75 saniye

**En İyi Hiperparametreler:**
```python
{
  "colsample_bytree": 0.6727299868828402,
  "learning_rate": 0.06502135295603015,
  "max_depth": 14,
  "min_child_samples": 31,
  "n_estimators": 285,
  "num_leaves": 108,
  "reg_alpha": 0.2912291401980419,
  "reg_lambda": 0.6118528947223795,
  "subsample": 0.6557975442608167
}
```

**Performans:**
- CV F1-Score: 0.8584
- Test F1-Score: 0.1362
- Test ROC-AUC: 0.6809

## 3. Original vs Tuned Karşılaştırma

![Tuning Comparison](hyperparameter_tuning/01_tuning_comparison.png)

### Performans Karşılaştırması

| Model | Original F1 | Tuned F1 | Original ROC-AUC | Tuned ROC-AUC |
|-------|-------------|----------|------------------|---------------|
| Random Forest | 0.2402 | 0.1141 | 0.6507 | 0.6589 |
| XGBoost | 0.0810 | 0.0960 | 0.6674 | 0.6672 |
| LightGBM | 0.1150 | 0.1362 | 0.6743 | 0.6809 |

## 4. İyileşme Analizi

![Improvement Percentage](hyperparameter_tuning/02_improvement_percentage.png)

### İyileşme Yüzdeleri

| Model | F1-Score İyileşmesi | ROC-AUC İyileşmesi |
|-------|--------------------|--------------------|
| Random Forest | -52.50% | +1.25% |
| XGBoost | +18.64% | -0.04% |
| LightGBM | +18.38% | +0.98% |

**Ortalama İyileşme:**
- F1-Score: -5.16%
- ROC-AUC: +0.73%

## 5. Hiperparametre Etkileri

### Key Parametreler ve Etkileri

**Learning Rate (XGBoost, LightGBM):**
- Düşük değer (0.01-0.05): Daha yavaş öğrenme, daha iyi generalization, overfitting riski az
- Yüksek değer (0.1-0.3): Hızlı öğrenme, overfitting riski artar
- **Optimal**: 0.1 (XGBoost)

**Max Depth:**
- Ağacın maksimum derinliği
- Düşük: Underfitting riski
- Yüksek: Overfitting riski
- **Optimal**: None (Random Forest)

**N Estimators:**
- Ağaç sayısı (ensemble size)
- Daha fazla ağaç → genelde daha iyi performans
- Diminishing returns after certain point
- **Optimal**: 285 (LightGBM)

**Subsample / Colsample:**
- Training data'nın / feature'ların ne kadarı kullanılacak
- <1.0: Regularization effect, overfitting önler
- **Optimal Subsample**: 0.9 (XGBoost)

## 6. Computational Efficiency

### Tuning Süreleri

| Model | Tuning Süresi | İterasyon Sayısı | Saniye/İterasyon |
|-------|---------------|------------------|------------------|
| Random Forest | 683.32s | 250 | 2.73s |
| XGBoost | 3566.29s | 20736 | 0.17s |
| LightGBM | 403.75s | 250 | 1.61s |

**Gözlemler:**
- RandomizedSearchCV genelde GridSearchCV'den daha hızlı
- LightGBM en hızlı training süresine sahip
- XGBoost GridSearch en uzun sürdü (comprehensive search)

## 7. En İyi Model Seçimi

**Test F1-Score'a Göre En İyi**: **LightGBM** (0.1362)

**Test ROC-AUC'ye Göre Sıralama:**
1. LightGBM: 0.6809
2. XGBoost: 0.6672
3. Random Forest: 0.6589

## 8. Önemli Çıkarımlar

### 8.1 Hiperparametre Tuning Değerli mi?


**HAYIR** - Bu durumda default parametreler yeterli

Ortalama F1-Score İyileşmesi: **-5.16%**
Ortalama ROC-AUC İyileşmesi: **+0.73%**

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

1. **Model Selection**: LightGBM kullan (en iyi test performance)

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
*Rapor otomatik olarak oluşturulmuştur - 2025-12-14 00:06:57*
