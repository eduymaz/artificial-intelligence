"""
=================================================================================
COMPLETE END-TO-END PIPELINE
=================================================================================
Amaç: Tüm machine learning pipeline adımlarını tek bir scriptte birleştirmek

Pipeline Adımları:
1. Data Exploration
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Hyperparameter Tuning
6. Model Interpretation

Yazar: Machine Learning Final Project
Tarih: Aralık 2024
=================================================================================
"""

import subprocess
import sys
from pathlib import Path
import time

# Proje root
PROJECT_ROOT = Path(__file__).parent.parent

print("=" * 80)
print("COMPLETE END-TO-END MACHINE LEARNING PIPELINE")
print("=" * 80)
print("\nDiabetic Hospital Readmission Prediction Project")
print("=" * 80)

# =================================================================================
# PIPELINE SCRIPTS
# =================================================================================

scripts = [
    ("01_data_exploration.py", "Veri Keşfi ve İlk Analiz"),
    ("02_data_preprocessing.py", "Veri Ön İşleme ve Özellik Mühendisliği"),
    ("03_model_training.py", "Model Eğitimi ve Karşılaştırma"),
    ("04_model_evaluation.py", "Detaylı Model Değerlendirme"),
    ("05_hyperparameter_tuning.py", "Hiperparametre Optimizasyonu"),
    ("06_model_interpretation.py", "Model Yorumlanabilirliği ve SHAP Analizi")
]

total_start = time.time()
results = []

for idx, (script_name, description) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"ADIM {idx}/{len(scripts)}: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}\n")
    
    script_path = PROJECT_ROOT / 'codes' / script_name
    
    if not script_path.exists():
        print(f"❌ HATA: {script_name} bulunamadı!")
        results.append({
            'script': script_name,
            'description': description,
            'status': 'FAILED',
            'error': 'File not found',
            'time': 0
        })
        continue
    
    start_time = time.time()
    
    try:
        # Script'i çalıştır
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800  # 30 dakika max
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {description} BAŞARILI")
            print(f"⏱️  Süre: {elapsed_time:.2f} saniye")
            
            results.append({
                'script': script_name,
                'description': description,
                'status': 'SUCCESS',
                'time': elapsed_time
            })
        else:
            print(f"\n❌ {description} BAŞARISIZ")
            print(f"Hata çıktısı:")
            print(result.stderr)
            
            results.append({
                'script': script_name,
                'description': description,
                'status': 'FAILED',
                'error': result.stderr[:500],
                'time': elapsed_time
            })
            
            # Kritik adımlar başarısız olursa dur
            if idx <= 3:  # İlk 3 adım kritik
                print(f"\n⚠️  Kritik adım başarısız! Pipeline durduruluyor.")
                break
    
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} TIMEOUT")
        print(f"Script 30 dakikadan fazla sürdü, sonlandırıldı.")
        
        results.append({
            'script': script_name,
            'description': description,
            'status': 'TIMEOUT',
            'time': elapsed_time
        })
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {description} HATA")
        print(f"Hata: {str(e)}")
        
        results.append({
            'script': script_name,
            'description': description,
            'status': 'ERROR',
            'error': str(e),
            'time': elapsed_time
        })

# =================================================================================
# ÖZET RAPOR
# =================================================================================

total_time = time.time() - total_start

print(f"\n{'='*80}")
print("PIPELINE ÖZET RAPORU")
print(f"{'='*80}\n")

print(f"Toplam Süre: {total_time/60:.2f} dakika ({total_time:.2f} saniye)\n")

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
failed_count = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])

print(f"✅ Başarılı: {success_count}/{len(results)}")
print(f"❌ Başarısız: {failed_count}/{len(results)}\n")

print("Detaylı Sonuçlar:")
print("-" * 80)

for idx, result in enumerate(results, 1):
    status_emoji = "✅" if result['status'] == 'SUCCESS' else "❌"
    print(f"{idx}. {status_emoji} {result['description']}")
    print(f"   Script: {result['script']}")
    print(f"   Durum: {result['status']}")
    print(f"   Süre: {result['time']:.2f}s")
    
    if 'error' in result:
        print(f"   Hata: {result['error'][:200]}...")
    
    print()

# =================================================================================
# BAŞARI DURUMU
# =================================================================================

if success_count == len(results):
    print(f"\n{'='*80}")
    print("🎉 TÜM PIPELINE BAŞARIYLA TAMAMLANDI! 🎉")
    print(f"{'='*80}\n")
    print("Sonraki adımlar:")
    print("1. docs/ klasöründeki tüm raporları inceleyin")
    print("2. models/ klasöründeki eğitilmiş modelleri kontrol edin")
    print("3. Görselleştirmeleri gözden geçirin")
    print("4. Final proje raporu için docs/proje_raporu.md dosyasını okuyun")
else:
    print(f"\n{'='*80}")
    print("⚠️  PIPELINE TAMAMLANAMADI")
    print(f"{'='*80}\n")
    print(f"{failed_count} adım başarısız oldu.")
    print("Hata detaylarını yukarıda kontrol edin.")

print(f"\n{'='*80}")
print("Pipeline tamamlandı - " + time.strftime("%Y-%m-%d %H:%M:%S"))
print(f"{'='*80}\n")
