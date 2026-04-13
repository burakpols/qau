"""Test günlük pipeline"""
from src.pipeline import pipeline

# Initialize
print("Initializing...")
pipeline.initialize()

# Train models
print("\nTraining models...")
metrics = pipeline.train_models(days=365)
print('Training Metrics:', metrics)

# Run daily prediction
print("\nRunning daily prediction...")
result = pipeline.run_daily()

print('\n=== GUNLUK SONUC ===')
print('Sinyal:', result["signal"]["signal"])
print('Guven:', result["signal"]["confidence"])
print('Mevcut Fiyat:', result["current_price"])

preds = result.get("predictions", {})
if "ensemble_prediction" in preds:
    print('Ensemble Tahmin:', preds["ensemble_prediction"])

analysis = result.get("analysis", "Rapor yok")
if isinstance(analysis, str):
    print('\n--- Analiz Raporu ---')
    print(analysis[:500])
elif isinstance(analysis, dict):
    print('\n--- Analiz Raporu ---')
    print(analysis.get("summary", "ozet yok")[:500])
