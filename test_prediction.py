from src.pipeline import QAUPipeline

# Initialize
pipeline = QAUPipeline()
pipeline.initialize()

# Train models
print("\n=== Training Models ===")
metrics = pipeline.train_models()

# Run daily prediction
print("\n=== Daily Prediction ===")
result = pipeline.run_daily()

# Print results
print("\n" + "="*50)
print("TAHMIN SONUCU")
print("="*50)
print(f"Mevcut Fiyat: {result['current_price']:,.2f} TL")
if 'ensemble_prediction' in result['predictions']:
    print(f"Ensemble Tahmin: {result['predictions']['ensemble_prediction']:,.2f} TL")
if 'xgboost_prediction' in result['predictions']:
    print(f"XGBoost Tahmin: {result['predictions']['xgboost_prediction']:,.2f} TL")
if 'arima_prediction' in result['predictions']:
    print(f"ARIMA Tahmin: {result['predictions']['arima_prediction']:,.2f} TL")

print(f"\nSignal: {result['signal']['signal']}")
print(f"Güven: {result['signal']['confidence']:.0f}%")
print(f"Yön: {result['signal'].get('direction', 'N/A')}")

if result.get('analysis'):
    print(f"\nAnaliz:\n{result['analysis'][:500]}...")