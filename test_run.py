from src.pipeline import pipeline

# Initialize
pipeline.initialize()

# Train
metrics = pipeline.train_models(days=365)
print("Training OK")
print(f"XGBoost Direction Accuracy: {metrics['xgboost']['direction_accuracy']:.1%}")

# Run daily
result = pipeline.run_daily()
print(f"\nSignal: {result['signal']['signal']}")
print(f"Direction: {result['predictions'].get('xgboost_direction')}")
print(f"Confidence: {result['predictions'].get('xgboost_confidence', 0):.0%}")
print(f"Current Price: {result['current_price']:,.2f}")
print(f"\nAnalysis Preview:\n{result['analysis'][:300]}...")