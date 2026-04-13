"""Model training test"""
from src.pipeline import pipeline
import json

# Initialize
print('=== Initializing pipeline ===')
df = pipeline.initialize()
print(f'Features: {len(df)} rows')

# Train models
print('\n=== Training models ===')
metrics = pipeline.train_models(days=365)
print(f'Metrics: {json.dumps(metrics, indent=2)}')

# Status
status = pipeline.get_status()
print(f'\nStatus: {json.dumps(status["models_trained"], indent=2)}')
print('\n=== Test complete ===')