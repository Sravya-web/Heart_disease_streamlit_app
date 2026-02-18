import joblib
import os
from train import model, X   # or however you create model & features

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "columns": X.columns.tolist()
    },
    "models/heart_disease_model.pkl"
)

print("✅ Model saved successfully!")