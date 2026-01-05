import os
import joblib
from typing import Tuple

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(file)), "model", "model.pkl")
_model = None

def load_model():
global _model
if _model is None:
if not os.path.exists(MODEL_PATH):
raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
_model = joblib.load(MODEL_PATH)
return _model

def predict_single(features):
model = load_model()
proba = model.predict_proba([features])[0]
label = int(proba[1] >= 0.5)
return label, float(proba[1])
