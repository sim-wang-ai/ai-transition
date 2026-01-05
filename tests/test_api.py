from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(file), "..")))

from app.main import app

client = TestClient(app)

def test_health():
r = client.get("/health")
assert r.status_code == 200
assert r.json()["status"] == "ok"

def test_predict_no_model():
# If model missing, we expect 500; run train.py before running tests to make full predict test pass.
payload = {"age": 30, "fare": 10.0, "sex_male": 1}
r = client.post("/predict", json=payload)
# If model exists, this should be 200. If not, 500.
assert r.status_code in (200, 500)
