from fastapi import FastAPI, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.utils import predict_single

app = FastAPI(title="sklearn-fastapi-starter", version="0.1")

@app.get("/health")
def health():
return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
try:
features = [req.age, req.fare, req.sex_male]
label, prob = predict_single(features)
return PredictResponse(predicted_label=label, probability=prob)
except FileNotFoundError as e:
raise HTTPException(status_code=500, detail=str(e))
