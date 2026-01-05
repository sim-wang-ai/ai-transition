from pydantic import BaseModel

class PredictRequest(BaseModel):
age: float
fare: float
sex_male: int  # 1 for male, 0 for female

class PredictResponse(BaseModel):
predicted_label: int
probability: float
