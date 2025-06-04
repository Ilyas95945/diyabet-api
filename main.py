from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Eğitilmiş modeli yükle
model = joblib.load("model.pkl")

# Giriş verileri sınıfı
class InputData(BaseModel):
    AGE: int
    HbA1c: float
    BMI: float
    Gender: int
    Chol: float
    TG: float
    HDL: float
    LDL: float
    Cr: float
    Urea: float

# Tahmin endpoint'i
@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.AGE, data.HbA1c, data.BMI, data.Gender,
        data.Chol, data.TG, data.HDL, data.LDL,
        data.Cr, data.Urea
    ]]
    
    probabilities = model.predict_proba(input_data)[0]
    predicted_class = int(probabilities.argmax())
    confidence = float(round(probabilities.max() * 100, 2))

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }
