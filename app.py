from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Drug Prediction API")

# Load model and scaler
MODEL_PATH = "drug_decission_tree.pkl"
SCALER_PATH = "drug_scaler.pkl"

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print("Model or Scaler file not found!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

class PatientData(BaseModel):
    Age: int
    Sex: str  # 'F' or 'M'
    BP: str   # 'HIGH', 'LOW', 'NORMAL'
    Cholesterol: str # 'HIGH', 'NORMAL'
    Na_to_K: float

DRUG_MAPPING = {
    0: "drugA",
    1: "drugB",
    2: "drugC",
    3: "drugX",
    4: "drugY"
}

SEX_MAPPING = {"F": 0, "M": 1}
BP_MAPPING = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
CHOL_MAPPING = {"HIGH": 0, "NORMAL": 1}

@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Preprocess categorical features
        sex = SEX_MAPPING.get(data.Sex.upper())
        bp = BP_MAPPING.get(data.BP.upper())
        chol = CHOL_MAPPING.get(data.Cholesterol.upper())
        
        if sex is None or bp is None or chol is None:
            raise HTTPException(status_code=400, detail="Invalid categorical values")
        
        # Preprocess numerical features
        # Scaler is for Na_to_K only (as found in research)
        na_to_k_scaled = scaler.transform([[data.Na_to_K]])[0][0]
        
        # Prepare feature array
        features = np.array([[data.Age, sex, bp, chol, na_to_k_scaled]])
        
        # Predict
        prediction_idx = int(model.predict(features)[0])
        prediction_drug = DRUG_MAPPING.get(prediction_idx, "Unknown")
        
        return {
            "prediction_idx": prediction_idx,
            "prediction_drug": prediction_drug
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Drug Prediction API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
