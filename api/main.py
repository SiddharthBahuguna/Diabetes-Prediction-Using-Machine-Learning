from fastapi import FastAPI, HTTPException, Query
import joblib, os
from typing import Union

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "../model/model_joblib_diabetes")
model = joblib.load(model_path)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/api/predict")
async def predict(
    Pregnancies: int = Query(...),
    Glucose: Union[int, float] = Query(...),
    BloodPressure: Union[int, float] = Query(...),
    SkinThickness: Union[int, float] = Query(...),
    Insulin: Union[int, float] = Query(...),
    BMI: Union[int, float] = Query(...),
    DiabetesPedigreeFunction: Union[int, float] = Query(...),
    Age: int = Query(...)
):
    try:
        values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        prediction = model.predict([values])
        predicted = "Non-Diabetic" if prediction[0] == 0 else "Diabetic"
        
        return {"prediction": predicted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
