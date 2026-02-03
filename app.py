from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

with open("car_price_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    model = saved_data["model"]
    encoders = saved_data["encoders"]

class CarFeatures(BaseModel):
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int
    Vehicle_Age: int

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(features: CarFeatures):
    input_data = pd.DataFrame([features.model_dump()])

    for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)

    return {"predicted_price": f"{round(float(prediction[0]), 2)} Lakh"}