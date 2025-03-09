from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib 

app = FastAPI()


model = joblib.load('clf.joblib')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:

    input_data = np.array([[item.year, item.km_driven, item.fuel, item.seller_type,
                             item.transmission, item.owner, item.mileage,
                             item.engine, item.max_power, item.torque, item.seats]])
    
    prediction = model.predict(input_data)
    
    return float(prediction[0])
