from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib 
import re
from fastapi import File, UploadFile, HTTPException
import io
from fastapi.responses import Response 

app = FastAPI()

model = joblib.load('clf.joblib')

def transform_mil(row):
  if pd.isna(row['mileage']):
    return np.nan
  unit_mileage = re.sub(r'[\d.]+', '', row['mileage']).strip()
  if re.sub(r'[\d.]+', '', unit_mileage).strip() == "km/kg":
    if row['fuel'] == 'Petrol':
      return float(re.findall(r'\d+\.?\d*', row['mileage'])[0])*0.75
    elif row['fuel'] == 'LPG':
      return float(re.findall(r'\d+\.?\d*', row['mileage'])[0])*0.54
    elif row['fuel'] == 'CNG':
      return float(re.findall(r'\d+\.?\d*', row['mileage'])[0])*0.54
    return float(re.findall(r'\d+\.?\d*', row['mileage'])[0])*0.13
  return float(re.findall(r'\d+\.?\d*', row['mileage'])[0])


def transform_engine(x):
  if pd.isna(x):
    return np.nan
  return int(re.findall(r'\d+\.?\d*', str(x))[0])


def transform_max_power(row):
  if pd.isna(row):
    return np.nan
  try:
    return float(re.findall(r'\d+\.?\d*', str(row))[0])
  except:
    return np.nan


def transform_torque(row):
  if pd.isna(row['torque']):
    return row

  unit_torque = np.nan
  if not (pd.isna(row['torque'])):
    unit_torque = re.sub(r'[^a-zA-Z]|at', '', row['torque']).lower()

  numbers = re.findall(r'\d+(?:[\.,]\d+)?', row['torque'].replace(",", ""))
  for i in range(len(numbers)):
    numbers[i] = float(numbers[i].replace(',', '.'))

  #получаем наши параметры
  torque = np.nan
  max_torque_rpm = np.nan

  if(len(numbers)>=1):
    torque=numbers[0]

  if len(numbers)==2:
    max_torque_rpm = numbers[1]
  elif len(numbers)==3:
    max_torque_rpm = (numbers[1] + numbers[2])/2

  row['max_torque_rpm'] = max_torque_rpm

  if unit_torque == 'kgmrpm':
    row['torque'] = torque*9.807
  elif unit_torque == 'nmrpm':
    row['torque'] = torque
  else: 
    row['torque'] = np.nan
  return row


def transform_seats(x):
  if pd.isna(x):
    return 'missing'
  return str(x)


def df_transform(df):
    df.replace(pd.NA, np.nan, inplace=True)
    df.drop('selling_price', inplace = True, axis=1)
    df['mileage'] = df.apply(transform_mil, axis=1)
    df['engine'] = df['engine'].apply(transform_engine)
    df['max_power'] = df['max_power'].apply(transform_max_power)
    df['max_torque_rpm'] = np.nan
    df = df.apply(transform_torque, axis=1)
    df['seats'] = df['seats'].apply(transform_seats)
    return df


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

    input_data = np.array([[ item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type,
                             item.transmission, item.owner, item.mileage,
                             item.engine, item.max_power, item.torque, item.seats]])
    
    df=pd.DataFrame(input_data, columns=['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
       'seats'])
    df = df_transform(df)
    prediction = model.predict(df)
    return float(prediction[0])


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    input_data = np.array([[item.name, item.year, item.selling_price, item.km_driven, item.fuel, item.seller_type,
                             item.transmission, item.owner, item.mileage,
                             item.engine, item.max_power, item.torque, item.seats] for item in items])
    
    df=pd.DataFrame(input_data, columns=['name', 'year','selling_price', 'km_driven', 'fuel','seller_type',
       'transmission', 'owner','mileage', 'engine','max_power', 'torque',
       'seats'])
    df = df_transform(df)
    predictions = model.predict(df)
    print(predictions)
    return [float(i) for i in predictions]


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> Response:
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
    df = pd.read_csv(io.BytesIO(contents))
    df = df_transform(df)
    predictions = model.predict(df)
    df_predict = pd.read_csv(io.BytesIO(contents))
    df_predict['predict_selling_price'] = predictions
    csv_string = df_predict.to_csv(index=False)
    return Response(content=csv_string, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

