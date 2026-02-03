import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"C:\Users\ozmir\Downloads\archive\cardekho_data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

df.drop(['Car_Name'], axis=1, inplace=True)

#(Feature Engineering)
df['Vehicle_Age'] = 2026 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

model = RandomForestRegressor(n_estimators=50, random_state=15)
model.fit(X, y)

saved_data = {
    "model": model,
    "encoders": encoders,
    "columns": X.columns.tolist()
}

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(saved_data, f)

print("Created successfully!")