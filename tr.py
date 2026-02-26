import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1️⃣ LOAD DATASET
# ============================================================

DATA_PATH = r"D:\PROJECT_FINALYEAR_1\data\Air ViewClear Skies Hourly Dataset.xlsx"

df = pd.read_excel(DATA_PATH)

df = df.rename(columns={
    'PM2_5 (ug/m3)': 'PM2_5',
    'PM10 (ug/m3)': 'PM10',
    'AT (degC)': 'AT',
    'RH (%)': 'RH',
    'CO2 (PPM)': 'CO2',
    'local_time (IST)': 'local_time'
})

df['local_time'] = pd.to_datetime(df['local_time'])

# Remove rows where PM2.5 missing
df = df.dropna(subset=['PM2_5'])

# Fill remaining missing values using dataset statistics
df['PM10'] = df.groupby('station_name')['PM10'].transform(lambda x: x.fillna(x.mean()))
df['AT'] = df['AT'].fillna(df['AT'].mean())
df['RH'] = df['RH'].fillna(df['RH'].mean())
df['CO2'] = df['CO2'].fillna(df['CO2'].mean())

print("Dataset shape after cleaning:", df.shape)

# COMPUTE AQI 

def calculate_subindex(conc, breakpoints):
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= conc <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (conc - bp_lo) + i_lo
    return 500

pm25_bp = [
    (0,30,0,50),(31,60,51,100),(61,90,101,200),
    (91,120,201,300),(121,250,301,400),(251,500,401,500)
]

pm10_bp = [
    (0,50,0,50),(51,100,51,100),(101,250,101,200),
    (251,350,201,300),(351,430,301,400),(431,600,401,500)
]

def compute_aqi(row):
    aqi_pm25 = calculate_subindex(row['PM2_5'], pm25_bp)
    aqi_pm10 = calculate_subindex(row['PM10'], pm10_bp)
    return max(aqi_pm25, aqi_pm10)

df['AQI'] = df.apply(compute_aqi, axis=1)

print("AQI generated.")



df['hour'] = df['local_time'].dt.hour
df['day_of_week'] = df['local_time'].dt.dayofweek
df['month'] = df['local_time'].dt.month

def get_season(month):
    if month in [12,1,2]:
        return 0
    elif month in [3,4,5,6]:
        return 1
    elif month in [7,8,9]:
        return 2
    else:
        return 3

df['season'] = df['month'].apply(get_season)

print("Time & season features created.")


#  DEFINE FEATURES & TARGET

features = [
    'PM2_5',      # Dominant
    'PM10',       # Dominant
    'AT',         # Weather
    'RH',         # Weather
    'CO2',        # Environmental influence
    'hour',
    'day_of_week',
    'month',
    'season'
]

X = df[features]
y = df['AQI']

print("Number of features:", len(features))

#  TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  TRAIN MODEL
model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed.")


#  EVALUATE

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", round(r2,4))
print("RMSE:", round(rmse,2))


# SAVE MODEL
os.makedirs("models", exist_ok=True)

with open("models/ai_aqi_model_2.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/features_2.pkl", "wb") as f:
    pickle.dump(features, f)

print("Model saved successfully.")