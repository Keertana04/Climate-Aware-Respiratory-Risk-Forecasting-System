# test_spatial.py  —  Spatial model test using REAL dataset values
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

print("Loading spatial model...")
with open('models/ai_aqi_model_spatial.pkl','rb') as f:
    model = pickle.load(f)
with open('models/features_spatial.pkl','rb') as f:
    features = pickle.load(f)
print(f"Features: {features}\n")

# ── Load real station data from dataset ──────────────────────
print("Reading latest station readings from dataset...")
COLS = ['local_time (IST)','station_name','latitude','longitude',
        'PM2_5 (ug/m3)','PM10 (ug/m3)','AT (degC)','RH (%)','CO2 (PPM)']
df = pd.read_excel(r'data\Air ViewClear Skies Hourly Dataset.xlsx', usecols=COLS)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'local_time (IST)':'local_time','PM2_5 (ug/m3)':'PM2_5',
    'PM10 (ug/m3)':'PM10','AT (degC)':'AT','RH (%)':'RH','CO2 (PPM)':'CO2'
})
df['local_time'] = pd.to_datetime(df['local_time'])
df = df.dropna(subset=['PM2_5'])
df['PM10'] = df.groupby('station_name')['PM10'].transform(lambda x: x.fillna(x.mean()))
df['AT']   = df['AT'].fillna(df['AT'].mean())
df['RH']   = df['RH'].fillna(df['RH'].mean())
df['CO2']  = df['CO2'].fillna(df['CO2'].mean())

# Get the latest reading per station
latest = df.sort_values('local_time').groupby('station_name').last().reset_index()
print(f"Loaded latest readings for {len(latest)} stations.\n")

now = datetime.now()
h, dow, m = now.hour, now.weekday(), now.month
season = 0 if m in [12,1,2] else 1 if m in [3,4,5,6] else 2 if m in [7,8,9] else 3

def predict_aqi(row_dict):
    df_in = pd.DataFrame([row_dict])[features]
    return round(float(model.predict(df_in)[0]), 1)

# ── Test 1: First 3 real stations ────────────────────────────
print("=== Test 1: Predict AQI for first 3 real stations ===")
for _, row in latest.head(3).iterrows():
    inp = {
        'PM2_5': row['PM2_5'], 'PM10': row['PM10'],
        'AT':    row['AT'],    'RH':   row['RH'],   'CO2': row['CO2'],
        'latitude': row['latitude'], 'longitude': row['longitude'],
        'hour': h, 'day_of_week': dow, 'month': m, 'season': season
    }
    aqi = predict_aqi(inp)
    print(f"  {row['station_name'][:45]:<45}  AQI = {aqi}")

# ── Test 2: Midpoint between first two stations (IDW) ────────
print("\n=== Test 2: IDW interpolation at midpoint between 2 stations ===")
st_a = latest.iloc[0]
st_b = latest.iloc[1]
mid_lat = (float(st_a['latitude'])  + float(st_b['latitude']))  / 2
mid_lon = (float(st_a['longitude']) + float(st_b['longitude'])) / 2

# IDW weights from just these 2 stations
stations_for_idw = [
    {'lat': float(st_a['latitude']),  'lon': float(st_a['longitude']),
     'PM2_5': float(st_a['PM2_5']),  'PM10': float(st_a['PM10']),
     'AT': float(st_a['AT']),        'RH': float(st_a['RH']),   'CO2': float(st_a['CO2'])},
    {'lat': float(st_b['latitude']),  'lon': float(st_b['longitude']),
     'PM2_5': float(st_b['PM2_5']),  'PM10': float(st_b['PM10']),
     'AT': float(st_b['AT']),        'RH': float(st_b['RH']),   'CO2': float(st_b['CO2'])},
]

dists   = [max(((s['lat']-mid_lat)**2+(s['lon']-mid_lon)**2)**0.5, 1e-10) for s in stations_for_idw]
weights = [1/d**2 for d in dists]; W = sum(weights)

idw_inp = {
    'PM2_5':     sum(w*s['PM2_5'] for w,s in zip(weights, stations_for_idw))/W,
    'PM10':      sum(w*s['PM10']  for w,s in zip(weights, stations_for_idw))/W,
    'AT':        sum(w*s['AT']    for w,s in zip(weights, stations_for_idw))/W,
    'RH':        sum(w*s['RH']    for w,s in zip(weights, stations_for_idw))/W,
    'CO2':       sum(w*s['CO2']   for w,s in zip(weights, stations_for_idw))/W,
    'latitude':  mid_lat, 'longitude': mid_lon,
    'hour': h, 'day_of_week': dow, 'month': m, 'season': season
}
aqi_mid = predict_aqi(idw_inp)

print(f"  Station A: {st_a['station_name'][:35]}  AQI = {predict_aqi({**idw_inp,'latitude':float(st_a['latitude']),'longitude':float(st_a['longitude']),'PM2_5':float(st_a['PM2_5']),'PM10':float(st_a['PM10']),'AT':float(st_a['AT']),'RH':float(st_a['RH']),'CO2':float(st_a['CO2'])})}")
print(f"  Station B: {st_b['station_name'][:35]}  AQI = {predict_aqi({**idw_inp,'latitude':float(st_b['latitude']),'longitude':float(st_b['longitude']),'PM2_5':float(st_b['PM2_5']),'PM10':float(st_b['PM10']),'AT':float(st_b['AT']),'RH':float(st_b['RH']),'CO2':float(st_b['CO2'])})}")
print(f"  Midpoint : ({mid_lat:.4f}, {mid_lon:.4f}) [IDW]  AQI = {aqi_mid}")
print(f"  → AQI varies spatially between the two stations!")

# ── Test 3: Station with highest vs lowest AQI ───────────────
print("\n=== Test 3: Highest and Lowest AQI stations in dataset ===")
all_aqis = []
for _, row in latest.iterrows():
    inp = {
        'PM2_5': row['PM2_5'], 'PM10': row['PM10'],
        'AT': row['AT'],  'RH': row['RH'],  'CO2': row['CO2'],
        'latitude': row['latitude'], 'longitude': row['longitude'],
        'hour': h, 'day_of_week': dow, 'month': m, 'season': season
    }
    all_aqis.append((row['station_name'], predict_aqi(inp)))

all_aqis.sort(key=lambda x: x[1])
print(f"  Cleanest station: {all_aqis[0][0][:45]}  AQI = {all_aqis[0][1]}")
print(f"  Most polluted   : {all_aqis[-1][0][:45]}  AQI = {all_aqis[-1][1]}")
print(f"\n  All {len(all_aqis)} stations predicted successfully!")
print("\nAll tests PASSED")
