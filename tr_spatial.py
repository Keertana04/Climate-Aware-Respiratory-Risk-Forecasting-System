import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# tr_spatial.py
# Retrains the AQI model with SPATIAL features (latitude, longitude).
#
# New features added vs tr.py:
#   + latitude   → station's geographic latitude
#   + longitude  → station's geographic longitude
#
# This allows the model to:
#   1. Learn location-specific pollution patterns
#   2. Predict AQI at ANY lat/lon point (not just sensor stations)
#   3. Enable continuous route AQI prediction via IDW interpolation
#
# Saves:
#   models/ai_aqi_model_spatial.pkl   ← spatial XGBoost model
#   models/features_spatial.pkl       ← feature list (with lat/lon)
# ============================================================

DATA_PATH = r"D:\PROJECT_FINALYEAR_1\data\Air ViewClear Skies Hourly Dataset.xlsx"

print("📂 Loading dataset...")
df = pd.read_excel(DATA_PATH)

df = df.rename(columns={
    'PM2_5 (ug/m3)':   'PM2_5',
    'PM10 (ug/m3)':    'PM10',
    'AT (degC)':       'AT',
    'RH (%)':          'RH',
    'CO2 (PPM)':       'CO2',
    'local_time (IST)':'local_time'
})

df['local_time'] = pd.to_datetime(df['local_time'])

# ── CLEAN ─────────────────────────────────────────────────────
df = df.dropna(subset=['PM2_5'])
df['PM10'] = df.groupby('station_name')['PM10'].transform(lambda x: x.fillna(x.mean()))
df['AT']   = df['AT'].fillna(df['AT'].mean())
df['RH']   = df['RH'].fillna(df['RH'].mean())
df['CO2']  = df['CO2'].fillna(df['CO2'].mean())

# Drop rows with missing lat/lon (needed for spatial features)
df = df.dropna(subset=['latitude', 'longitude'])

print(f"✓ Dataset shape after cleaning: {df.shape}")

# ── COMPUTE AQI (CPCB formula) ────────────────────────────────
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
    return max(
        calculate_subindex(row['PM2_5'], pm25_bp),
        calculate_subindex(row['PM10'],  pm10_bp)
    )

df['AQI'] = df.apply(compute_aqi, axis=1)
print("✓ AQI computed using CPCB formula")

# ── TIME FEATURES ─────────────────────────────────────────────
df['hour']        = df['local_time'].dt.hour
df['day_of_week'] = df['local_time'].dt.dayofweek
df['month']       = df['local_time'].dt.month

def get_season(month):
    if   month in [12,1,2]:  return 0  # Winter
    elif month in [3,4,5,6]: return 1  # Summer
    elif month in [7,8,9]:   return 2  # Monsoon
    else:                    return 3  # Post-Monsoon

df['season'] = df['month'].apply(get_season)
print("✓ Time & season features created")

# ── FEATURES ── SPATIAL VERSION (adds latitude & longitude) ───
features = [
    'PM2_5',        # Dominant pollutant
    'PM10',         # Dominant pollutant
    'AT',           # Temperature
    'RH',           # Humidity
    'CO2',          # Environmental
    'latitude',     # ← SPATIAL NEW
    'longitude',    # ← SPATIAL NEW
    'hour',
    'day_of_week',
    'month',
    'season'
]

X = df[features]
y = df['AQI']

print(f"✓ Features ({len(features)}): {features}")

# ── TRAIN / TEST SPLIT ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── TRAIN ─────────────────────────────────────────────────────
print("🧠 Training spatial XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
print("✓ Model training complete")

# ── EVALUATE ─────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n📊 Spatial Model Performance:")
print(f"   R² Score : {round(r2,   4)}")
print(f"   RMSE     : {round(rmse, 2)}")

# Compare with original feature importance
print(f"\n🔑 Top Feature Importances:")
importances = model.feature_importances_
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"   {feat:<15} {bar} {imp:.4f}")

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/ai_aqi_model_spatial.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/features_spatial.pkl", "wb") as f:
    pickle.dump(features, f)

print("\n✅ Spatial model saved:")
print("   models/ai_aqi_model_spatial.pkl")
print("   models/features_spatial.pkl")
