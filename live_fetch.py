# Fetches live environmental data per station.
# Webservice is used:
#   1. Open-Meteo Air Quality
#   2. Open-Meteo Weather

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

AQI_URL  = "https://air-quality-api.open-meteo.com/v1/air-quality"
WTHR_URL = "https://api.open-meteo.com/v1/forecast"
SESSION  = requests.Session()   # reuse TCP connections

# In-memory cache to store the last successful reading per station
_last_successful_fetch = {}

def fetch_live_readings(lat: float, lon: float) -> dict | None:
    """
    Fetch PM2.5, PM10, Temperature, Humidity for one lat/lon.
    Coordinates must come from the dataset station metadata.
    Fires both API calls IN PARALLEL (2 workers).
    """
    params = {"latitude": lat, "longitude": lon}

    def get_aqi():
        aqi_params = {
            **params, 
            "current": "pm10,pm2_5,carbon_dioxide", 
            "cell_selection": "nearest"
        }
        r = SESSION.get(AQI_URL, params=aqi_params, timeout=8)
        r.raise_for_status()
        return r.json().get("current", {})

    def get_wthr():
        r = SESSION.get(WTHR_URL, params={**params, "current": "temperature_2m,relative_humidity_2m"}, timeout=8)
        r.raise_for_status()
        return r.json().get("current", {})

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_aqi  = ex.submit(get_aqi)
            f_wthr = ex.submit(get_wthr)
            aqi_data  = f_aqi.result()
            wthr_data = f_wthr.result()

        # Extract base values
        base_pm25 = float(aqi_data.get("pm2_5", 0))
        base_pm10 = float(aqi_data.get("pm10", 0))
        base_co2  = float(aqi_data.get("carbon_dioxide", 400))
        base_temp = float(wthr_data.get("temperature_2m", 0))
        base_hum  = float(wthr_data.get("relative_humidity_2m", 0))

        # Return direct raw values from the Open-Meteo API without alteration
        return {
            "pm25":     round(base_pm25, 2),
            "pm10":     round(base_pm10, 2),
            "co2":      round(base_co2, 2),
            "temp":     round(base_temp, 2),
            "humidity": round(base_hum, 2),
        }
    except requests.exceptions.Timeout:
        print(f"⚠  Open-Meteo timeout (lat={lat:.4f}, lon={lon:.4f})")
        return None
    except Exception as e:
        print(f"⚠  Open-Meteo error: {e}")
        return None


def fetch_historical_readings(lat: float, lon: float, past_days: int) -> dict | None:
    """
    Fetch historical PM2.5, PM10, CO2, Temp, Humidity for one lat/lon.
    Uses 'past_days' parameter in Open-Meteo API.
    Returns: dict {"time": [...], "pm25": [...], "pm10": [...], ...}
    """
    params = {"latitude": lat, "longitude": lon, "past_days": past_days, "timezone": "auto"}

    def get_aqi():
        aqi_params = {
            **params, 
            "hourly": "pm10,pm2_5,carbon_dioxide", 
            "cell_selection": "nearest"
        }
        r = SESSION.get(AQI_URL, params=aqi_params, timeout=12)
        r.raise_for_status()
        return r.json().get("hourly", {})

    def get_wthr():
        r = SESSION.get(WTHR_URL, params={**params, "hourly": "temperature_2m,relative_humidity_2m"}, timeout=12)
        r.raise_for_status()
        return r.json().get("hourly", {})

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_aqi  = ex.submit(get_aqi)
            f_wthr = ex.submit(get_wthr)
            aqi_data  = f_aqi.result()
            wthr_data = f_wthr.result()

        return {
            "time": aqi_data.get("time", []),
            "pm25": aqi_data.get("pm2_5", []),
            "pm10": aqi_data.get("pm10", []),
            "co2": aqi_data.get("carbon_dioxide", []),
            "temp": wthr_data.get("temperature_2m", []),
            "humidity": wthr_data.get("relative_humidity_2m", [])
        }
    except Exception as e:
        print(f"⚠  Historical fetch error: {e}")
        return None


def fetch_batch(station_list: list[dict], max_workers: int = 16) -> dict:
    """
    Fetch live readings for MANY stations simultaneously.
    Returns: dict mapping station name → readings dict (or None on fail)
    """
    results = {}

    def fetch_one(st):
        return st["name"], fetch_live_readings(st["lat"], st["lon"])

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_one, s): s["name"] for s in station_list}
        for f in as_completed(futures):
            name, live_data = f.result()
            
            if live_data is not None:
                # 1. API SUCCESS: Save the new live data into our fallback memory
                _last_successful_fetch[name] = live_data
                results[name] = live_data
            else:
                # 2. API FAIL: Try to use the last known successful reading from memory
                if name in _last_successful_fetch:
                    print(f"⚠  API failed for {name}. Using last successful live reading.")
                    results[name] = _last_successful_fetch[name]
                else:
                    # 3. ABSOLUTE FAIL: API is down and we have no history (e.g. app just booted)
                    print(f"❌ API failed for {name} and no history exists.")
                    results[name] = None

    return results

if __name__ == "__main__":
    import time
    import pandas as pd

    DATA_PATH = r"D:\PROJECT_FINALYEAR_1\data\Air ViewClear Skies Hourly Dataset.xlsx"
    COLS = ["station_name", "latitude", "longitude", "local_time (IST)"]

    print("Loading station coordinates from dataset...")
    df = pd.read_excel(DATA_PATH, usecols=COLS)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"local_time (IST)": "local_time"})
    df["local_time"] = pd.to_datetime(df["local_time"])

    latest = (df.sort_values("local_time")
                .groupby("station_name")[["latitude","longitude"]]
                .last().reset_index())

    test_stations = [
        {"name": row["station_name"],
         "lat":  float(row["latitude"]),
         "lon":  float(row["longitude"])}
        for _, row in latest.head(3).iterrows()
    ]

    print(f"\nFetching live data for {len(test_stations)} dataset stations (parallel)...")
    for s in test_stations:
        print(f"  {s['name']}: lat={s['lat']:.4f}, lon={s['lon']:.4f}")

    t0  = time.time()
    res = fetch_batch(test_stations)
    elapsed = time.time() - t0

    print(f"\nResults ({elapsed:.1f}s elapsed):")
    for name, data in res.items():
        if data:
            print(f"  {name}: PM2.5={data['pm25']}  PM10={data['pm10']}  "
                  f"Temp={data['temp']}°C  Hum={data['humidity']}%  CO2={data['co2']}ppm")
        else:
            print(f"  {name}: ❌ Fetch failed (using dataset fallback)")
    print("\n✅ Done — all coordinates sourced from dataset")
