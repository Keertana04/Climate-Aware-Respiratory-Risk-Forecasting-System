# import the necessary libraries

from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle, os, json, logging
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from live_fetch import fetch_live_readings, fetch_batch, fetch_historical_readings   # Open-Meteo, no key
from profile_manager import create_user, get_user, update_health

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
app.secret_key = "carrfs_v2_2024"
CORS(app)

# LOGGING
import pathlib
LOG_DIR  = pathlib.Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "carrfs.log"

_log_records = []   # in-memory ring buffer (last 200 entries) for /api/logs

class _RingHandler(logging.Handler):
    def emit(self, record):
        _log_records.append({
            "ts":  datetime.now().strftime("%H:%M:%S"),
            "lvl": record.levelname,
            "msg": self.format(record)
        })
        if len(_log_records) > 200:
            _log_records.pop(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        _RingHandler()
    ]
)
log = logging.getLogger("carrfs")
log.info("═══ CARRFS starting ═══")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "ai_aqi_model_spatial.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "features_spatial.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "data", "Air ViewClear Skies Hourly Dataset.xlsx")

with open(MODEL_PATH,  "rb") as f: model         = pickle.load(f)
with open(FEATURE_PATH,"rb") as f: feature_order = pickle.load(f)
print("✓ Spatial model loaded")

# HELPERS

SEASON_MAP = {0:"Winter", 1:"Summer", 2:"Monsoon", 3:"Post-Monsoon"}
COLOR_MAP  = {
    "Good":               "#16a34a",   # forest green
    "Satisfactory":       "#65a30d",   # olive green
    "Moderately Polluted":"#d97706",   # amber
    "Poor":               "#ea580c",   # orange
    "Very Poor":          "#dc2626",   # red
    "Severe":             "#7c3aed"    # deep violet
}

def get_season(month):
    if   month in [12,1,2]:  return 0
    elif month in [3,4,5,6]: return 1
    elif month in [7,8,9]:   return 2
    else:                    return 3

def categorize_aqi(aqi):
    if   aqi <= 50:  return "Good",               "0-50"
    elif aqi <= 100: return "Satisfactory",        "51-100"
    elif aqi <= 200: return "Moderately Polluted", "101-200"
    elif aqi <= 300: return "Poor",                "201-300"
    elif aqi <= 400: return "Very Poor",           "301-400"
    else:            return "Severe",              "401-500"

def run_model(lat, lon, pm25, pm10, temp, hum, co2, hour=None, day_of_week=None):
    now = datetime.now()
    h = hour if hour is not None else now.hour
    d = day_of_week if day_of_week is not None else now.weekday()
    row = {
        "PM2_5":pm25,"PM10":pm10,"AT":temp,"RH":hum,"CO2":co2,
        "latitude":lat,"longitude":lon,
        "hour":h,"day_of_week":d,
        "month":now.month,"season":get_season(now.month)
    }
    return round(float(model.predict(pd.DataFrame([row])[feature_order])[0]), 1)

# STATION CACHE
_station_meta      = {}   # name → {lat, lon, state, city, co2}
_station_hierarchy = {}   # state → city → [names]
_station_latest    = {}   # name → last pollutant readings (for CO2 and fallback)

def build_cache():
    global _station_meta, _station_hierarchy, _station_latest
    print("Loading station metadata...")
    COLS = ['local_time (IST)','state','city','station_name',
            'latitude','longitude',
            'PM2_5 (ug/m3)','PM10 (ug/m3)','AT (degC)','RH (%)','CO2 (PPM)']
    df = pd.read_excel(DATA_PATH, usecols=COLS)
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

    latest = df.sort_values('local_time').groupby('station_name').last().reset_index()
    for _, row in latest.iterrows():
        name = str(row['station_name'])
        s, c = str(row['state']), str(row['city'])
        lat, lon = float(row['latitude']), float(row['longitude'])
        co2 = round(float(row['CO2']), 1)

        _station_meta[name] = {"lat":lat,"lon":lon,"state":s,"city":c,"co2":co2}
        _station_latest[name] = {
            "pm25":round(float(row['PM2_5']),1),"pm10":round(float(row['PM10']),1),
            "temp":round(float(row['AT']),1),"humidity":round(float(row['RH']),1),
            "co2":co2,"lat":lat,"lon":lon,"state":s,"city":c
        }
        _station_hierarchy.setdefault(s,{}).setdefault(c,[])
        if name not in _station_hierarchy[s][c]:
            _station_hierarchy[s][c].append(name)

    print(f"✓ {len(_station_meta)} stations ready")

build_cache()

# PRECAUTIONS
def get_precautions(aqi, severity, season, age, gender, smoker, conditions):
    import random
    variety = random.choice(["lifestyle focus", "clinical focus", "preventative focus"])
    prompt = f"""You are a respiratory health expert providing highly PERSONALIZED medical advice.
User Profile: {age} year old {gender}, Smoker: {smoker}, Conditions: {', '.join(conditions) if conditions else 'None'}.
Current Air: AQI {aqi} ({severity}), Season: {SEASON_MAP.get(season,'Unknown')}.

PERSONALIZATION PROTOCOL:
1. Every single tip MUST directly address the user's profile (e.g., "As an asthma patient...", "Given your age...").
2. For smokers, emphasize how current pollution compounds with smoking risks.
3. For Elderly (>60) or Children (<12), focus on lower lung resilience and immunity.
4. Style: Informative, authoritative, but using simple words. No jargon.
5. Length: Each tip should be exactly 1-2 concise sentences.
6. Format: Return ONLY a JSON array of 6 strings."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.9,
            max_completion_tokens=600
        )
        import re
        out = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', out, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(out)
    except Exception as e:
        log.error(f"Groq LLM Precaution Error: {e}")
        return ["Stay indoors when AQI is high.",
                "Use an N95 mask if going outside.",
                "Avoid outdoor exercise during peak hours.",
                "Keep windows closed during high pollution periods.",
                "Stay hydrated.",
                "Consult a doctor if you experience respiratory symptoms."]

def get_route_precautions(avg_aqi, severity, age, conditions, smoker, worst_aqi):
    import random
    variety = random.choice(["transit focus", "recovery focus", "gear focus"])
    prompt = f"""You are a travel health expert providing PERSONALIZED transit safety advice.
User Profile: {age} years old, Smoker: {smoker}, Conditions: {', '.join(conditions) if conditions else 'None'}.
Route Stats: Avg AQI {avg_aqi} ({severity}), Worst Peak {worst_aqi}.

PERSONALIZATION PROTOCOL:
1. Reference the user's specific age or health condition in every tip.
2. Focus strictly on commute protection (N95 masks, car recirculation, peak timing).
3. Style: Extremely direct, ultra-simple, and short. Avoid complex terms. Tell the user exactly what to do using everyday English.
4. Format: Return ONLY a JSON array of 4 short strings."""
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":prompt}],
            temperature=0.9,
            max_completion_tokens=400
        )
        import re
        out = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', out, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(out)
    except Exception as e:
        log.error(f"Groq LLM Route Precaution Error: {e}")
        return ["Wear an N95 mask during travel.",
                "Keep windows closed if travelling by vehicle.",
                "Minimize physical exertion during the journey."]

@app.route("/api/route-precautions", methods=["POST"])
def api_route_precautions():
    data = request.json
    avg_aqi = data.get("avg_aqi")
    severity = data.get("severity")
    worst_aqi = data.get("worst_aqi", avg_aqi)

    age = session.get("age", "—")
    chronic_conds = session.get("chronic_conditions", [])
    temp_conds = session.get("temp_conditions", [])
    conditions = chronic_conds + temp_conds
    smoker = session.get("smoker", "No")

    log.info(f"   🤖 Groq: Generating travel precautions for route (Avg AQI: {avg_aqi})")
    precautions = get_route_precautions(avg_aqi, severity, age, conditions, smoker, worst_aqi)
    return jsonify({"precautions": precautions})

# HEALTH-AWARE ROUTE SCORING
def health_score(route, age, conditions, smoker):
    """
    Computes a personalised health risk score for a route.
    Higher score = worse for the user's health.

    Score = weighted combination of:
      - Average AQI across route stations
      - Max (peak) AQI on the route
      - Count of stations with AQI > 200 (Poor or worse)
      - PM2.5 penalty for respiratory conditions
      - Age penalty (elderly more sensitive)
      - Smoker penalty (already higher baseline sensitivity)

    Routes are then sorted ascending (lowest health risk = recommended).
    """
    stations = route["stations"]
    if not stations:
        return float("inf")

    # Guard: need valid AQI values
    valid_stations = [s for s in stations if s.get("aqi") is not None]
    if not valid_stations:
        return float("inf")

    avg_aqi  = route["avg_aqi"]
    max_aqi  = max(s["aqi"] for s in valid_stations)
    n_valid  = len(valid_stations)

    # bad_fraction: proportion of stations with AQI > 200 (normalized, not raw count)
    # Using raw bad_cnt would penalize routes that happen to have more stations,
    # which was causing wrong recommendations when routes differ in station count.
    bad_fraction = sum(1 for s in valid_stations if s["aqi"] > 200) / n_valid

    pm25_vals = [s["pm25"] for s in valid_stations if s.get("pm25") is not None]
    avg_pm25 = sum(pm25_vals) / len(pm25_vals) if pm25_vals else 0

    # Base score — avg_aqi is the PRIMARY factor. Route with lower avg AQI always wins
    # unless health conditions create a large enough secondary penalty.
    # bad_fraction adds up to 15 points max (when every station is Poor or worse).
    score = avg_aqi * 1.0 + max_aqi * 0.03 + bad_fraction * 15.0

    # Respiratory conditions → weight PM2.5 heavily
    RESP_CONDS = {"asthma", "copd", "bronchitis", "respiratory",
                  "lung", "breathing", "allergy", "allergies", "emphysema", "pneumonia"}
    conds_low  = {c.lower() for c in (conditions or [])}
    has_resp   = bool(conds_low & RESP_CONDS)
    if has_resp:
        score += avg_pm25 * 1.5

    # Heart/cardiovascular conditions → weight max AQI exposure
    HEART_CONDS = {"heart", "cardiac", "hypertension", "blood pressure",
                   "cardiovascular", "stroke"}
    has_heart = bool(conds_low & HEART_CONDS)
    if has_heart:
        score += max_aqi * 0.2

    # Age-based sensitivity
    try:
        age_val = int(str(age).strip())
        if age_val >= 65:
            score *= 1.25   # elderly: 25% more sensitive
        elif age_val >= 50:
            score *= 1.10
        elif age_val <= 12:
            score *= 1.20   # children: also sensitive
    except (ValueError, TypeError):
        pass

    # Smoker: already has reduced lung capacity
    if str(smoker).lower() in ("yes", "true", "1"):
        score *= 1.15

    return round(score, 2)

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/api/signup", methods=["POST"])
def signup():
    import re
    d = request.get_json()
    email = (d.get("email") or "").strip().lower()
    phone = re.sub(r'\D', '', (d.get("phone") or "").strip())  # digits only

    # At least one identifier required
    if not email and not phone:
        return jsonify({"error": "Email or Phone number is required"}), 400

    # Format validation
    if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({"error": "Invalid email format."}), 400
    if phone and not re.match(r'^\d{10}$', phone):
        return jsonify({"error": "Phone number must be exactly 10 digits."}), 400

    # Required profile fields
    name = (d.get("name") or "").strip()
    age  = (d.get("age") or "").strip()
    gender = (d.get("gender") or "").strip()
    smoker = (d.get("smoker") or "").strip()

    if not name:
        return jsonify({"error": "Full Name is required."}), 400
    try:
        age_val = int(age)
        if not (1 <= age_val <= 120):
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Age must be a valid number between 1 and 120."}), 400
    if not gender:
        return jsonify({"error": "Gender is required."}), 400
    if not smoker:
        return jsonify({"error": "Smoker status is required."}), 400

    success, result = create_user(
        email=email,
        phone=phone,
        name=name,
        age=age,
        gender=gender,
        smoker=smoker,
        chronic_conditions=d.get("chronic_conditions", []),
        temp_conditions=d.get("temp_conditions", [])
    )
    
    if success:
        session.update(result)
        session["profile_complete"] = True
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": result}), 400

@app.route("/api/login", methods=["POST"])
def login():
    import re
    d = request.get_json()
    email = (d.get("email") or "").strip().lower()
    phone = re.sub(r'\D', '', (d.get("phone") or "").strip())  # digits only

    if not email and not phone:
        return jsonify({"error": "Email or Phone number is required"}), 400

    # Format validation
    if email and not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({"error": "Invalid email format."}), 400
    if phone and not re.match(r'^\d{10}$', phone):
        return jsonify({"error": "Phone number must be exactly 10 digits."}), 400

    user = get_user(email, phone)
    if user:
        session.update(user)
        session["profile_complete"] = True
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "User not found. Please sign up."}), 404


@app.route("/api/update-health", methods=["POST"])
def update_health_route():
    if not session.get("profile_complete"):
        return jsonify({"error": "unauthorized"}), 401
    
    d = request.get_json()
    email = session.get("email")
    phone = session.get("phone")
    contact = session.get("contact") # legacy fallback
    if not email and not phone and not contact:
        return jsonify({"error": "missing identity in session"}), 400

    e_val = email or contact
    p_val = phone or contact

    success, result = update_health(
        email=e_val,
        phone=p_val,
        smoker=d.get("smoker", session.get("smoker", "No")),
        chronic_conditions=d.get("chronic_conditions", session.get("chronic_conditions", [])),
        temp_conditions=d.get("temp_conditions", session.get("temp_conditions", []))
    )
    if success:
        session.update(result)
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": result}), 400

@app.route("/get-profile")
def get_profile():
    if not session.get("profile_complete"): return jsonify({"error":"no profile"}),401
    return jsonify({k:session.get(k) for k in ["name","age","gender","email","phone","contact","smoker","chronic_conditions","temp_conditions"]})

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

# PAGE ROUTES
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", profile_complete=session.get("profile_complete", False))

@app.route("/map")
def aqi_map():
    return render_template("route.html", profile_complete=session.get("profile_complete", False))

# DATA ROUTES
def _get_corridor(fLat, fLon, tLat, tLon, off_lat, off_lon, width, from_name, to_name):
    """
    Returns list of dataset stations that lie within a shifted corridor
    between (fLat,fLon) and (tLat,tLon).
    Always includes the From and To stations even if outside the corridor width.
    """
    sfLat=fLat+off_lat; sfLon=fLon+off_lon
    stLat=tLat+off_lat; stLon=tLon+off_lon
    rdx=stLon-sfLon; rdy=stLat-sfLat
    rlen2=rdx*rdx+rdy*rdy
    result=[]
    for name, meta in _station_meta.items():
        lat,lon=meta["lat"],meta["lon"]
        t=((lon-sfLon)*rdx+(lat-sfLat)*rdy)/max(rlen2,1e-10)
        dist=((lon-(sfLon+t*rdx))**2+(lat-(sfLat+t*rdy))**2)**0.5
        if -0.05<=t<=1.05 and dist<width:
            result.append({"name":name,"meta":meta,"t":round(t,3)})
    result.sort(key=lambda x:x["t"])
    names_in={r["name"] for r in result}
    if from_name not in names_in:
        result.insert(0,{"name":from_name,"meta":_station_meta[from_name],"t":0.0})
    if to_name not in names_in:
        result.append({"name":to_name,"meta":_station_meta[to_name],"t":1.0})
    return result


@app.route("/find-routes", methods=["POST"])
def find_routes():
    """
    PHASE 1 (fast, no live API calls):
    Finds corridor stations from the dataset for each of the 3 route options.
    Returns station lat/lon (from dataset) so the frontend can immediately
    draw the road paths via OSRM and show station markers as 'loading'.
    No Open-Meteo calls here — AQI is fetched separately in Phase 2.
    """
    data      = request.get_json()
    from_name = data.get("from")
    to_name   = data.get("to")
    from_st = _station_meta.get(from_name)
    to_st   = _station_meta.get(to_name)
    if not from_st or not to_st:
        return jsonify({"error":"Station not found"}), 404

    fLat,fLon = from_st["lat"],from_st["lon"]
    tLat,tLon = to_st["lat"],  to_st["lon"]
    dx,dy  = tLon-fLon, tLat-fLat
    length = max((dx**2+dy**2)**0.5, 1e-10)
    perp_lat=  dx/length; perp_lon= -dy/length
    OFF=0.035

    route_defs = [
        {"id":"direct","name":"Direct Route",  "label":"🟦","offset_lat":0.0,           "offset_lon":0.0,           "width":0.08},
        {"id":"north", "name":"Northern Route","label":"🟩","offset_lat": perp_lat*OFF,  "offset_lon": perp_lon*OFF,  "width":0.09},
        {"id":"south", "name":"Southern Route","label":"🟨","offset_lat":-perp_lat*OFF, "offset_lon":-perp_lon*OFF, "width":0.09},
    ]

    routes=[]
    for rd in route_defs:
        corridor = _get_corridor(fLat,fLon,tLat,tLon,
                                 rd["offset_lat"],rd["offset_lon"],rd["width"],
                                 from_name,to_name)
        stations=[{"name":item["name"],
                   "lat":item["meta"]["lat"],"lon":item["meta"]["lon"],
                   "state":item["meta"]["state"],"city":item["meta"]["city"],
                   "is_from":item["name"]==from_name,"is_to":item["name"]==to_name}
                  for item in corridor]
        routes.append({"id":rd["id"],"name":rd["name"],"label":rd["label"],"stations":stations})

    return jsonify({"from":from_name,"to":to_name,"routes":routes})

@app.route("/stations")
def stations():
    """Station hierarchy + lat/lon metadata for dropdowns and OSRM calls."""
    coords = {n: {"lat": m["lat"], "lon": m["lon"]} for n, m in _station_meta.items()}
    return jsonify({"hierarchy": _station_hierarchy, "latest": _station_latest, "coords": coords})

@app.route("/station-info/<path:station_name>")
def station_info(station_name):
    """Returns lat/lon for a named station (for OSRM routing)."""
    meta = _station_meta.get(station_name)
    if not meta:
        return jsonify({"error": "Station not found"}), 404
    return jsonify({"name": station_name, "lat": meta["lat"], "lon": meta["lon"],
                    "state": meta["state"], "city": meta["city"]})


@app.route("/analyze-route", methods=["POST"])
def analyze_route():
    """
    Finds dataset stations within a 1.5km buffer of an OSRM route,
    fetches live Open-Meteo data for them, predicts AQI, and returns health-scored results.
    """
    data      = request.get_json()
    from_name = data.get("from")
    to_name   = data.get("to")
    route_coords = data.get("coordinates", [])   # [[lon, lat], ...]
    buffer_km = float(data.get("buffer_km", 1.5))   # tighter = only on-road stations
    buffer_deg = buffer_km / 111.0                   # 1° ≈ 111 km

    from_st = _station_meta.get(from_name)
    to_st   = _station_meta.get(to_name)
    if not from_st or not to_st:
        log.error(f"Station not found: from='{from_name}' to='{to_name}'")
        return jsonify({"error": "Station not found"}), 404
    if len(route_coords) < 2:
        return jsonify({"error": "Route geometry too short"}), 400

    log.info(f"─────────────────────────────────────────────────────")
    log.info(f"🛣  /analyze-route  FROM: {from_name}")
    log.info(f"               TO:   {to_name}")
    log.info(f"   Route points: {len(route_coords)}  Buffer: {buffer_km} km")

    total_segs = len(route_coords) - 1

    # Find dataset stations within buffer of the OSRM road
    near = []
    for name, meta in _station_meta.items():
        slon, slat = meta["lon"], meta["lat"]
        min_dist = float("inf")
        best_t   = 0.0
        for i in range(total_segs):
            x1, y1 = route_coords[i];    x2, y2 = route_coords[i+1]
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx*dx + dy*dy
            if seg_len2 < 1e-14:
                continue
            t = ((slon-x1)*dx + (slat-y1)*dy) / seg_len2
            t = max(0.0, min(1.0, t))
            px, py = x1 + t*dx, y1 + t*dy
            dist = ((slon-px)**2 + (slat-py)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                best_t   = (i + t) / max(total_segs, 1)
        if min_dist <= buffer_deg:
            near.append({"name": name, "meta": meta, "t": best_t,
                         "dist_km": round(min_dist*111, 2)})

    near.sort(key=lambda x: x["t"])

    # Always include From and To even if outside buffer
    names_in = {s["name"] for s in near}
    if from_name not in names_in:
        near.insert(0, {"name": from_name, "meta": from_st, "t": 0.0, "dist_km": 0.0})
    if to_name not in names_in:
        near.append({"name": to_name, "meta": to_st, "t": 1.0, "dist_km": 0.0})

    # ── Filter: remove intermediates that are AT the endpoint (≤0.5km) ──
    # Stations physically at the start/end location are NOT "in the middle".
    ENDPOINT_EXCL_DEG = 0.5 / 111.0   # ~0.5 km in degrees
    def geo_dist(a_meta, b_meta):
        return ((a_meta["lat"]-b_meta["lat"])**2 + (a_meta["lon"]-b_meta["lon"])**2)**0.5

    filtered = []
    for item in near:
        nm = item["name"]
        if nm == from_name or nm == to_name:
            filtered.append(item)    # endpoints always kept
            continue
        d_from = geo_dist(item["meta"], from_st)
        d_to   = geo_dist(item["meta"], to_st)
        if d_from <= ENDPOINT_EXCL_DEG or d_to <= ENDPOINT_EXCL_DEG:
            log.info(f"     ✂  {nm}: excluded (≤0.5km from endpoint — not truly intermediate)")
            continue
        filtered.append(item)
    near = filtered

    log.info(f"   Stations on route ({buffer_km}km buffer, endpoint-filtered): {len(near)}")
    for s in near:
        tag = " [START]" if s["name"]==from_name else (" [END]" if s["name"]==to_name else " [MID]")
        log.info(f"     • {s['name']}{tag}  t={s['t']:.3f}  dist={s['dist_km']}km")

    # ── Log user health profile so it can be verified in /logs ──────
    age        = session.get("age",        "—")
    chronic_conds = session.get("chronic_conditions", [])
    temp_conds = session.get("temp_conditions", [])
    conditions = chronic_conds + temp_conds
    smoker     = session.get("smoker",     "No")
    log.info(f"   👤 Health profile: Age={age}  Smoker={smoker}  Conditions={conditions or 'None'}")

    # ── Fetch live data ONLY from Open-Meteo (NO dataset fallback) ──
    log.info(f"   📡 Fetching Open-Meteo live data for {len(near)} stations…")
    batch_input = [{"name": s["name"], "lat": s["meta"]["lat"], "lon": s["meta"]["lon"]}
                   for s in near]
    live_cache = fetch_batch(batch_input, max_workers=16)

    # ── Predict AQI per station ────────────────────────────────────
    log.info("   📊 AQI predictions:")
    stations_out = []
    for item in near:
        name, meta = item["name"], item["meta"]
        readings = live_cache.get(name)

        if readings is None:
            # Per project spec: NO dataset fallback — mark as unavailable
            log.warning(f"     ⚠  {name}: Open-Meteo UNAVAILABLE — skipped (no fallback)")
            stations_out.append({
                "station": name, "lat": meta["lat"], "lon": meta["lon"],
                "state": meta["state"], "city": meta["city"],
                "aqi": None, "severity": "No Live Data", "color": "#475569",
                "pm25": None, "pm10": None, "temp": None, "humidity": None,
                "source": "unavailable",
                "is_from": name == from_name, "is_to": name == to_name
            })
            continue

        # co2 = meta["co2"] # Removed dataset fallback
        co2 = readings["co2"] # Live from Open-Meteo
        aqi = run_model(meta["lat"], meta["lon"],
                        readings["pm25"], readings["pm10"],
                        readings["temp"], readings["humidity"], co2)
        sev, _ = categorize_aqi(aqi)
        tag_str = "[S]" if name==from_name else ("[E]" if name==to_name else "[M]")
        log.info(f"     ✓ {tag_str} {name}: PM2.5={readings['pm25']} PM10={readings['pm10']}"
                 f" Temp={readings['temp']}°C Hum={readings['humidity']}% CO2={co2}ppm"
                 f" → AQI={aqi} [{sev}]")
        stations_out.append({
            "station": name, "lat": meta["lat"], "lon": meta["lon"],
            "state": meta["state"], "city": meta["city"],
            "aqi": aqi, "severity": sev,
            "color": COLOR_MAP.get(sev, "#888"),
            "pm25": readings["pm25"], "pm10": readings["pm10"],
            "temp": readings["temp"], "humidity": readings["humidity"],
            "co2": co2, "source": "open_meteo_live",
            "is_from": name == from_name, "is_to": name == to_name
        })

    valid = [s for s in stations_out if s["aqi"] is not None]
    if not valid:
        log.error("   ✗ No live AQI data — all Open-Meteo calls failed")
        return jsonify({"error": "No live AQI — Open-Meteo unavailable for all stations"}), 503

    # avg_aqi = average over ALL on-route stations (start + middle + end)
    avg_aqi = round(sum(s["aqi"] for s in valid) / len(valid), 1)
    sev, _  = categorize_aqi(avg_aqi)

    # Also compute intermediate-only avg for transparency
    mid_valid = [s for s in valid if not s["is_from"] and not s["is_to"]]
    mid_avg   = round(sum(s["aqi"] for s in mid_valid) / len(mid_valid), 1) if mid_valid else None

    route_result = {
        "stations":    stations_out,
        "avg_aqi":     avg_aqi,   # used for recommendation: lower = better
        "mid_avg_aqi": mid_avg,   # intermediate stations only
        "severity":    sev,
        "color":       COLOR_MAP.get(sev, "#888"),
        "worst":       max(valid, key=lambda s: s["aqi"]),
        "best":        min(valid, key=lambda s: s["aqi"]),
        "count":       len(stations_out),
        "valid_count": len(valid),
        "mid_count":   len(mid_valid),
        "recommended": False
    }
    hs = health_score(route_result, age, conditions, smoker)
    route_result["health_score"] = hs

    log.info(f"   📈 Route summary: AvgAQI={avg_aqi}(all) MidAvg={mid_avg}(intermediate)"
             f"  Worst={route_result['worst']['aqi']}  HealthScore={hs}"
             f"  (Age={age} Smoker={smoker} Conditions={conditions or 'None'})")
    return jsonify(route_result)


@app.route("/api/logs")
def api_logs():
    """Returns last N log entries from the in-memory ring buffer as JSON."""
    n = int(request.args.get("n", 100))
    return jsonify({"logs": _log_records[-n:]})


@app.route("/logs")
def view_logs():
    """Serves the live log viewer HTML page."""
    return render_template("route_logs.html")



@app.route("/predict-station", methods=["POST"])
def predict_station():
    """
    Takes a station name → fetches live PM2.5/PM10/temp/humidity
    from Open-Meteo → predicts CPCB AQI + personalised precautions.
    """
    data = request.get_json()
    station_name = data.get("station")
    if not station_name or station_name not in _station_meta:
        return jsonify({"error":"Station not found"}), 404
    meta = _station_meta[station_name]
    lat, lon = meta["lat"], meta["lon"]
    # co2      = meta["co2"] # Removed dataset fallback
    readings = fetch_live_readings(lat, lon)
    if readings is None:
        return jsonify({"error":"Live data unavailable from Open-Meteo"}), 503
    
    source = "open_meteo_live"
    pm25,pm10 = readings["pm25"],readings["pm10"]
    temp,hum  = readings["temp"],readings["humidity"]
    co2       = readings["co2"] # Extracted here for live API
    aqi = run_model(lat,lon,pm25,pm10,temp,hum,co2)
    severity, band = categorize_aqi(aqi)
    now    = datetime.now()
    season = get_season(now.month)
    age    = session.get("age","—"); gender = session.get("gender","—")
    smoker = session.get("smoker","No")
    chronic_conds = session.get("chronic_conditions", [])
    temp_conds = session.get("temp_conditions", [])
    conds = chronic_conds + temp_conds
    precs  = get_precautions(aqi,severity,season,age,gender,smoker,conds)
    
    # Cigarette calculation (Berkeley Earth formula: ~22 ug/m3 PM2.5 ≈ 1 cigarette/day)
    cigs_raw = pm25 / 22.0 if pm25 else 0
    
    return jsonify({
        "station":station_name,"lat":lat,"lon":lon,
        "aqi":aqi,"severity":severity,"band":band,"color":COLOR_MAP.get(severity,"#888"),
        "pm25":pm25,"pm10":pm10,"temp":temp,"humidity":hum,"co2":co2,
        "season":SEASON_MAP[season],"source":source,"precautions":precs,
        "cigs": {
            "daily": round(cigs_raw, 1),
            "weekly": round(cigs_raw * 7, 1),
            "monthly": round(cigs_raw * 30, 1)
        }
    })

@app.route("/api/aqi-trends", methods=["POST"])
def aqi_trends():
    data = request.json
    station = data.get("station")
    range_type = data.get("range", "24h")
    if not station: return jsonify({"error": "No station"}), 400
    if station not in _station_meta: return jsonify({"error": "Station not found"}), 404
    
    meta = _station_meta[station]
    lat, lon = meta["lat"], meta["lon"]

    # Determine past_days needed to cover the range natively
    if range_type in ["12h", "24h"]:
        past_days = 2
    elif range_type == "7d":
        past_days = 8
    elif range_type == "30d":
        past_days = 31
    else:
        past_days = 2

    hist_data = fetch_historical_readings(lat, lon, past_days)
    if not hist_data or not hist_data.get("time"):
        return jsonify({"error": "Failed to fetch historical data"}), 503

    times = hist_data["time"]
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
    end_idx = 0
    for i, t in enumerate(times):
        if t <= now_iso:
            end_idx = i
        else:
            break

    points = []
    if range_type == "12h":
        start_idx = max(0, end_idx - 11)
        for i in range(start_idx, end_idx + 1):
            hr_str = datetime.fromisoformat(times[i]).strftime("%I %p").lstrip('0')
            points.append((hr_str, [i]))
            
    elif range_type == "24h":
        start_idx = max(0, end_idx - 23)
        for i in range(start_idx, end_idx + 1):
            hrs_ago = end_idx - i
            if hrs_ago == 0:
                hr_str = "Now"
            elif hrs_ago % 4 == 0:
                hr_str = datetime.fromisoformat(times[i]).strftime("%I %p").lstrip('0')
            else:
                hr_str = ""
            points.append((hr_str, [i]))
            
    elif range_type == "7d":
        start_idx = max(0, end_idx - (7 * 24) + 1)
        day_groups = {}
        for i in range(start_idx, end_idx + 1):
            day_str = datetime.fromisoformat(times[i]).strftime("%a")
            if day_str not in day_groups: day_groups[day_str] = []
            day_groups[day_str].append(i)
        for day, idxs in day_groups.items():
            points.append((day, idxs))
            
    elif range_type == "30d":
        start_idx = max(0, end_idx - (30 * 24) + 1)
        day_groups = {}
        for i in range(start_idx, end_idx + 1):
            day_str = datetime.fromisoformat(times[i]).strftime("%b %d")
            if day_str not in day_groups: day_groups[day_str] = []
            day_groups[day_str].append(i)
        for day, idxs in day_groups.items():
            points.append((day, idxs))

    trend_data = []
    avg_total = 0
    valid_points = 0

    for label, idxs in points:
        pm25_vals = [hist_data["pm25"][i] for i in idxs if i < len(hist_data["pm25"]) and hist_data["pm25"][i] is not None]
        pm10_vals = [hist_data["pm10"][i] for i in idxs if i < len(hist_data["pm10"]) and hist_data["pm10"][i] is not None]
        co2_vals  = [hist_data["co2"][i] for i in idxs if i < len(hist_data["co2"]) and hist_data["co2"][i] is not None]
        temp_vals = [hist_data["temp"][i] for i in idxs if i < len(hist_data["temp"]) and hist_data["temp"][i] is not None]
        hum_vals  = [hist_data["humidity"][i] for i in idxs if i < len(hist_data["humidity"]) and hist_data["humidity"][i] is not None]
        
        if not pm25_vals or not temp_vals: 
            # If no data for this block, skip or use a previous valid point if building a continuous line. 
            # For now, we just skip it to avoid plotting 0. 
            continue
            
        avg_pm25 = sum(pm25_vals) / len(pm25_vals)
        avg_pm10 = sum(pm10_vals) / len(pm10_vals) if pm10_vals else 0
        avg_co2  = sum(co2_vals) / len(co2_vals) if co2_vals else 400
        avg_temp = sum(temp_vals) / len(temp_vals)
        avg_hum  = sum(hum_vals) / len(hum_vals) if hum_vals else 50
        
        aqi = run_model(lat, lon, avg_pm25, avg_pm10, avg_temp, avg_hum, avg_co2)
        sev, _ = categorize_aqi(aqi)
        trend_data.append({"label": label, "aqi": aqi, "color": COLOR_MAP.get(sev, "#888")})
        
        avg_total += aqi
        valid_points += 1

    station_avg = int(round(avg_total / valid_points)) if valid_points > 0 else 0

    return jsonify({
        "trend": trend_data,
        "range": range_type
    })


@app.route("/compare-routes", methods=["POST"])
def compare_routes():
    """
    Core route optimization endpoint.

    Given From and To station names:
    1. Generates 3 alternative route corridors through dataset stations:
       - Route A: Direct (straight corridor, tight)
       - Route B: Northern detour (corridor shifted northward)
       - Route C: Southern detour (corridor shifted southward)
    2. Fetches live PM2.5/PM10/temp/humidity from Open-Meteo ONLY
       for the stations in each corridor (not all 108)
    3. Runs XGBoost spatial model for each station
    4. Compares average AQI per route
    5. Recommends the route with lowest average AQI

    POST body: { "from": "Station A", "to": "Station B" }
    """
    data      = request.get_json()
    from_name = data.get("from")
    to_name   = data.get("to")

    from_st = _station_meta.get(from_name)
    to_st   = _station_meta.get(to_name)
    if not from_st or not to_st:
        return jsonify({"error": "Station not found"}), 404

    fLat, fLon = from_st["lat"], from_st["lon"]
    tLat, tLon = to_st["lat"],   to_st["lon"]

    dx, dy  = tLon - fLon, tLat - fLat
    length  = max((dx**2 + dy**2)**0.5, 1e-10)
    perp_lat =  dx / length
    perp_lon = -dy / length
    OFF      = 0.035   # ~3.5 km perpendicular offset

    route_defs = [
        {"id":"direct", "name":"Direct Route",   "label":"🟦",
         "offset_lat":0.0,         "offset_lon":0.0,         "width":0.08},
        {"id":"north",  "name":"Northern Route",  "label":"🟩",
         "offset_lat": perp_lat*OFF,"offset_lon": perp_lon*OFF,"width":0.09},
        {"id":"south",  "name":"Southern Route",  "label":"🟨",
         "offset_lat":-perp_lat*OFF,"offset_lon":-perp_lon*OFF,"width":0.09},
    ]

    def get_corridor(off_lat, off_lon, width):
        sfLat=fLat+off_lat; sfLon=fLon+off_lon
        stLat=tLat+off_lat; stLon=tLon+off_lon
        rdx,rdy=stLon-sfLon,stLat-sfLat
        rlen2=rdx*rdx+rdy*rdy
        result=[]
        for name, meta in _station_meta.items():
            lat,lon=meta["lat"],meta["lon"]
            t=((lon-sfLon)*rdx+(lat-sfLat)*rdy)/max(rlen2,1e-10)
            dist=((lon-(sfLon+t*rdx))**2+(lat-(sfLat+t*rdy))**2)**0.5
            if -0.05<=t<=1.05 and dist<width:
                result.append({"name":name,"meta":meta,"t":t})
        result.sort(key=lambda x:x["t"])
        names={r["name"] for r in result}
        if to_name   not in names: result.append(  {"name":to_name,  "meta":to_st,  "t":1.0})
        return result

    def fetch_station(name, meta):
        readings = fetch_live_readings(meta["lat"], meta["lon"])
        if readings is None:
            return None # Skip if no readings available even after fallback
        
        # co2 = meta["co2"] # Prevent using static metadata
        co2 = readings["co2"]

        aqi = run_model(meta["lat"],meta["lon"],
                        readings["pm25"],readings["pm10"],
                        readings["temp"],readings["humidity"],co2)
        severity, band = categorize_aqi(aqi)
        return {"station":name,"lat":meta["lat"],"lon":meta["lon"],
                "state":meta["state"],"city":meta["city"],
                "aqi":aqi,"severity":severity,"band":band,
                "color":COLOR_MAP.get(severity,"#888"),
                "pm25":readings["pm25"],"pm10":readings["pm10"],
                "temp":readings["temp"],"humidity":readings["humidity"],
                "co2":co2,"source":"open_meteo_live"}

    # ── Build corridors first (no API calls yet) ────────────
    corridors = {}
    for rd in route_defs:
        corridors[rd["id"]] = (rd, get_corridor(rd["offset_lat"], rd["offset_lon"], rd["width"]))

    # ── Collect ALL unique stations across ALL routes ────────
    all_station_names = {}
    for rd, corridor in corridors.values():
        for item in corridor:
            n, m = item["name"], item["meta"]
            if n not in all_station_names:
                all_station_names[n] = m

    # ── Fetch ALL stations IN PARALLEL (single batch) ────────
    print(f"⚡ Parallel fetching {len(all_station_names)} unique stations...")
    live_cache = fetch_batch(
        [{"name":n,"lat":m["lat"],"lon":m["lon"]} for n,m in all_station_names.items()],
        max_workers=16
    )

    # Build route results from cache
    routes = []
    for rd, corridor in corridors.values():
        stations = []
        for item in corridor:
            name, meta = item["name"], item["meta"]
            readings = live_cache.get(name)
            if readings is None:
                continue # Strictly skip if live API fails
            co2 = readings["co2"]

            aqi = run_model(meta["lat"],meta["lon"],
                            readings["pm25"],readings["pm10"],
                            readings["temp"],readings["humidity"],co2)
            severity, band = categorize_aqi(aqi)
            stations.append({
                "station":name,"lat":meta["lat"],"lon":meta["lon"],
                "state":meta["state"],"city":meta["city"],
                "aqi":aqi,"severity":severity,"band":band,
                "color":COLOR_MAP.get(severity,"#888"),
                "pm25":readings["pm25"],"pm10":readings["pm10"],
                "temp":readings["temp"],"humidity":readings["humidity"],
                "co2":co2,"source":"open_meteo_live",
                "is_from":name==from_name,"is_to":name==to_name
            })
        if not stations: continue
        avg_aqi = round(sum(s["aqi"] for s in stations)/len(stations),1)
        sev, _  = categorize_aqi(avg_aqi)
        routes.append({
            "id":rd["id"],"name":rd["name"],"label":rd["label"],
            "stations":stations,"avg_aqi":avg_aqi,"severity":sev,
            "color":COLOR_MAP.get(sev,"#888"),
            "worst":max(stations,key=lambda s:s["aqi"]),
            "best": min(stations,key=lambda s:s["aqi"]),
            "count":len(stations),"recommended":False
        })

    if not routes:
        return jsonify({"error":"No routes found"}), 404

    #Health-aware recommendation
    age        = session.get("age",      "—")
    conditions = session.get("conditions", [])
    smoker     = session.get("smoker",   "No")

    for route in routes:
        route["health_score"] = health_score(route, age, conditions, smoker)
        route["recommended"]  = False

    best_route = min(routes, key=lambda r: r["health_score"])
    best_route["recommended"] = True

    # Add explanation of why this route was recommended
    reason_parts = [f"Avg AQI {best_route['avg_aqi']} ({best_route['severity']})"]
    if any(c.lower() in {"asthma","copd","respiratory","lung","breathing"}
           for c in (conditions or [])):
        reason_parts.append("lowest PM2.5 exposure for your respiratory condition")
    if str(smoker).lower() in ("yes", "true"):
        reason_parts.append("accounts for smoker sensitivity")
    try:
        if int(str(age).strip()) >= 60:
            reason_parts.append("optimised for age 60+")
    except: pass
    best_route["recommendation_reason"] = " · ".join(reason_parts)

    return jsonify({
        "from":from_name,"to":to_name,
        "from_meta":from_st,"to_meta":to_st,
        "routes":routes,"recommended":best_route["id"],
        "user_profile":{"age":age,"conditions":conditions,"smoker":smoker}
    })

@app.route("/api/assistant-chat", methods=["POST"])
def assistant_chat():
    data = request.json
    message = data.get("message", "")
    profile = data.get("profile", {})
    context = data.get("context", "dashboard")
    
    if not message:
        return jsonify({"reply": "I'm here to help!", "recovered": [], "added_chronic": [], "added_temp": []})
        
    chronic = profile.get("chronic_conditions", [])
    temp = profile.get("temp_conditions", [])
    all_conditions = chronic + temp
    
    page_instructions = ""
    if context == "route":
        page_instructions = "The user is currently on the Route Optimization page. You should give them advice about travel, pollution on their routes, and general respiratory precautions."
    else:
        page_instructions = "The user is currently on their Health Dashboard. You should give them advice about managing their existing conditions, new symptoms, or general wellness."
    
    if message == "GREETING_TRIGGER":
        prompt = f"""You are CARRFS Health Assistant. The user just logged in.
Their known conditions: {', '.join(all_conditions) if all_conditions else 'None'}.
If they have conditions, greet them warmly and ask specifically how their conditions are feeling today (max 2 sentences).
If they have no conditions, just say "Hello! How are you feeling today?" (1 sentence).
Return JSON EXACTLY in this format: {{"reply": "...", "recovered": [], "added_chronic": [], "added_temp": []}}"""
    else:
        prompt = f"""You are CARRFS Health Assistant, a friendly and concise AI doctor.
The user's current known CHRONIC conditions (long-term, do NOT remove unless explicitly named): {', '.join(chronic) if chronic else 'None'}.
The user's current known TEMPORARY conditions (short-term, clear these if they feel better): {', '.join(temp) if temp else 'None'}.
{page_instructions}
User message: "{message}"

Your tasks:
1. Provide a short, empathetic response (max 2-3 sentences).
2. If the user says they are "feeling better", "good", or generally recovered, add ALL of their TEMPORARY conditions to the "recovered" JSON array to clear them. NEVER add CHRONIC conditions to the "recovered" array unless they specifically name the chronic disease and say it's cured.
3. ONLY if the user explicitly states they are currently suffering from, or diagnosed with a NEW condition (not in their known list), categorize it as either "added_chronic" or "added_temp" and add the clinical name to the respective array.

Return EXACTLY a valid JSON object with four keys: "reply" (string), "recovered" (list of strings), "added_chronic" (list of strings), "added_temp" (list of strings).
Example format:
{{"reply": "...", "recovered": ["..."], "added_chronic": ["..."], "added_temp": ["..."]}}
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        response_text = completion.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        
        recovered_list = parsed.get("recovered", [])
        valid_recovered = [c for c in recovered_list if c in all_conditions]
        
        return jsonify({
            "reply": parsed.get("reply", "I am glad to hear that. Stay safe!"),
            "recovered": valid_recovered,
            "added_chronic": parsed.get("added_chronic", []),
            "added_temp": parsed.get("added_temp", [])
        })
    except Exception as e:
        log.error(f"Groq Assistant Chat Error: {e}")
        return jsonify({"reply": "Sorry, I am having trouble connecting to my brain right now. Please try again.", "recovered": [], "added_chronic": [], "added_temp": []}), 500

@app.route("/api/suggest-condition", methods=["POST"])
def suggest_condition():
    data = request.json
    description = data.get("description", "")
    
    CHRONIC_CONDITIONS = [
        "Asthma", "COPD", "Chronic Bronchitis", "Emphysema", "Pulmonary Fibrosis",
        "Lung Cancer", "Pneumonia History", "Heart Disease", "Hypertension", "Allergies",
        "Dust Sensitivity"
    ]
    
    prompt = f"""The user is trying to add a medical condition to their profile, but it is not in the standard list.
User's description: "{description}"

Standard list:
{', '.join(CHRONIC_CONDITIONS)}

Analyze the description. If it closely matches or is a synonym for one of the standard conditions, return exactly that condition string.
If it is completely different, return a short, properly capitalized clinical name for what they described (max 3 words).
Reply ONLY with the condition name. No extra text, no quotes."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        suggestion = completion.choices[0].message.content.strip().replace('"', '')
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        log.error(f"Groq Suggestion Error: {e}")
        return jsonify({"error": "Failed to generate suggestion"}), 500

if __name__ == "__main__":
    print("CARRFS at http://localhost:5005")
    app.run(debug=True, port=5005)
