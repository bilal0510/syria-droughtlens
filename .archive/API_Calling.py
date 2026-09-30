import time
import requests
import pandas as pd
import numpy as np

# --- 1. Locations (name: (lat, lon)) ---
LOCATIONS = {
    "Damascus":    (33.51, 36.29),
    "Aleppo":      (36.20, 37.16),
    "Homs":        (34.73, 36.72),
    "Latakia":     (35.52, 35.79),
    "Raqqa":       (35.95, 39.01),
    "Deir ez-Zor": (35.34, 40.14),
    "Hasakah":     (36.50, 40.75),
    "Daraa":       (32.63, 36.10),
}

CORE  = ["PRECTOTCORR", "T2M"]
EXTRA = ["T2M_MAX", "T2M_MIN", "RH2M", "GWETROOT", "GWETTOP", "EVLAND"]
URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


# --- 2. Fetch one location ---
def fetch_point(name, lat, lon, params, retries=3):
    q = {
        "parameters": ",".join(params),
        "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": "19810101", "end": "20251231",
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=q, timeout=180)
            r.raise_for_status()
            data = r.json()["properties"]["parameter"]
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            df.insert(0, "region", name)
            return df
        except Exception as e:
            print(f"  {name}: attempt {attempt + 1} failed -> {e}")
            time.sleep(3)
    return None


# --- 3. Pull everything (try extras first, fall back to core) ---
frames = []
for name, (lat, lon) in LOCATIONS.items():
    print(f"Fetching {name}...")
    df = fetch_point(name, lat, lon, CORE + EXTRA)
    if df is None:
        print(f"  {name}: retrying with core variables only")
        df = fetch_point(name, lat, lon, CORE)
    if df is not None:
        frames.append(df)
    time.sleep(1)  # be polite to the API

raw = pd.concat(frames)
raw.to_csv("power_raw.csv")
print(raw.shape)