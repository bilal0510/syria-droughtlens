"""Step 1: download daily climate data from the NASA POWER API."""
import time

import pandas as pd
import requests

from config import LOCATIONS, POWER_URL, COMMUNITY, CORE_PARAMS, EXTRA_PARAMS, START_DATE, END_DATE


def fetch_point(name, lat, lon, params, retries=3):
    """Download daily data for one location. Returns a DataFrame or None."""
    query = {
        "parameters": ",".join(params),
        "community": COMMUNITY,
        "latitude": lat,
        "longitude": lon,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(POWER_URL, params=query, timeout=180)
            response.raise_for_status()
            data = response.json()["properties"]["parameter"]
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            df.insert(0, "region", name)
            return df
        except Exception as exc:
            print(f"  {name}: attempt {attempt}/{retries} failed -> {exc}")
            time.sleep(3)
    return None


def fetch_all(locations=None):
    """Download every location and stack them into one DataFrame."""
    locations = locations or LOCATIONS
    frames = []
    for name, (lat, lon) in locations.items():
        print(f"Fetching {name}...")
        df = fetch_point(name, lat, lon, CORE_PARAMS + EXTRA_PARAMS)
        if df is None:
            print(f"  {name}: retrying with core variables only")
            df = fetch_point(name, lat, lon, CORE_PARAMS)
        if df is None:
            # Fail loudly: silently skipping a region would bias every result.
            raise RuntimeError(f"Could not download data for {name}")
        frames.append(df)
        time.sleep(1)  # be polite to the API
    return pd.concat(frames)
