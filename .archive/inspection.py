import time
import requests
import pandas as pd
import numpy as np

# --- 4. Inspect ---
raw = pd.read_csv("power_raw.csv", parse_dates=["date"], index_col="date")
print(raw.groupby("region").size())     # expect 16436 each
print(raw.dtypes)
print((raw == -999).sum())              # NASA's missing-value flag
print(raw.describe().T)

# --- 5. Clean ---
df = raw.replace(-999, np.nan)

# Interpolate short gaps per region WITHOUT touching the region column
num_cols = [c for c in df.columns if c != "region"]
df[num_cols] = df.groupby("region")[num_cols].transform(lambda s: s.interpolate(limit=3))

# Time features
df["year"]  = df.index.year
df["month"] = df.index.month
df["hydro_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])

# Keep complete hydrological years only (1982-2025)
df = df[(df["hydro_year"] >= 1982) & (df["hydro_year"] <= 2025)]

df.to_csv("power_clean.csv")
print(df.columns.tolist())
print(df.groupby("region").size())   # expect 16071 each