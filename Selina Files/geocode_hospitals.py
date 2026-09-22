import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- FILE PATHS ----------
INPUT_FILE = "Raw_NSW_hospital_data.csv"
OUTPUT_FILE = "hospitals_with_lat_lon.csv"

# ---------- LOAD DATA ----------
df = pd.read_csv(INPUT_FILE)

# Make sure these columns exist (adjust names if needed)
REQUIRED_COLS = ["Name", "Address", "Suburb", "Postcode", "LHD", "ED"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------- FILTER TO EMERGENCY-CAPABLE HOSPITALS ----------
df = df[df["ED"].str.contains("Reporting", na=False)].copy()

# ---------- SET UP GEOCODER ----------
geolocator = Nominatim(user_agent="rural_uav_emergency_access_study")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# ---------- BUILD FULL ADDRESS ----------
def build_address(row):
    parts = [
        str(row["Name"]),
        str(row["Address"]),
        str(row["Suburb"]),
        "NSW",
        str(row["Postcode"]),
        "Australia"
    ]
    return ", ".join(p for p in parts if p and p != "nan")

df["full_address"] = df.apply(build_address, axis=1)

# ---------- GEOCODE ----------
latitudes = []
longitudes = []

for i, addr in enumerate(df["full_address"], start=1):
    print(f"Geocoding {i}/{len(df)}: {addr}")
    try:
        location = geocode(addr)
        if location:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)
    except Exception as e:
        print("  ERROR:", e)
        latitudes.append(None)
        longitudes.append(None)

df["latitude"] = latitudes
df["longitude"] = longitudes

# ---------- SAVE ----------
df.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved to: {OUTPUT_FILE}")
