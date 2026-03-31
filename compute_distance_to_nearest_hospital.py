import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ----------------------------
# FILE PATHS
# ----------------------------
SA2_FILE = "sa2_master_table.xlsx"
HOSPITAL_FILE = "Emergency_hospitals_with_lat_lon.csv"
OUTPUT_FILE = "sa2_master_with_hospital_distance.xlsx"

# ----------------------------
# HAVERSINE DISTANCE FUNCTION
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ----------------------------
# LOAD DATA
# ----------------------------
sa2 = pd.read_excel(SA2_FILE)
hosp = pd.read_csv(HOSPITAL_FILE)

# Drop hospitals without coordinates (safety)
hosp = hosp.dropna(subset=["latitude", "longitude"])

# ----------------------------
# COMPUTE DISTANCE
# ----------------------------
nearest_distances = []

for _, sa2_row in sa2.iterrows():
    sa2_lat = sa2_row["latitude"]
    sa2_lon = sa2_row["longitude"]

    # Compute distance to all hospitals
    distances = hosp.apply(
        lambda h: haversine_km(
            sa2_lat, sa2_lon,
            h["latitude"], h["longitude"]
        ),
        axis=1
    )

    nearest_distances.append(distances.min())

sa2["distance_to_nearest_emergency_hospital_km"] = nearest_distances

# ----------------------------
# SAVE OUTPUT
# ----------------------------
sa2.to_excel(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved to: {OUTPUT_FILE}")
