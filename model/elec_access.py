import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

df_verts = pd.read_excel('vertiport_name_with_coord.xlsx')
df_power = pd.read_csv('Major_Power_Stations.csv')

# drop power stations with no coordinates
df_power = df_power.dropna(subset=['X', 'Y'])

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

MIN_STATION_MW  = 10    # minimum generation capacity to count as reliable grid
MAX_DISTANCE_KM = 50    # maximum distance to nearest viable station

# ─────────────────────────────────────────────────────────────
# HAVERSINE
# ─────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ─────────────────────────────────────────────────────────────
# FILTER TO VIABLE POWER STATIONS ONLY
# ─────────────────────────────────────────────────────────────

df_power_feasible = df_power[df_power['generationmw'] >= MIN_STATION_MW].copy()

# ─────────────────────────────────────────────────────────────
# FIND NEAREST VIABLE POWER STATION PER VERTIPORT
# ─────────────────────────────────────────────────────────────

rows = []

for _, vrow in df_verts.iterrows():
    v_name = vrow['name']
    v_lat  = float(vrow['latitude'])
    v_lon  = float(vrow['longitude'])

    best_dist = float('inf')
    best_row  = None

    for _, prow in df_power_feasible.iterrows():
        p_lat = float(prow['Y'])
        p_lon = float(prow['X'])
        dist  = haversine_km(v_lat, v_lon, p_lat, p_lon)
        if dist < best_dist:
            best_dist = dist
            best_row  = prow

    if best_row is not None:
        electricity_feasible = int(best_dist <= MAX_DISTANCE_KM)
        rows.append({
            'vertiport':             v_name,
            'vertiport_lat':         v_lat,
            'vertiport_lon':         v_lon,
            'nearest_power_station': best_row['name'],
            'distance_km':           round(best_dist, 2),
            'generationMW':          best_row['generationmw'],
            'electricity_feasible':  electricity_feasible,
        })
    else:
        rows.append({
            'vertiport':             v_name,
            'vertiport_lat':         v_lat,
            'vertiport_lon':         v_lon,
            'nearest_power_station': None,
            'distance_km':           None,
            'generationMW':          None,
            'electricity_feasible':  0,
        })

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
df_out.to_excel('vertiport_power_station.xlsx', index=False)

feasible_count = df_out['electricity_feasible'].sum()
print(f"\nSaved {len(df_out)} rows to vertiport_power_station.xlsx")
print(f"Electrically feasible  : {feasible_count} / {len(df_out)} vertiports")
print(f"Electrically infeasible: {len(df_out) - feasible_count} / {len(df_out)} vertiports")

# ────────────────────────────────────────────────────────────
# FEASIBILITY ANALYSIS
# ────────────────────────────────────────────────────────────
df_power_viable = df_power[df_power['generationmw'] >= MIN_STATION_MW].copy()

nearest_dist = []
for _, vrow in df_verts.iterrows():
    v_lat, v_lon = float(vrow['latitude']), float(vrow['longitude'])
    df_power_viable['dist'] = df_power_viable.apply(
        lambda row: haversine_km(v_lat, v_lon, float(row['Y']), float(row['X'])), axis=1
    )
    nearest_dist.append(df_power_viable['dist'].min())

nearest_dist = pd.Series(nearest_dist)

thresholds = [1, 2, 4, 5, 10, 15, 20, 30, 40, 50]
feasible_counts = [int((nearest_dist <= t).sum()) for t in thresholds]

plt.figure(figsize=(8,6))
plt.plot(thresholds, feasible_counts, marker='o', color='steelblue')
for t, c in zip(thresholds, feasible_counts):
    plt.annotate(str(c), (t, c),
                 textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
plt.xlabel('Max Distance to Nearest Viable Power Station (km)')
plt.ylabel('Number of Feasible Vertiport Sites')
plt.title(f'Electrically Feasible Vertiport Sites vs Distance Threshold (MW >= {MIN_STATION_MW})')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('electricity_sensitivity.png', dpi=150)
plt.show()
print("Plot saved to electricity_sensitivity.png")