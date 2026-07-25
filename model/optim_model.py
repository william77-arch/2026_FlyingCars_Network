import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import requests
import pickle
import time
import folium

#--------------------
# LOAD THE DATA SETS
#--------------------

dfi = pd.read_excel('vertiport_name_with_coord.xlsx')
dfj = pd.read_excel('sa2_names_with_coord.xlsx')
dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')

# Candidate vertiports
candidate_coords = dict(zip(dfi['name'],  zip(dfi['latitude'], dfi['longitude'])))

# Demand nodes
demand_coords    = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))

# Hospitals
hospital_coords  = dict(zip(dfh['Name'], zip(dfh['latitude'], dfh['longitude'])))

#--------------------
#   PARAMETERS
#--------------------

UAV_SPEED_KMH = 120
TIME_THRESHOLD = 90
COVERAGE_THRESHOLD = 0.85 # Minimum coverage percentage (0.0 to 1.0)

#--------------------
# D_IH MATRIX
#--------------------
# Form a matrix describing distance between each candidate vertiport and hospital
'''
# Get the actual distance rather than the mathematical distance
 
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Create a list of dictionaries{vertiport name, hospital name, distance}
rows = []

for i, (i_lat, i_lon) in candidate_coords.items():
    for h, (h_lat, h_lon) in hospital_coords.items():
        dist = haversine_km(i_lat, i_lon, h_lat, h_lon)
        rows.append({
            'vertiport':   i,
            'hospital':    h,
            'distance_km': round(dist, 2)
        })

# Convert to excel, each cell contains the distance

d_ih = pd.DataFrame(rows)
d_ih_pivot = d_ih.pivot(index='vertiport', columns='hospital', values='distance_km')
d_ih_pivot.to_excel('vertiport_hospital_distances.xlsx')
print('d_ih matrix saved')

#--------------------
# D_IJ MATRIX
#--------------------
# Form a matrix describing distance and time between each demand node and candidate vertiport
# using OSRM - uses OpenStreetMap

def get_osrm_table(demand_coords, candidate_coords, chunk_size=25):
    demand_list    = list(demand_coords.keys())
    candidate_list = list(candidate_coords.keys())
    demand_vals    = list(demand_coords.values())
    candidate_vals = list(candidate_coords.values())

    # Create a list of dictionaries{demand node, vertiport, time(min), distance(km)}
    rows = []

    # Break down the data into chunks for faster processing
    for i_start in range(0, len(demand_list), chunk_size):
        i_keys = demand_list[i_start:i_start + chunk_size]
        i_vals = demand_vals[i_start:i_start + chunk_size]

        for j_start in range(0, len(candidate_list), chunk_size):
            j_keys = candidate_list[j_start:j_start + chunk_size]
            j_vals = candidate_vals[j_start:j_start + chunk_size]

            all_coords   = i_vals + j_vals
            coords_str   = ';'.join(f"{lon},{lat}" for lat, lon in all_coords)
            n_i          = len(i_keys)
            sources      = ';'.join(str(k) for k in range(n_i))
            destinations = ';'.join(str(k) for k in range(n_i, n_i + len(j_keys)))

            url = (f"http://router.project-osrm.org/table/v1/driving/{coords_str}"
                   f"?sources={sources}&destinations={destinations}&annotations=duration,distance")

            response = requests.get(url, timeout=60)
            data = response.json()

            for i, i_name in enumerate(i_keys):
                for j, j_name in enumerate(j_keys):
                    rows.append({
                        'sa2':         i_name,
                        'vertiport':   j_name,
                        'time_min':    data['durations'][i][j] / 60,
                        'distance_km': data['distances'][i][j] / 1000
                    })

            time.sleep(0.3)

    return pd.DataFrame(rows)

d_ij = get_osrm_table(demand_coords, candidate_coords)
d_ij.to_excel('d_ij.xlsx', index=False)

print(f"D_ij matrix saved")
'''
#------------------
# COVERAGE MATRIX
#------------------
d_ih = pd.read_excel('vertiport_hospital_distances.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel('d_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min'] 
                    for _, row in d_ij.iterrows()}


a = {}
for i in demand_coords:
    for j in candidate_coords:
        road_time = road_time_lookup[i, j]
        air_time  = d_ih.loc[j].min() / UAV_SPEED_KMH * 60
        if road_time + air_time <= TIME_THRESHOLD:
            a[i, j] = 1
        else:
            a[i,j] = 0

#--------------------
# PYOMO MODEL
#--------------------

I = list(demand_coords.keys())
J = list(candidate_coords.keys())

m = ConcreteModel()
m.I = Set(initialize=I)
m.J = Set(initialize=J)

m.y = Var(m.J, domain=Binary)   # 1 = select this vertiport
m.z = Var(m.I, domain=Binary)   # 1 = this SA2 is covered

m.obj = Objective(expr=sum(m.y[j] for j in J), sense=minimize)

m.cov = Constraint(m.I, rule=lambda m, i:
    m.z[i] <= sum(a[i, j] * m.y[j] for j in J))

m.demand = Constraint(expr= sum(m.z[i] for i in I) >= COVERAGE_THRESHOLD * len(I))

SolverFactory('cbc').solve(m)

#--------------------
# RESULTS
#--------------------

selected  = [j for j in J if value(m.y[j]) > 0.5]
covered   = [i for i in I if value(m.z[i]) > 0.5]
uncovered = [i for i in I if value(m.z[i]) < 0.5]

print('Time threshold =',TIME_THRESHOLD, 'min')
print('Vehicle Speed =',UAV_SPEED_KMH, 'km/h\n')

print(f"Minimum vertiports : {len(selected)}")
print("Selected sites     :")
for port in selected:
    print('\t', port)
print('\n')

print(f"Regions covered    : {len(covered)} / {len(I)}")
print("Uncovered regions  :")
for reg in uncovered:
    print('\t', reg)
print('\n')

#------------------
# MAP THE RESULTS
#------------------

map = folium.Map(location=[-33.0, 151.0], zoom_start=6)

for j in selected:
    lat, lon = candidate_coords[j]
    folium.Marker(
        location=[lat, lon],
        popup=j,
        tooltip=j,
        icon=folium.Icon(color='blue', icon='plane', prefix='fa')
    ).add_to(map)

map.save('vertiports_map.html')
print("Map saved to vertiports_map.html")
