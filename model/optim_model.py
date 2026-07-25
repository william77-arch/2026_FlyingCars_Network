import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import requests
import time
import folium
import matplotlib.pyplot as plt
import numpy as np

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

UAV_SPEED_KMH = 250
TIME_THRESHOLD = 120
COVERAGE_THRESHOLD = 0.9

#--------------------
# D_IH MATRIX
#--------------------
# Form a matrix describing distance between each candidate vertiport and hospital
# Get the actual distance rather than the mathematical distance
'''
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

            # Add the data to the list
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

# Extract the road time from d_ij matrix and store it in a dictionary 
road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min'] 
                    for _, row in d_ij.iterrows()}


a = {}
for i in demand_coords:
    for j in candidate_coords:
        road_time = road_time_lookup[i, j]
        air_time  = d_ih.loc[j].min() / UAV_SPEED_KMH * 60
        if road_time + air_time <= TIME_THRESHOLD:
            a[i, j] = 1 # SA2 = 1 if covered within time
        else:
            a[i,j] = 0 # SA2 = 0 if not covered within time

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

# Add coverage constraint
m.cov = Constraint(m.I, rule=lambda m, i:
    m.z[i] <= sum(a[i, j] * m.y[j] for j in J))

# Adjust coverage percentage criteria 
m.demand = Constraint(expr= sum(m.z[i] for i in I) >= COVERAGE_THRESHOLD * len(I))

SolverFactory('cbc').solve(m)
'''
#--------------------
# SENSITIVITY ANALYSIS
#--------------------

time_thresholds  = [30, 45, 60, 75, 90, 105, 120]
coverage_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]

d_ih = pd.read_excel('vertiport_hospital_distances.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel('d_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min'] 
                    for _, row in d_ij.iterrows()}

results = {}

for T in time_thresholds:
    a = {}
    for i in demand_coords:
        for j in candidate_coords:
            road_time = road_time_lookup[i, j]
            air_time  = d_ih.loc[j].min() / UAV_SPEED_KMH * 60
            if road_time + air_time <= T:
                a[i, j] = 1
            else:
                a[i,j] = 0

    for alpha in coverage_targets:
        I = list(demand_coords.keys())
        J = list(candidate_coords.keys())

        m = ConcreteModel()
        m.I = Set(initialize=I)
        m.J = Set(initialize=J)
        m.y = Var(m.J, domain=Binary)
        m.z = Var(m.I, domain=Binary)

        m.obj = Objective(expr=sum(m.y[j] for j in J), sense=minimize)

        m.cov = Constraint(m.I, rule=lambda m, i:
            m.z[i] <= sum(a[i, j] * m.y[j] for j in J))

        m.demand = Constraint(expr=
            sum(m.z[i] for i in I) >= alpha * len(I))

        result = SolverFactory('cbc').solve(m)

        if result.solver.termination_condition == TerminationCondition.optimal:
            n = sum(1 for j in J if value(m.y[j]) > 0.5)
            results[T, alpha] = n
            print(f"T={T} min | coverage={alpha*100:.0f}% | vertiports={n}")
        else:
            results[T, alpha] = None
            print(f"T={T} min | coverage={alpha*100:.0f}% | INFEASIBLE")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1 — lines per coverage target, x = time threshold
ax1 = axes[0]
for alpha in coverage_targets:
    y_vals = [results[T, alpha] for T in time_thresholds]
    y_vals = [v if v is not None else np.nan for v in y_vals]
    ax1.plot(time_thresholds, y_vals, marker='o', label=f'{alpha*100:.0f}%')
    for T, v in zip(time_thresholds, y_vals):
        if not np.isnan(v):
            ax1.annotate(str(int(v)), (T, v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

ax1.set_xlabel('Time Threshold (minutes)')
ax1.set_ylabel('Minimum Vertiports Required')
ax1.set_title('Vertiports vs Time Threshold')
ax1.legend(title='Coverage Target')
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot 2 — lines per time threshold, x = coverage target
ax2 = axes[1]
for T in time_thresholds:
    y_vals = [results[T, alpha] for alpha in coverage_targets]
    y_vals = [v if v is not None else np.nan for v in y_vals]
    ax2.plot([a*100 for a in coverage_targets], y_vals, marker='o', label=f'{T} min')
    for alpha, v in zip(coverage_targets, y_vals):
        if not np.isnan(v):
            ax2.annotate(str(int(v)), (alpha*100, v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

ax2.set_xlabel('Coverage Target (%)')
ax2.set_ylabel('Minimum Vertiports Required')
ax2.set_title('Vertiports vs Coverage Target')
ax2.legend(title='Time Threshold')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=150)
plt.show()
print("Plot saved to sensitivity_analysis.png")

# for each coverage target, find the threshold where adding more time saves least vertiports
for alpha in coverage_targets:
    vals = [(T, results[T, alpha]) for T in time_thresholds if results[T, alpha] is not None]
    
    if len(vals) < 2:
        continue
    
    # find biggest drop in vertiports per minute increase
    best_efficiency = 0
    best_T = None
    for k in range(1, len(vals)):
        T_prev, n_prev = vals[k-1]
        T_curr, n_curr = vals[k]
        if n_prev is not None and n_curr is not None:
            drop = n_prev - n_curr          # vertiports saved
            time_cost = T_curr - T_prev     # minutes added
            efficiency = drop / time_cost   # vertiports saved per minute
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_T = T_curr
    
    print(f"Coverage {alpha*100:.0f}%: biggest gain at T={best_T} min "
          f"({best_efficiency:.2f} vertiports saved per minute)")

'''
#--------------------
# RESULTS
#--------------------

selected  = [j for j in J if value(m.y[j]) > 0.5]
covered   = [i for i in I if value(m.z[i]) > 0.5]
uncovered = [i for i in I if value(m.z[i]) < 0.5]

print('Time threshold =',TIME_THRESHOLD, 'min')
print('Vehicle Speed =',UAV_SPEED_KMH, 'km/h')
print(f"Coverage threshold = {COVERAGE_THRESHOLD*100:.0f}%\n")

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

air_radius = (TIME_THRESHOLD / 60) * UAV_SPEED_KMH * 1000

m_map = folium.Map(location=[-33.0, 151.0], zoom_start=7)

# selected vertiports - blue
for j in selected:
    lat, lon = candidate_coords[j]
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color='blue',
        fill=True,
        fill_opacity=1,
        popup=folium.Popup(j, max_width=200),
        tooltip=j
    ).add_to(m_map)

# covered SA2s - red
for i in covered:
    lat, lon = demand_coords[i]
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color='red',
        fill=True,
        fill_opacity=1,
        popup=folium.Popup(i, max_width=200),
        tooltip=i
    ).add_to(m_map)

#uncovered - orange
for i in uncovered:
    lat, lon = demand_coords[i]
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        color='orange',
        fill=True,
        fill_opacity=1,
        popup=folium.Popup(i, max_width=200),
        tooltip=i
    ).add_to(m_map)

# reachable hospitals only - green
for h, (h_lat, h_lon) in hospital_coords.items():
    reachable = any(
        d_ih.loc[j, h] / UAV_SPEED_KMH * 60 <= TIME_THRESHOLD
        for j in selected
    )
    if reachable:
        folium.CircleMarker(
            location=[h_lat, h_lon],
            radius=6,
            color='green',
            fill=True,
            fill_opacity=1,
            popup=folium.Popup(h, max_width=200),
            tooltip=h
        ).add_to(m_map)

# Create legend HTML
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 220px; height: 160px; 
     background-color: white; border:2px solid grey; z-index:9999; 
     font-size:14px; padding: 10px; border-radius: 5px;">
     
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>

<p style="margin: 5px 0;"><i style="background:blue; width: 12px; height: 12px; 
display: inline-block; border-radius: 50%;"></i> Selected Vertiports</p>

<p style="margin: 5px 0;"><i style="background:red; width: 12px; height: 12px; 
display: inline-block; border-radius: 50%;"></i> Covered Regions (SA2)</p>

<p style="margin: 5px 0;"><i style="background:orange; width: 12px; height: 12px; 
display: inline-block; border-radius: 50%;"></i> Uncovered Regions (SA2)</p>

<p style="margin: 5px 0;"><i style="background:green; width: 12px; height: 12px; 
display: inline-block; border-radius: 50%;"></i> Reachable Hospitals</p>

</div>
'''

m_map.get_root().html.add_child(folium.Element(legend_html))

m_map.save('vertiports_map.html')
print("Map saved to vertiports_map.html")


#70-100
#2 hours