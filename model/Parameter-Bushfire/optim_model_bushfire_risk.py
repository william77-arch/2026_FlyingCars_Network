"""
optim_model_bushfire_risk.py

Extends optim_model.py to incorporate bushfire (BFPL) risk into vertiport site selection using a SOFT PENALTY 
with a built-in trade-off rather than a flat penalty.

Design Intent
------------------------------------------------------------------------
Vertiports sited on high bushfire-risk land (BFPL Category 1) carry a penalty, since the physical asset itself could be 
threatened by fire. BUT eVTOL infrastructure is also a plausible emergency evacuation tool during a bushfire, so a 
Category 1 site isn't simply excluded (if it can reach OTHER high-risk zones within a designated emergency response
time (default 60 min, i.e. 1 hour, by air), some of that penalty is earned back). This reflects its value as a 
rapid-response evacuation link during a fire event - weighted by the POPULATION each reachable high-risk vertiport 
serves, so reaching a site protecting a large community counts for more than reaching one covering a handful of people.

    net_bushfire_cost(j) = risk_penalty(j) - emergency_value(j)
                           (floored at 0 -- offset, never a net
                            incentive to add unnecessary vertiports)

    emergency_value(j) = EMERGENCY_VALUE_PER_1000_POP x
                          (population served by reachable Category 1
                           vertiports / 1000)

"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import requests
import time
import folium
import matplotlib.pyplot as plt

#--------------------
# LOAD THE DATA SETS
#--------------------

dfi = pd.read_excel('vertiport_name_with_coord.xlsx')
dfj = pd.read_excel('sa2_names_with_coord.xlsx')
dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')
dfb = pd.read_csv('vertiport_bushfire_risk.csv')

candidate_coords = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords    = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords  = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))

# clean up bushfire labels (strip whitespace, treat blank/unlabelled as unknown)
dfb['bfpl_category_label'] = dfb['bfpl_category_label'].astype(str).str.strip()
dfb = dfb.dropna(subset=['name']).drop_duplicates(subset=['name']).set_index('name')

#--------------------
#   PARAMETERS
#--------------------

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 90
COVERAGE_THRESHOLD = 0.85

# --- bushfire risk settings ---
BUSHFIRE_PENALTY_WEIGHT   = 3.0    # cost applied at Category 1 (scaled down for lower categories)
EMERGENCY_TIME_THRESHOLD  = 60     # minutes (required time to reach a high-risk zone for evacuation)
# --- population weighting ---
EMERGENCY_VALUE_PER_1000_POP = 0.4   # cost offset earned per 1,000 people served by a reachable high-risk vertiport

demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))

# Relative severity by BFPL category -- Category 1 is highest risk (forest/
# woodland/heath, the most fire-prone vegetation class); severity scales
# the penalty down for lower-risk categories rather than treating risk as
# binary in/out.
BFPL_SEVERITY = {
    'Vegetation Category 1': 1.0, # High risk (forest/woodland/heath)
    'Vegetation Category 2': 0.3, # Low risk (rainforests/managed grasslands)
    'Vegetation Category 3': 0.6, # Medium risk (grasslands/shrublands)
    'Vegetation Buffer':     0.15,
    'Not bush fire prone':   0.0,
}

#--------------------
# HAVERSINE (for vertiport-to-vertiport flight time)
#--------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

#--------------------
# BUSHFIRE RISK METRICS
#--------------------

missing_bushfire = [j for j in candidate_coords if j not in dfb.index]
if missing_bushfire:
    print(f"NOTE: {len(missing_bushfire)} candidate vertiport(s) have no bushfire risk match "
          f"-- treated as zero risk:")
    for j in missing_bushfire:
        print('\t', j)

def bfpl_severity(j):
    if j not in dfb.index:
        return 0.0
    label = dfb.loc[j, 'bfpl_category_label']
    return BFPL_SEVERITY.get(label, 0.0)   # unrecognised/blank labels are treated as zero risk

# Category 1 (highest-risk) vertiports used as the "bushfire emergency location" proxy for the evacuation-reach bonus
high_risk_vertiports = [
    j for j in candidate_coords
    if j in dfb.index and dfb.loc[j, 'bfpl_category_label'] == 'Vegetation Category 1'
]
print(f"High-risk (Category 1) vertiports identified: {len(high_risk_vertiports)}")

def population_served(j, a):
    """Population of the SA2s this vertiport covers within TIME_THRESHOLD --
    i.e. how many people currently rely on this vertiport for coverage.
    Used to weight how valuable it is to reach this vertiport in an
    emergency: a high-risk vertiport serving a large population is a more
    meaningful evacuation target than one serving very few people."""
    return sum(demand_population[i] for i in demand_coords if a[i, j] == 1)

def emergency_reach_population(j, a, pop_served):
    """Total population served by Category 1 vertiports reachable from j
    by direct flight within EMERGENCY_TIME_THRESHOLD minutes (excluding
    itself)."""
    j_lat, j_lon = candidate_coords[j]
    total_pop = 0
    for target in high_risk_vertiports:
        if target == j:
            continue
        t_lat, t_lon = candidate_coords[target]
        flight_time = haversine_km(j_lat, j_lon, t_lat, t_lon) / UAV_SPEED_KMH * 60
        if flight_time <= EMERGENCY_TIME_THRESHOLD:
            total_pop += pop_served[target]
    return total_pop

def bushfire_cost(j, a, pop_served):
    risk_penalty    = BUSHFIRE_PENALTY_WEIGHT * bfpl_severity(j)
    emergency_value = EMERGENCY_VALUE_PER_1000_POP * (emergency_reach_population(j, a, pop_served) / 1000)
    return max(0.0, risk_penalty - emergency_value)   # floor at 0 (offset, not incentive)

#--------------------
# D_IH MATRIX
#--------------------
'''
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

rows = []
for i, (i_lat, i_lon) in candidate_coords.items():
    for h, (h_lat, h_lon) in hospital_coords.items():
        dist = haversine_km(i_lat, i_lon, h_lat, h_lon)
        rows.append({'vertiport': i, 'hospital': h, 'distance_km': round(dist, 2)})

d_ih = pd.DataFrame(rows)
d_ih_pivot = d_ih.pivot(index='vertiport', columns='hospital', values='distance_km')
d_ih_pivot.to_excel('vertiport_hospital_distances.xlsx')
print('d_ih matrix saved')

#--------------------
# D_IJ MATRIX
#--------------------

def get_osrm_table(demand_coords, candidate_coords, chunk_size=25):
    demand_list    = list(demand_coords.keys())
    candidate_list = list(candidate_coords.keys())
    demand_vals    = list(demand_coords.values())
    candidate_vals = list(candidate_coords.values())
    rows = []

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
print("D_ij matrix saved")
'''

#--------------------
# LOAD MATRICES
#--------------------

d_ih = pd.read_excel('vertiport_hospital_distances.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel('d_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min']
                    for _, row in d_ij.iterrows()}

#--------------------
# FUNCTIONS
#--------------------

def build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T):
    a = {}
    for i in demand_coords:
        for j in candidate_coords:
            road_time = road_time_lookup[i, j]
            air_time  = d_ih.loc[j].min() / UAV_SPEED_KMH * 60
            a[i, j]   = 1 if road_time + air_time <= T else 0
    return a

def run_model(a, demand_coords, candidate_coords, alpha, cost):
    I = list(demand_coords.keys())
    J = list(candidate_coords.keys())

    mod = ConcreteModel()
    mod.I = Set(initialize=I)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.J, domain=Binary)
    mod.z = Var(mod.I, domain=Binary)

    # Objective now includes a soft bushfire-risk penalty alongside site count
    mod.obj = Objective(
        expr=sum(mod.y[j] for j in J) + sum(cost[j] * mod.y[j] for j in J),
        sense=minimize
    )

    mod.cov = Constraint(mod.I, rule=lambda mod, i:
        mod.z[i] <= sum(a[i, j] * mod.y[j] for j in J))

    mod.demand = Constraint(expr=
        sum(mod.z[i] for i in I) >= floor(alpha * len(I)))

    result = SolverFactory('cbc').solve(mod)

    if result.solver.termination_condition == TerminationCondition.optimal:
        selected  = [j for j in J if value(mod.y[j]) > 0.5]
        covered   = [i for i in I if value(mod.z[i]) > 0.5]
        uncovered = [i for i in I if value(mod.z[i]) < 0.5]
        return 'optimal', selected, covered, uncovered
    else:
        return 'infeasible', [], [], []

#--------------------
# MAIN MODEL
#--------------------

a = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)

pop_served = {j: population_served(j, a) for j in candidate_coords}
cost = {j: bushfire_cost(j, a, pop_served) for j in candidate_coords}

# sanity check
coverable = [i for i in demand_coords if any(a[i, j] == 1 for j in candidate_coords)]
print(f"Max achievable coverage: {len(coverable)}/{len(demand_coords)} = {len(coverable)/len(demand_coords)*100:.1f}%")
print(f"Requested coverage: {COVERAGE_THRESHOLD*100}%")

status, selected, covered, uncovered = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost)

#--------------------
# RESULTS
#--------------------

print(f"\nTime threshold      = {TIME_THRESHOLD} min")
print(f"Vehicle speed        = {UAV_SPEED_KMH} km/h")
print(f"Coverage target      = {COVERAGE_THRESHOLD*100}%")
print(f"Bushfire penalty wt  = {BUSHFIRE_PENALTY_WEIGHT}")
print(f"Emergency time thresh = {EMERGENCY_TIME_THRESHOLD} min")
print(f"Emergency value per 1,000 pop reached = {EMERGENCY_VALUE_PER_1000_POP}\n")

if status == 'optimal':
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites     :")
    header = f"    {'Site':<45} {'Bushfire':<22} {'Pop served':>14} {'Emergency reach':>18} {'Net cost':>11}"
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for port in selected:
        label = dfb.loc[port, 'bfpl_category_label'] if port in dfb.index else 'unknown'
        reach_pop = emergency_reach_population(port, a, pop_served)
        print(f"    {port:<45} {label:<22} {pop_served[port]:>14,.0f} {reach_pop:>18,.0f} {cost[port]:>11.3f}")
    print(f"\nRegions covered    : {len(covered)} / {len(demand_coords)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg)
else:
    print("INFEASIBLE — lower COVERAGE_THRESHOLD or increase TIME_THRESHOLD")

#--------------------
# SENSITIVITY ANALYSIS
#--------------------
'''
time_thresholds  = [30, 45, 60, 75, 90, 105, 120]
coverage_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]

results = {}

for T in time_thresholds:
    a_temp = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T)
    pop_served_temp = {j: population_served(j, a_temp) for j in candidate_coords}
    cost_temp = {j: bushfire_cost(j, a_temp, pop_served_temp) for j in candidate_coords}
    for alpha in coverage_targets:
        s, sel, cov, uncov = run_model(a_temp, demand_coords, candidate_coords, alpha, cost_temp)
        if s == 'optimal':
            results[T, alpha] = len(sel)
            print(f"T={T:3d} min | coverage={alpha*100:.0f}% | vertiports={len(sel)}")
        else:
            results[T, alpha] = None
            print(f"T={T:3d} min | coverage={alpha*100:.0f}% | INFEASIBLE")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
for alpha in coverage_targets:
    y_vals = [results[T, alpha] if results[T, alpha] is not None else np.nan for T in time_thresholds]
    ax1.plot(time_thresholds, y_vals, marker='o', label=f'{alpha*100:.0f}%')
    for T, v in zip(time_thresholds, y_vals):
        if not np.isnan(v):
            ax1.annotate(str(int(v)), (T, v), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8)

ax1.set_xlabel('Time Threshold (minutes)')
ax1.set_ylabel('Minimum Vertiports Required')
ax1.set_title('Vertiports vs Time Threshold')
ax1.legend(title='Coverage Target')
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = axes[1]
for T in time_thresholds:
    y_vals = [results[T, alpha] if results[T, alpha] is not None else np.nan for alpha in coverage_targets]
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
plt.savefig('sensitivity_analysis_bushfire_risk.png', dpi=150)
plt.show()
print("Plot saved to sensitivity_analysis_bushfire_risk.png")

for alpha in coverage_targets:
    vals = [(T, results[T, alpha]) for T in time_thresholds if results[T, alpha] is not None]
    if len(vals) < 2:
        continue
    best_efficiency = 0
    best_T = None
    for k in range(1, len(vals)):
        T_prev, n_prev = vals[k-1]
        T_curr, n_curr = vals[k]
        drop       = n_prev - n_curr
        time_cost  = T_curr - T_prev
        efficiency = drop / time_cost
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_T = T_curr
    print(f"Coverage {alpha*100:.0f}%: biggest gain at T={best_T} min ({best_efficiency:.2f} vertiports saved per minute)")
'''
#--------------------
# MAP
#--------------------
m_map = folium.Map(
    location=[-33.0, 151.0],
    zoom_start=7,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri, HERE, Garmin, FAO, NOAA, USGS'
)

m_map.get_root().html.add_child(folium.Element('''
    <style>.leaflet-interactive:focus { outline: none !important; }</style>
'''))

# high-risk (Category 1) vertiports -- shown first so other markers layer on top
for j in high_risk_vertiports:
    lat, lon = candidate_coords[j]
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        color='darkred',
        fill=True,
        fill_opacity=0.4,
        weight=1,
        popup=folium.Popup(f"{j} (Category 1 - high bushfire risk)", max_width=220),
        tooltip=f"{j} (Category 1)"
    ).add_to(m_map)

# selected vertiports - blue
for j in selected:
    lat, lon = candidate_coords[j]
    label = dfb.loc[j, 'bfpl_category_label'] if j in dfb.index else 'unknown'
    reach_pop = emergency_reach_population(j, a, pop_served)
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color='blue',
        fill=True,
        fill_opacity=1,
        popup=folium.Popup(
            f"{j}<br>Bushfire: {label}<br>Pop served: {pop_served[j]:,.0f}"
            f"<br>Emergency reach pop: {reach_pop:,.0f}<br>Net cost: {cost[j]:.3f}",
            max_width=240),
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

# uncovered SA2s - orange
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

# reachable hospitals - green
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

legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 240px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
<p style="margin: 5px 0;"><i style="background:blue; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Selected Vertiports</p>
<p style="margin: 5px 0;"><i style="background:red; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Covered Regions (SA2)</p>
<p style="margin: 5px 0;"><i style="background:orange; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Uncovered Regions (SA2)</p>
<p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Reachable Hospitals</p>
<p style="margin: 5px 0;"><i style="background:darkred; opacity:0.4; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> High Bushfire Risk (Category 1)</p>
</div>
'''
m_map.get_root().html.add_child(folium.Element(legend_html))
m_map.save('vertiports_map_bushfire_risk.html')
print("Map saved to vertiports_map_bushfire_risk.html")
