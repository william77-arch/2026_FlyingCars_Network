"""
Combined vertiport placement model incorporating operational risk:
  - Bushfire risk  — based on BFPL vegetation category
  - Wind risk      — based on % of observations exceeding threshold
  - Rain risk      — based on avg daily rainfall and heavy rain days

Objective:
    minimise  Σ y_j  +  Σ (w_bush * bushfire_cost_j
                          + w_wind * wind_cost_j
                          + w_rain * rain_cost_j) * y_j
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import folium
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# LOAD DATASETS
# ─────────────────────────────────────────────────────────────

dfi = pd.read_excel('vertiport_name_with_coord.xlsx')
dfj = pd.read_excel('sa2_names_with_coord.xlsx')
dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')

candidate_coords = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords    = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords  = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))

# bushfire
df_bushfire = pd.read_csv('vertiport_bushfire_risk.csv')
df_bushfire['bfpl_category_label'] = df_bushfire['bfpl_category_label'].astype(str).str.strip()
df_bushfire = df_bushfire.dropna(subset=['name']).drop_duplicates(subset=['name'])
bushfire_label_lookup = dict(zip(df_bushfire['name'], df_bushfire['bfpl_category_label']))

# wind 
df_wind = pd.read_csv('vertiport_wind_risk.csv').drop_duplicates(subset=['vertiport'])
wind_pct_lookup = dict(zip(df_wind['vertiport'], df_wind['pct_above_threshold']))

# rain
df_rain = pd.read_excel('vertiport_rain.xlsx').drop_duplicates(subset=['name'])
rain_avg_lookup   = dict(zip(df_rain['name'], df_rain['Avg_Rainfall']))
rain_heavy_lookup = dict(zip(df_rain['name'], df_rain['Heavy_Rain_Days']))

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 90
COVERAGE_THRESHOLD = 0.85

# risk penalty weights
BUSHFIRE_PENALTY = 1.0
WIND_PENALTY     = 1.0
RAIN_PENALTY     = 1.0

# bushfire
BUSHFIRE_PENALTY_WEIGHT = 3.0
BFPL_SEVERITY = {
    'Vegetation Category 1': 1.0,
    'Vegetation Category 2': 0.6,
    'Vegetation Category 3': 0.3,
    'Vegetation Buffer':     0.15,
    'Not bush fire prone':   0.0,
}

# wind
WIND_PENALTY_WEIGHT = 3.0

# rain
RAIN_MAX_AVG_MM_DAY  = 5.0
RAIN_MAX_HEAVY_DAYS  = 40
RAIN_PENALTY_WEIGHT  = 3.0

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
# LOAD MATRICES
# ─────────────────────────────────────────────────────────────

d_ih = pd.read_excel('vertiport_hospital_distances.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel('d_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min']
                    for _, row in d_ij.iterrows()}

# ─────────────────────────────────────────────────────────────
# RISK COST FUNCTIONS 
# ─────────────────────────────────────────────────────────────

def bushfire_cost(j):
    label = bushfire_label_lookup.get(j, 'Not bush fire prone')
    return BUSHFIRE_PENALTY_WEIGHT * BFPL_SEVERITY.get(label, 0.0)

def wind_cost(j):
    pct = wind_pct_lookup.get(j, None)
    if pct is None or np.isnan(float(pct)):
        return WIND_PENALTY_WEIGHT   # unknown = assume worst
    return WIND_PENALTY_WEIGHT * min(float(pct) / 100.0, 1.0)

def rain_cost(j):
    if j not in rain_avg_lookup:
        return 0.0
    avg_rain   = float(rain_avg_lookup[j])
    heavy_days = float(rain_heavy_lookup[j])
    rain_norm  = min(avg_rain   / RAIN_MAX_AVG_MM_DAY, 1.0)
    heavy_norm = min(heavy_days / RAIN_MAX_HEAVY_DAYS, 1.0)
    return RAIN_PENALTY_WEIGHT * (rain_norm + heavy_norm) / 2.0

def combined_cost(j):
    return (BUSHFIRE_PENALTY * bushfire_cost(j) +
            WIND_PENALTY     * wind_cost(j)     +
            RAIN_PENALTY     * rain_cost(j))

# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX
# ─────────────────────────────────────────────────────────────

def build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T):
    a = {}
    for i in demand_coords:
        for j in candidate_coords:
            road_time = road_time_lookup[i, j]
            air_time  = d_ih.loc[j].min() / UAV_SPEED_KMH * 60
            a[i, j]   = 1 if road_time + air_time <= T else 0
    return a

# ─────────────────────────────────────────────────────────────
# PYOMO MODEL
# ─────────────────────────────────────────────────────────────

def run_model(a, demand_coords, candidate_coords, alpha, cost):
    I = list(demand_coords.keys())
    J = list(candidate_coords.keys())

    mod = ConcreteModel()
    mod.I = Set(initialize=I)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.J, domain=Binary)
    mod.z = Var(mod.I, domain=Binary)

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

# ─────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────

a    = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)
cost = {j: combined_cost(j) for j in candidate_coords}

coverable = [i for i in demand_coords if any(a[i, j] == 1 for j in candidate_coords)]
print(f"\nMax achievable coverage: {len(coverable)}/{len(demand_coords)} = {len(coverable)/len(demand_coords)*100:.1f}%")
print(f"Requested coverage: {COVERAGE_THRESHOLD*100}%")

status, selected, covered, uncovered = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost)

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────

print(f"\nTime threshold  = {TIME_THRESHOLD} min")
print(f"Vehicle speed   = {UAV_SPEED_KMH} km/h")
print(f"Coverage target = {COVERAGE_THRESHOLD*100}%")
print(f"Risk weights    = bushfire:{BUSHFIRE_PENALTY} wind:{WIND_PENALTY} rain:{RAIN_PENALTY}\n")

if status == 'optimal':
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites:")
    header = f"  {'Site':<45} {'Bushfire':>10} {'Wind':>10} {'Rain':>10} {'Total':>10}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for port in selected:
        bc = BUSHFIRE_PENALTY * bushfire_cost(port)
        wc = WIND_PENALTY     * wind_cost(port)
        rc = RAIN_PENALTY     * rain_cost(port)
        print(f"  {port:<45} {bc:>10.3f} {wc:>10.3f} {rc:>10.3f} {cost[port]:>10.3f}")
    print(f"\nRegions covered    : {len(covered)} / {len(demand_coords)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg)
else:
    print("INFEASIBLE — lower COVERAGE_THRESHOLD or increase TIME_THRESHOLD")

# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
'''
time_thresholds  = [30, 45, 60, 75, 90, 105, 120]
coverage_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = {}

for T in time_thresholds:
    a_temp    = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T)
    for alpha in coverage_targets:
        s, sel, cov, uncov = run_model(a_temp, demand_coords, candidate_coords, alpha, cost)
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
plt.savefig('sensitivity_analysis_combined.png', dpi=150)
plt.show()
print("Plot saved to sensitivity_analysis_combined.png")
'''

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────

if status == 'optimal':
    m_map = folium.Map(
        location=[-33.0, 151.0],
        zoom_start=7,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri, HERE, Garmin, FAO, NOAA, USGS'
    )

    m_map.get_root().html.add_child(folium.Element('''
        <style>.leaflet-interactive:focus { outline: none !important; }</style>
    '''))

    for j in selected:
        lat, lon = candidate_coords[j]
        bc = BUSHFIRE_PENALTY * bushfire_cost(j)
        wc = WIND_PENALTY     * wind_cost(j)
        rc = RAIN_PENALTY     * rain_cost(j)

        dominant = max([('bushfire', bc), ('wind', wc), ('rain', rc)], key=lambda x: x[1])[0]
        colour   = {'bushfire': 'darkred', 'wind': 'purple', 'rain': 'darkblue'}.get(dominant, 'blue')

        bfpl     = bushfire_label_lookup.get(j, 'unknown')
        wind_pct = wind_pct_lookup.get(j, float('nan'))
        rain_avg = rain_avg_lookup.get(j, float('nan'))
        heavy    = rain_heavy_lookup.get(j, float('nan'))

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=colour,
            fill=True,
            fill_opacity=1,
            popup=folium.Popup(
                f"<b>{j}</b><br>"
                f"Bushfire: {bfpl} (cost: {bc:.3f})<br>"
                f"Wind: {wind_pct:.1f}% > threshold (cost: {wc:.3f})<br>"
                f"Rain: {rain_avg:.2f} mm/day, {heavy:.0f} heavy days (cost: {rc:.3f})<br>"
                f"Total risk cost: {cost[j]:.3f}",
                max_width=280),
            tooltip=j
        ).add_to(m_map)

    for i in covered:
        lat, lon = demand_coords[i]
        folium.CircleMarker(
            location=[lat, lon], radius=4, color='red', fill=True, fill_opacity=1,
            popup=folium.Popup(i, max_width=200), tooltip=i
        ).add_to(m_map)

    for i in uncovered:
        lat, lon = demand_coords[i]
        folium.CircleMarker(
            location=[lat, lon], radius=4, color='orange', fill=True, fill_opacity=1,
            popup=folium.Popup(i, max_width=200), tooltip=i
        ).add_to(m_map)

    for h, (h_lat, h_lon) in hospital_coords.items():
        reachable = any(d_ih.loc[j, h] / UAV_SPEED_KMH * 60 <= TIME_THRESHOLD for j in selected)
        if reachable:
            folium.CircleMarker(
                location=[h_lat, h_lon], radius=6, color='green', fill=True, fill_opacity=1,
                popup=folium.Popup(h, max_width=200), tooltip=h
            ).add_to(m_map)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 260px;
         background-color: white; border:2px solid grey; z-index:9999;
         font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
    <p style="margin: 5px 0;"><i style="background:darkred; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Vertiport — dominant bushfire risk</p>
    <p style="margin: 5px 0;"><i style="background:purple; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Vertiport — dominant wind risk</p>
    <p style="margin: 5px 0;"><i style="background:darkblue; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Vertiport — dominant rain risk</p>
    <p style="margin: 5px 0;"><i style="background:red; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Covered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:orange; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Uncovered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Reachable Hospitals</p>
    </div>
    '''
    m_map.get_root().html.add_child(folium.Element(legend_html))
    m_map.save('vertiports_map.html')
    print("Map saved to vertiports_map.html")