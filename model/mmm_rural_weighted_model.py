"""
Objective:
    minimise  Σ y_j  +  Σ (w_bush * bushfire_cost_j
                          + w_wind * wind_cost_j
                          + w_rain * rain_cost_j) * y_j

Coverage requirement (rural-weighted):
    Σ w_i * z_i  >=  alpha * Σ w_i
    where w_i = RURAL_PRIORITY (1.5) if SA2 i is rural, else 1.0

    An SA2 is rural if its Modified Monash Model (MMM) category is RURAL_MMM_MIN
    or higher. The category (1-7) is read from an 'mmm' column in
    sa2_names_with_coord_mmm.xlsx.
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import requests
import time
import folium
from folium.plugins import MeasureControl
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# LOAD DATA SETS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

dfi = pd.read_excel(BASE_DIR / 'vertiport_name_with_coord.xlsx')
dfj = pd.read_excel(BASE_DIR / 'sa2_names_with_coord_mmm.xlsx')
dfh = pd.read_excel(BASE_DIR / 'Emergency_hospitals_with_lat_lon.xlsx')

df_bushfire = pd.read_csv(BASE_DIR / 'Parameter-Bushfire' / 'vertiport_bushfire_risk.csv')
df_rain = pd.read_excel(BASE_DIR / 'Parameter-Rain' / 'vertiport_rain.xlsx').set_index('name')
df_wind = pd.read_csv(BASE_DIR / 'Parameter-Wind' / 'vertiport_wind_risk.csv').set_index('vertiport')

candidate_coords  = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords     = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords   = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))
demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))

candidate_coords  = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords     = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords   = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))
demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 90
COVERAGE_THRESHOLD = 0.85

# --- rural priority settings ---
# Modified Monash Model categories (Australian Department of Health):
#   MM1 metropolitan        MM2 regional centres     MM3 large rural towns
#   MM4 medium rural towns  MM5 small rural towns    MM6 remote    MM7 very remote
# An SA2 is rural if its category is RURAL_MMM_MIN or higher. A rural SA2 counts
# RURAL_PRIORITY times as much as a non-rural SA2 towards the coverage target.
# Set RURAL_PRIORITY to 1.0 to recover the original (unweighted) model.
RURAL_MMM_MIN  = 3
RURAL_PRIORITY = 1.5

# Re-solve with the rural cut-off at each of these MMM categories to see whether
# the chosen sites depend on where the line is drawn. Set to [] to skip.
MMM_CUTOFF_CHECK = [3, 4, 5]

# --- risk penalty weights ---
WEIGHT_BUSHFIRE = 1.0
WEIGHT_WIND     = 1.0
WEIGHT_RAIN     = 1.0

# --- bushfire settings ---
BUSHFIRE_PENALTY_WEIGHT      = 3.0
EMERGENCY_TIME_THRESHOLD     = 60 #threshold in minutes for a vertiport to reach a high-risk bushfire area during an emergency scenario
EMERGENCY_VALUE_PER_1000_POP = 0.4 #weight for incentive for vertiports near high-risk bushfire areas, to aid during emergency scenarios

BFPL_SEVERITY = {
    'Vegetation Category 1': 1.0,
    'Vegetation Category 2': 0.3,
    'Vegetation Category 3': 0.6,
    'Vegetation Buffer':     0.15,
    'Not bush fire prone':   0.0,
}

# --- wind settings ---
WIND_THRESHOLD_KT            = 20.0
WIND_PENALTY_WEIGHT          = 3.0
ALTERNATE_TIME_THRESHOLD     = 30
ALTERNATE_VALUE_PER_1000_POP = 0.4
SAFE_ALTERNATE_PCT_LIMIT     = 10.0

# --- rain settings ---
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
# RURAL CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def load_mmm(dfj):
    """{SA2_Name: MMM category 1-7}, read from the 'mmm' column of the SA2 sheet."""
    cols = {str(c).strip().lower(): c for c in dfj.columns}
    col  = next((cols[k] for k in ('mmm', 'mmm_2023', 'mmm_code', 'mm') if k in cols), None)
    if col is None:
        raise KeyError("sa2_names_with_coord_mmm.xlsx needs an 'mmm' column holding each SA2's "
                       "Modified Monash Model category (1-7).")
    def parse(v):                       # accepts 5, 5.0, '5', 'MM5', 'MM 5'
        digits = ''.join(ch for ch in str(v).split('.')[0] if ch.isdigit())
        return int(digits) if digits else None
    mmm = {i: parse(v) for i, v in zip(dfj['SA2_Name'], dfj[col])}
    bad = [i for i, v in mmm.items() if v not in range(1, 8)]
    if bad:
        raise ValueError(f"{len(bad)} SA2(s) have a missing or invalid MMM category (must be 1-7): {bad[:10]}")
    return mmm

sa2_mmm = load_mmm(dfj)

def rural_flags(mmm_min):
    """{SA2_Name: True if MMM category is mmm_min or higher}"""
    return {i: sa2_mmm[i] >= mmm_min for i in demand_coords}

def region_weights(is_rural, priority):
    """{SA2_Name: coverage weight}"""
    return {i: (priority if is_rural[i] else 1.0) for i in is_rural}

is_rural      = rural_flags(RURAL_MMM_MIN)
region_weight = region_weights(is_rural, RURAL_PRIORITY)
rural_regions = [i for i in demand_coords if is_rural[i]]
other_regions = [i for i in demand_coords if not is_rural[i]]

# ─────────────────────────────────────────────────────────────
# LOAD MATRICES
# ─────────────────────────────────────────────────────────────

d_ih = pd.read_excel(BASE_DIR / 'vertiport_hospital_distances.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel(BASE_DIR / 'd_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min']
                    for _, row in d_ij.iterrows()}

# ─────────────────────────────────────────────────────────────
# WIND SETUP
# ─────────────────────────────────────────────────────────────

safe_vertiports = [
    j for j in candidate_coords
    if j in df_wind.index
    and not np.isnan(df_wind.loc[j, 'pct_above_threshold'])
    and df_wind.loc[j, 'pct_above_threshold'] <= SAFE_ALTERNATE_PCT_LIMIT
]

# ─────────────────────────────────────────────────────────────
# BUSHFIRE SETUP
# ─────────────────────────────────────────────────────────────
# clean up bushfire labels
df_bushfire['bfpl_category_label'] = df_bushfire['bfpl_category_label'].astype(str).str.strip()
df_bushfire = df_bushfire.dropna(subset=['name']).drop_duplicates(subset=['name']).set_index('name')


high_risk_vertiports = [
    j for j in candidate_coords
    if j in df_bushfire.index and df_bushfire.loc[j, 'bfpl_category_label'] == 'Vegetation Category 1'
]

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
# RISK COST FUNCTIONS
# ─────────────────────────────────────────────────────────────

def population_served(j, a):
    return sum(demand_population[i] for i in demand_coords if a[i, j] == 1)

# --- bushfire ---
def bfpl_severity(j):
    if j not in df_bushfire.index:
        return 0.0
    return BFPL_SEVERITY.get(df_bushfire.loc[j, 'bfpl_category_label'], 0.0)

def emergency_reach_population(j, a, pop_served):
    j_lat, j_lon = candidate_coords[j]
    total_pop = 0
    for target in high_risk_vertiports:
        if target == j:
            continue
        t_lat, t_lon = candidate_coords[target]
        if haversine_km(j_lat, j_lon, t_lat, t_lon) / UAV_SPEED_KMH * 60 <= EMERGENCY_TIME_THRESHOLD:
            total_pop += pop_served[target]
    return total_pop

def bushfire_cost(j, a, pop_served):
    penalty = BUSHFIRE_PENALTY_WEIGHT * bfpl_severity(j)
    incentive   = EMERGENCY_VALUE_PER_1000_POP * (emergency_reach_population(j, a, pop_served) / 1000) # reduces penalty if the port is near a high-risk area, for aid during emergency scenarios
    return max(0.0, penalty - incentive)

# --- wind ---
def wind_severity_01(j):
    if j not in df_wind.index:
        return 1.0
    pct = df_wind.loc[j, 'pct_above_threshold']
    return 1.0 if np.isnan(pct) else min(pct / 100.0, 1.0)

def alternate_reach_population(j, a, pop_served):
    j_lat, j_lon = candidate_coords[j]
    best_pop = 0
    for target in safe_vertiports:
        if target == j:
            continue
        t_lat, t_lon = candidate_coords[target]
        if haversine_km(j_lat, j_lon, t_lat, t_lon) / UAV_SPEED_KMH * 60 <= ALTERNATE_TIME_THRESHOLD:
            best_pop = max(best_pop, pop_served[target])
    return best_pop

def wind_cost(j, a, pop_served):
    penalty = WIND_PENALTY_WEIGHT * wind_severity_01(j)
    incentive   = ALTERNATE_VALUE_PER_1000_POP * (alternate_reach_population(j, a, pop_served) / 1000)
    return max(0.0, penalty - incentive)

# --- rain ---
df_rain = pd.read_excel(BASE_DIR / 'Parameter-Rain' / 'vertiport_rain.xlsx').copy()
if 'name' in df_rain.columns:
    df_rain = df_rain.rename(columns={'name': 'vertiport'})
df_rain = df_rain.drop_duplicates(subset=['vertiport']).set_index('vertiport')


def rain_severity_01(j):
    if j not in df_rain.index:
        return 0.0
    row = df_rain.loc[j]
    avg_rain = float(row['Avg_Rainfall'])
    heavy_days = float(row['Heavy_Rain_Days'])
    rain_norm = min(avg_rain / RAIN_MAX_AVG_MM_DAY, 1.0)
    heavy_norm = min(heavy_days / RAIN_MAX_HEAVY_DAYS, 1.0)
    return (rain_norm + heavy_norm) / 2.0


def rain_cost(j):
    return RAIN_PENALTY_WEIGHT * rain_severity_01(j)

# --- combined ---
def combined_cost(j, a, pop_served):
    return (WEIGHT_BUSHFIRE * bushfire_cost(j, a, pop_served) +
            WEIGHT_WIND     * wind_cost(j, a, pop_served)     +
            WEIGHT_RAIN     * rain_cost(j))

# ─────────────────────────────────────────────────────────────
# PYOMO MODEL
# ─────────────────────────────────────────────────────────────

def run_model(a, demand_coords, candidate_coords, alpha, cost, weight):
    """
    weight[i] is the coverage weight of SA2 region i (RURAL_PRIORITY for rural
    regions, 1.0 otherwise). The coverage target alpha applies to the weighted
    total, so each rural region covered counts 1.5x towards meeting it.
    """
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

    total_weight = sum(weight[i] for i in I)
    mod.demand = Constraint(expr=
        sum(weight[i] * mod.z[i] for i in I) >= floor(alpha * total_weight))

    result = SolverFactory('cbc').solve(mod)

    if result.solver.termination_condition == TerminationCondition.optimal:
        selected  = [j for j in J if value(mod.y[j]) > 0.5]
        # Coverage is read off the selected sites rather than z, because the
        # solver may leave z[i] = 0 for a region that is in fact covered.
        covered   = [i for i in I if any(a[i, j] == 1 for j in selected)]
        uncovered = [i for i in I if i not in set(covered)]
        return 'optimal', selected, covered, uncovered
    else:
        return 'infeasible', [], [], []

# ─────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────

a = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)

pop_served = {j: population_served(j, a) for j in candidate_coords}
cost       = {j: combined_cost(j, a, pop_served) for j in candidate_coords}

total_weight     = sum(region_weight.values())
coverable        = [i for i in demand_coords if any(a[i, j] == 1 for j in candidate_coords)]
coverable_weight = sum(region_weight[i] for i in coverable)

print(f"\nSA2 regions: {len(rural_regions)} rural (MM{RURAL_MMM_MIN}-MM7, weight {RURAL_PRIORITY}), "
      f"{len(other_regions)} non-rural (weight 1.0)")
print("SA2s by MMM category: " + ", ".join(f"MM{k}: {sum(1 for v in sa2_mmm.values() if v == k)}" for k in range(1, 8)))
print(f"Max achievable coverage: {len(coverable)}/{len(demand_coords)} regions = {len(coverable)/len(demand_coords)*100:.1f}%")
print(f"Max achievable weighted coverage: {coverable_weight/total_weight*100:.1f}%")
print(f"Requested weighted coverage: {COVERAGE_THRESHOLD*100}%")

status, selected, covered, uncovered = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost, region_weight)

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────

print(f"\nTime threshold  = {TIME_THRESHOLD} min")
print(f"Vehicle speed   = {UAV_SPEED_KMH} km/h")
print(f"Coverage target = {COVERAGE_THRESHOLD*100}% (rural-weighted, rural priority = {RURAL_PRIORITY}x)")
print(f"Risk weights    = bushfire:{WEIGHT_BUSHFIRE} wind:{WEIGHT_WIND} rain:{WEIGHT_RAIN}\n")

if status == 'optimal':
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites:")
    header = f"  {'Site':<45} {'Bushfire':>10} {'Wind':>10} {'Rain':>10} {'Total':>10}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for port in selected:
        bc = WEIGHT_BUSHFIRE * bushfire_cost(port, a, pop_served)
        wc = WEIGHT_WIND     * wind_cost(port, a, pop_served)
        rc = WEIGHT_RAIN     * rain_cost(port)
        print(f"  {port:<45} {bc:>10.3f} {wc:>10.3f} {rc:>10.3f} {cost[port]:>10.3f}")

    rural_covered  = [i for i in covered if is_rural[i]]
    other_covered  = [i for i in covered if not is_rural[i]]
    covered_weight = sum(region_weight[i] for i in covered)
    pct = lambda n, d: f"{n/d*100:.1f}%" if d else "n/a"

    print(f"\nRegions covered    : {len(covered)} / {len(demand_coords)} ({pct(len(covered), len(demand_coords))})")
    print(f"  Rural            : {len(rural_covered)} / {len(rural_regions)} ({pct(len(rural_covered), len(rural_regions))})")
    print(f"  Non-rural        : {len(other_covered)} / {len(other_regions)} ({pct(len(other_covered), len(other_regions))})")
    print(f"Weighted coverage  : {pct(covered_weight, total_weight)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg, f"(MM{sa2_mmm[reg]}{', rural' if is_rural[reg] else ''})")
else:
    print("INFEASIBLE — lower COVERAGE_THRESHOLD or increase TIME_THRESHOLD")

# ─────────────────────────────────────────────────────────────
# RURAL CUT-OFF CHECK
# ─────────────────────────────────────────────────────────────
# Re-solves with the rural line at each MMM category in MMM_CUTOFF_CHECK, plus
# the unweighted model, and reports whether the chosen sites change.

if MMM_CUTOFF_CHECK and status == 'optimal':
    print("\nRural cut-off check:")
    header = f"  {'Rural if MMM is':<20} {'Rural SA2s':>10} {'Vertiports':>11} {'Rural covered':>14} {'Other covered':>14}   Sites vs main run"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    scenarios = [('no rural priority', None)] + [(f"MM{k} or higher", k) for k in MMM_CUTOFF_CHECK]
    for label, k in scenarios:
        flags_k  = rural_flags(k) if k is not None else {i: False for i in demand_coords}
        weight_k = region_weights(flags_k, RURAL_PRIORITY)
        s, sel, cov, uncov = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost, weight_k)
        if s != 'optimal':
            print(f"  {label:<20} {sum(flags_k.values()):>10} {'INFEASIBLE':>11}")
            continue
        # rural/other covered counts use the main run's classification so rows are comparable
        r_cov = sum(1 for i in cov if is_rural[i])
        o_cov = sum(1 for i in cov if not is_rural[i])
        added   = sorted(set(sel) - set(selected))
        dropped = sorted(set(selected) - set(sel))
        diff = 'same' if not added and not dropped else f"+{added} -{dropped}"
        print(f"  {label:<20} {sum(flags_k.values()):>10} {len(sel):>11} "
              f"{f'{r_cov}/{len(rural_regions)}':>14} {f'{o_cov}/{len(other_regions)}':>14}   {diff}")

# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
'''
time_thresholds  = [30, 45, 60, 75, 90, 105, 120]
coverage_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = {}

for T in time_thresholds:
    a_temp          = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T)
    pop_served_temp = {j: population_served(j, a_temp) for j in candidate_coords}
    cost_temp       = {j: combined_cost(j, a_temp, pop_served_temp) for j in candidate_coords}
    for alpha in coverage_targets:
        s, sel, cov, uncov = run_model(a_temp, demand_coords, candidate_coords, alpha, cost_temp, region_weight)
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

    MeasureControl(
    position='bottomleft',
    primaryLengthUnit='kilometers',
    secondaryLengthUnit='meters',
    activeColor='orange',
    completedColor='green'
    ).add_to(m_map)

    m_map.get_root().html.add_child(folium.Element('''
        <style>.leaflet-interactive:focus { outline: none !important; }</style>
    '''))

    region_label = lambda i: f"{i} (MM{sa2_mmm[i]}{', rural' if is_rural[i] else ''})"

    for i in selected:
        lat, lon = candidate_coords[i]
        folium.CircleMarker(
            location=[lat, lon], radius=6, color='blue', fill=True, fill_opacity=1,
            popup=folium.Popup(i, max_width=200), tooltip=i
        ).add_to(m_map)

    for i in covered:
        lat, lon = demand_coords[i]
        folium.CircleMarker(
            location=[lat, lon], radius=4, color='red', fill=True, fill_opacity=1,
            popup=folium.Popup(region_label(i), max_width=200), tooltip=region_label(i)
        ).add_to(m_map)

    for i in uncovered:
        lat, lon = demand_coords[i]
        folium.CircleMarker(
            location=[lat, lon], radius=4, color='orange', fill=True, fill_opacity=1,
            popup=folium.Popup(region_label(i), max_width=200), tooltip=region_label(i)
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
    <p style="margin: 5px 0;"><i style="background:blue; width: 12px; height: 12px; 
       display:inline-block; border-radius: 50%;"></i> Selected Vertiports</p>
    <p style="margin: 5px 0;"><i style="background:red; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Covered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:orange; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Uncovered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Reachable Hospitals</p>
    </div>
    '''
    m_map.get_root().html.add_child(folium.Element(legend_html))
    m_map.save(BASE_DIR / 'vertiports_map_mmm.html')
print(f"Map saved to {BASE_DIR / 'vertiports_map_mmm.html'}")