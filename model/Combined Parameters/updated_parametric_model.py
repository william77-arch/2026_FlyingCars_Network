"""
Objective:
    minimise  Σ y_j  +  RISK_LAMBDA * Σ risk_j * y_j

    risk_j = (w_bush * bushfire_j + w_wind * wind_j + w_rain * rain_j) / (w_bush + w_wind + w_rain)
    Each hazard is normalised to 0-1 across the candidate sites, so risk_j is on a 0-1 scale.

    risk saved > 1 / λ
"""

import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, ceil
import folium
from folium.plugins import MeasureControl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point


# ─────────────────────────────────────────────────────────────
# LOAD DATA SETS
# ─────────────────────────────────────────────────────────────
dfi = pd.read_excel('vertiport_name_with_coord.xlsx')
dfj = pd.read_excel('sa2_names_with_coord.xlsx')
dfh = pd.read_excel('Emergency_only_hospitals_with_lat_lon.xlsx')

df_bushfire = pd.read_csv('vertiport_bushfire_risk.csv')
df_rain = pd.read_excel('vertiport_rain.xlsx')
df_wind = pd.read_csv('vertiport_wind_risk.csv').set_index('vertiport')

candidate_coords  = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords     = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords   = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))
demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 60
COVERAGE_THRESHOLD = 0.95

# How strongly risk trades against vertiport count.
# Higher = accept more vertiports to avoid risky sites.
RISK_LAMBDA = 0.5

# --- risk weights ---
WEIGHT_BUSHFIRE = 3.0
WEIGHT_WIND     = 5.0
WEIGHT_RAIN     = 1.0
W_SUM = WEIGHT_BUSHFIRE + WEIGHT_WIND + WEIGHT_RAIN

MISSING_DATA_SEVERITY = 1.0

# --- bushfire settings ---
EMERGENCY_TIME_THRESHOLD = 15    # threshold in minutes for a vertiport to reach a high-risk bushfire vertiport during an emergency
EMERGENCY_VALUE_PER_SITE = 0.1   # share of bushfire risk removed per high-risk bushfire vertiport within reach
MAX_BUSHFIRE_DISCOUNT    = 0.5   # discounts can never remove more than this share of a site's bushfire risk

BFPL_SEVERITY = {
    'Vegetation Category 1': 1.0,
    'Vegetation Category 2': 0.3,
    'Vegetation Category 3': 0.6,   # weights based on BFPL category level
    'Vegetation Buffer':     0.15,
    'Not bush fire prone':   0.0,
}

# --- wind settings ---
ALTERNATE_TIME_THRESHOLD = 30    # maximum time allowed in minutes for flight to reach a safe alternate vertiport
SAFE_ALTERNATE_PCT_LIMIT = 10.0  # maximum percentage of winds speeds above threshold for a vertiport to be considered safe
WIND_BACKUP_DISCOUNT     = 0.5   # share of a site's wind risk removed if a safe alternate exists within reach

# --- rain settings ---
RAIN_MAX_AVG_MM_DAY  = 5.0   # rain threshold in mm per day
RAIN_MAX_HEAVY_DAYS  = 40    # rain threshold in days per year

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

def flight_min(j, k) -> float:
    return haversine_km(*candidate_coords[j], *candidate_coords[k]) / UAV_SPEED_KMH * 60

# ─────────────────────────────────────────────────────────────
# LOAD MATRICES
# ─────────────────────────────────────────────────────────────

dfh = pd.read_excel('Emergency_only_hospitals_with_lat_lon.xlsx')

d_ih = pd.read_excel('vertiport_hospital_distances_emergency.xlsx')
d_ih = d_ih.set_index('vertiport')
d_ij = pd.read_excel('d_ij.xlsx')

road_time_lookup = {(row['sa2'], row['vertiport']): row['time_min']
                    for _, row in d_ij.iterrows()}

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
# safe vertiport if wind risk is below threshold and data is available
safe_vertiports = [
    j for j in candidate_coords
    if j in df_wind.index
    and not np.isnan(df_wind.loc[j, 'pct_above_threshold'])
    and df_wind.loc[j, 'pct_above_threshold'] <= SAFE_ALTERNATE_PCT_LIMIT
]

# safe alternates within reach of each site
wind_backups = {
    j: [k for k in safe_vertiports if k != j and flight_min(j, k) <= ALTERNATE_TIME_THRESHOLD]
    for j in candidate_coords
}

# clean up bushfire labels
df_bushfire['bfpl_category_label'] = df_bushfire['bfpl_category_label'].astype(str).str.strip()
df_bushfire = df_bushfire.dropna(subset=['name']).drop_duplicates(subset=['name']).set_index('name')

# identify high-risk vertiports (Vegetation Category 1)
high_risk_vertiports = [
    j for j in candidate_coords
    if j in df_bushfire.index and df_bushfire.loc[j, 'bfpl_category_label'] == 'Vegetation Category 1'
]

# rain
if 'name' in df_rain.columns:
    df_rain = df_rain.rename(columns={'name': 'vertiport'})
df_rain = df_rain.drop_duplicates(subset=['vertiport']).set_index('vertiport')

# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX
# ─────────────────────────────────────────────────────────────
# flight time in minutes to the nearest hospital per vertiport
air_time = {j: d_ih.loc[j].min() / UAV_SPEED_KMH * 60 for j in candidate_coords}

def build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T):
    a = {}
    for i in demand_coords:
        for j in candidate_coords:
            a[i, j] = 1 if road_time_lookup[i, j] + air_time[j] <= T else 0
    return a

# ─────────────────────────────────────────────────────────────
# RISK COST FUNCTIONS
# ─────────────────────────────────────────────────────────────
# normalize values between 0-1
def _minmax(d):
    lo, hi = min(d.values()), max(d.values())
    return {k: ((v - lo) / (hi - lo) if hi > lo else 0.0) for k, v in d.items()}

# returns bfpl severity score of vertiport site or 1.00 if missing
def bfpl_severity(j):
    if j not in df_bushfire.index:
        return MISSING_DATA_SEVERITY
    return BFPL_SEVERITY.get(df_bushfire.loc[j, 'bfpl_category_label'], MISSING_DATA_SEVERITY)

# number of other high-risk bushfire vertiports within emergency reach
def emergency_reach_count(j):
    return sum(1 for t in high_risk_vertiports
               if t != j and flight_min(j, t) <= EMERGENCY_TIME_THRESHOLD)

# returns wind severity score of vertiport site based on % of obs above threshold or 1.00 if missing
def wind_severity(j):
    if j not in df_wind.index:
        return MISSING_DATA_SEVERITY
    pct = df_wind.loc[j, 'pct_above_threshold']
    return MISSING_DATA_SEVERITY if np.isnan(pct) else min(pct / 100.0, 1.0)

# returns rain severity score of vertiport site based on average rainfall and heavy rain days or 1.00 if missing
def rain_severity(j):
    if j not in df_rain.index:
        return MISSING_DATA_SEVERITY
    row = df_rain.loc[j]
    avg_rain   = float(row['Avg_Rainfall'])
    heavy_days = float(row['Heavy_Rain_Days'])
    if np.isnan(avg_rain) or np.isnan(heavy_days):
        return MISSING_DATA_SEVERITY
    return (min(avg_rain / RAIN_MAX_AVG_MM_DAY, 1.0) + min(heavy_days / RAIN_MAX_HEAVY_DAYS, 1.0)) / 2.0

bf_norm   = _minmax({j: bfpl_severity(j) for j in candidate_coords})
wind_norm = _minmax({j: wind_severity(j) for j in candidate_coords})
rain_norm = _minmax({j: rain_severity(j) for j in candidate_coords})

def bushfire_cost(j):
    discount = min(MAX_BUSHFIRE_DISCOUNT, EMERGENCY_VALUE_PER_SITE * emergency_reach_count(j))
    return bf_norm[j] * (1 - discount)

def wind_cost(j):
    has_backup = len(wind_backups[j]) > 0  
    return wind_norm[j] * (1 - (WIND_BACKUP_DISCOUNT if has_backup else 0.0))

def rain_cost(j):
    return rain_norm[j]

# combined risk of site j (0-1)
def combined_cost(j):
    return (WEIGHT_BUSHFIRE * bushfire_cost(j) +
            WEIGHT_WIND     * wind_cost(j)     +
            WEIGHT_RAIN     * rain_cost(j)) / W_SUM

# ─────────────────────────────────────────────────────────────
# PYOMO MODEL
# ─────────────────────────────────────────────────────────────

def run_model(a, demand_coords, candidate_coords, alpha, cost, lam=RISK_LAMBDA):
    I = list(demand_coords.keys())
    J = list(candidate_coords.keys())
    n_req = ceil(alpha * len(I) - 1e-9)        

    mod = ConcreteModel()
    mod.I = Set(initialize=I)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.J, domain=Binary)
    mod.z = Var(mod.I, domain=Binary)

    mod.obj = Objective(
        expr=sum(mod.y[j] for j in J) + lam * sum(cost[j] * mod.y[j] for j in J),
        sense=minimize
    )

    mod.cov = Constraint(mod.I, rule=lambda mod, i:
        mod.z[i] <= sum(a[i, j] * mod.y[j] for j in J))

    mod.demand = Constraint(expr=sum(mod.z[i] for i in I) >= n_req)

    result = SolverFactory('cbc').solve(mod)

    if result.solver.termination_condition == TerminationCondition.optimal:
        selected  = [j for j in J if value(mod.y[j]) > 0.5]
        covered   = [i for i in I if any(a[i, j] == 1 for j in selected)]
        cov_set   = set(covered)
        uncovered = [i for i in I if i not in cov_set]
        return 'optimal', selected, covered, uncovered
    else:
        return 'infeasible', [], [], []

# ─────────────────────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────────────────────
a = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)

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
print(f"Risk weights    = Bushfire:{WEIGHT_BUSHFIRE} Wind:{WEIGHT_WIND} Rain:{WEIGHT_RAIN} | lambda={RISK_LAMBDA}\n")

if status == 'optimal':
    total_risk = 0.0
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites:")
    header = f"  {'Site':<55} {'Bushfire':>10} {'Wind':>10} {'Rain':>10} {'Total':>10}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for port in selected:
        bc = WEIGHT_BUSHFIRE * bushfire_cost(port) / W_SUM
        wc = WEIGHT_WIND     * wind_cost(port)     / W_SUM
        rc = WEIGHT_RAIN     * rain_cost(port)     / W_SUM
        total_risk += bc + wc + rc
        print(f"  {port:<55} {bc:>10.3f} {wc:>10.3f} {rc:>10.3f} {bc+wc+rc:>10.3f}")
    print(f"\nTotal risk         : {total_risk:.3f}")
    print(f"Regions covered    : {len(covered)} / {len(demand_coords)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg)
else:
    print("INFEASIBLE: No solution found for the given coverage target and time threshold.")


# ─────────────────────────────────────────────────────────────
# LAMBDA SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
lambda_analysis = False  # set to True to run lambda sensitivity analysis
if lambda_analysis:
    lambda_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
    lambda_results = []

    print("\n" + "="*60)
    print("LAMBDA SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"{'Lambda':<10} {'Vertiports':>12} {'Total Risk':>12} {'Avg Risk/Port':>15}")
    print("-"*50)

    for lam in lambda_values:
        s, sel, cov, uncov = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost, lam=lam)
        if s == 'optimal':
            total_risk = sum(cost[j] for j in sel)
            avg_risk   = total_risk / len(sel) if sel else 0
            lambda_results.append({
                'lambda':      lam,
                'n_vertiports': len(sel),
                'total_risk':  round(total_risk, 3),
                'avg_risk':    round(avg_risk, 3),
                'status':      'optimal'
            })
            print(f"{lam:<10} {len(sel):>12} {total_risk:>12.3f} {avg_risk:>15.3f}")
        else:
            lambda_results.append({'lambda': lam, 'n_vertiports': None, 'total_risk': None, 'avg_risk': None, 'status': 'infeasible'})
            print(f"{lam:<10} {'INFEASIBLE':>12}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    feasible = [r for r in lambda_results if r['status'] == 'optimal']
    lam_vals  = [r['lambda']       for r in feasible]
    n_ports   = [r['n_vertiports'] for r in feasible]
    t_risk    = [r['total_risk']   for r in feasible]

    ax1 = axes[0]
    ax1.plot(lam_vals, n_ports, marker='o', color='steelblue')
    for x, y in zip(lam_vals, n_ports):
        ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Minimum Vertiports Required')
    ax1.set_title('Vertiport Count vs Lambda')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2.plot(lam_vals, t_risk, marker='o', color='firebrick')
    for x, y in zip(lam_vals, t_risk):
        ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Total Risk of Selected Sites')
    ax2.set_title('Total Network Risk vs Lambda')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f'Lambda Sensitivity | T={TIME_THRESHOLD} min | Coverage={COVERAGE_THRESHOLD*100:.0f}%', fontsize=12)
    plt.tight_layout()
    plt.savefig('lambda_sensitivity.png', dpi=150)
    plt.show()
    print("Plot saved to lambda_sensitivity.png")

# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────
Run_analysis = False # set to True to run sensitivity analysis

if Run_analysis:
    time_thresholds  = [30, 45, 60, 75, 90, 105, 120]
    coverage_targets = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    results = {}

    for T in time_thresholds:
        a_temp = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, T)
        for alpha in coverage_targets:
            s, sel, cov, uncov = run_model(a_temp, demand_coords, candidate_coords, alpha, cost)
            if s == 'optimal':
                results[T, alpha] = len(sel)
            else:
                results[T, alpha] = None

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
        ax2.plot([al*100 for al in coverage_targets], y_vals, marker='o', label=f'{T} min')
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


# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
SA2_SHAPEFILE = 'SA2_2021_AUST_GDA2020.shp'   # path to the ABS boundary file
SA2_NAME_COL  = 'SA2_NAME21'                   # SA2 name column in that file

def match_polygons(names):
        """Return the boundary polygons that contain each SA2's centroid."""
        if not names:
            return None
        pts = gpd.GeoDataFrame(
            {'sa2': names},
            geometry=[Point(demand_coords[i][1], demand_coords[i][0]) for i in names],  # (lon, lat)
            crs='EPSG:4326'
        )
        joined = gpd.sjoin(pts, gdf[[SA2_NAME_COL, 'geometry']], how='left', predicate='within')
        idx = set(joined['index_right'].dropna().astype(int))

        missing = joined.loc[joined['index_right'].isna(), 'sa2'].tolist()
        if missing:
            near = gpd.sjoin_nearest(
                pts[pts['sa2'].isin(missing)].to_crs(epsg=3577),
                gdf[['geometry']].to_crs(epsg=3577),
                how='left'
            )
            idx |= set(near['index_right'].astype(int))

        out = gdf.loc[sorted(idx)].copy()
        out['geometry'] = out.geometry.simplify(0.001, preserve_topology=True)
        return out

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

    gdf = gpd.read_file(SA2_SHAPEFILE).to_crs(epsg=4326)

    covered_gdf   = match_polygons(covered)
    uncovered_gdf = match_polygons(uncovered)

    # uncovered regions
    if uncovered_gdf is not None:
        folium.GeoJson(
            uncovered_gdf[[SA2_NAME_COL, 'geometry']],
            style_function=lambda f: {
                'fillColor':   '#000000',
                'color':       '#000000',
                'weight':      1,
                'fillOpacity': 0.6,
            },
            tooltip=folium.GeoJsonTooltip(fields=[SA2_NAME_COL], aliases=['Uncovered:'])
        ).add_to(m_map)

    # covered regions
    if covered_gdf is not None:
        folium.GeoJson(
            covered_gdf[[SA2_NAME_COL, 'geometry']],
            style_function=lambda f: {
                'fillColor':   '#7fb8e0',
                'color':       '#2b7bb9',
                'weight':      1.5,
                'fillOpacity': 0.6,
            },
            tooltip=folium.GeoJsonTooltip(fields=[SA2_NAME_COL], aliases=['Region:'])
        ).add_to(m_map)

    # selected vertiports
    for j in selected:
        lat, lon = candidate_coords[j]
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(j, max_width=200),
            tooltip=j,
            icon=folium.Icon(color='red', icon='plane', prefix='fa')
        ).add_to(m_map)

    # reachable hospitals
    for h, (h_lat, h_lon) in hospital_coords.items():
        reachable = any(d_ih.loc[j, h] / UAV_SPEED_KMH * 60 <= TIME_THRESHOLD for j in selected)
        if reachable:
            folium.CircleMarker(
                location=[h_lat, h_lon], radius=2, color='green', fill=True, fill_color='green', fill_opacity=0.2,
                popup=folium.Popup(h, max_width=200), tooltip=h
            ).add_to(m_map)


    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 230px;
         background-color: white; border:2px solid grey; z-index:9999;
         font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
    <p style="margin: 5px 0;"><i style="background:#7fb8e0; width:12px; height:12px;
       display:inline-block; border:1px solid #2b7bb9;"></i> Covered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:black; width:12px; height:12px;
       display:inline-block; border:1px solid black;"></i> Uncovered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i class="fa fa-map-marker" style="color:red; font-size:16px;
       width:12px; text-align:center;"></i> Selected Vertiports</p>
    <p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Reachable Hospitals</p>
    </div>
    '''
    m_map.get_root().html.add_child(folium.Element(legend_html))
    m_map.save('vertiports_map.html')
    print("Map saved to vertiports_map.html")