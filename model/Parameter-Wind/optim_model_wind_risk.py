"""
optim_model_wind_risk.py

Wind operational risk in vertiport site selection, using the same wind
approach as the combined risk model:

    minimise  sum_j y_j  +  RISK_LAMBDA * sum_j wind_cost_j * y_j

    wind_severity_j = pct_above_threshold_j / 100   (capped at 1; missing data = MISSING_DATA_SEVERITY)
    wind_cost_j     = wind_severity_j min-max normalised across all candidate sites (0-1)

A selected site saves its wind risk only if the risk avoided is worth more
than one extra vertiport, i.e. risk saved > 1 / RISK_LAMBDA.

Wind is a SOFT penalty: no site is excluded. The project spec (section 6.3)
frames wind as a hard feasibility constraint -- state this deviation in the
write-up. No safe-alternate (neighbouring vertiport) discount is applied.
"""

import os
import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import folium
from folium.plugins import MeasureControl

#--------------------
# LOAD THE DATA SETS
#--------------------

dfi = pd.read_excel('vertiport_name_with_coord.xlsx')
dfj = pd.read_excel('sa2_names_with_coord.xlsx')
dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')

candidate_coords = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords    = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords  = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))

# Real population source -- same column the bushfire model already uses.
demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))

#--------------------
# LOAD WIND DATA (ASOS/METAR, mesonet.agron.iastate.edu)
#--------------------

# Wind data is now split into one file per year (to get past GitHub's file
# size limit on the single combined asos_full.csv). UPDATE WIND_CSV_YEARS
# and WIND_CSV_DIR/WIND_CSV_PATTERN if your filenames or folder differ.
WIND_CSV_DIR     = '.'                       # folder containing the yearly files
WIND_CSV_PATTERN = 'asos_{year}.csv'         # e.g. asos_2021.csv, asos_2022.csv, ...
WIND_CSV_YEARS   = [2021, 2022, 2023, 2024, 2025, 2026]
WIND_USECOLS     = ['station', 'valid', 'lon', 'lat', 'drct', 'sknt', 'gust']

def load_wind_data_year(path, usecols):
    df = pd.read_csv(path, usecols=usecols, na_values=['M'])
    df['valid'] = pd.to_datetime(df['valid'], errors='coerce')
    df['sknt']  = pd.to_numeric(df['sknt'], errors='coerce')
    df['gust']  = pd.to_numeric(df['gust'], errors='coerce')
    df['lat']   = pd.to_numeric(df['lat'], errors='coerce')
    df['lon']   = pd.to_numeric(df['lon'], errors='coerce')

    # Schweiger et al. (2023): use gust when reported, else sustained speed.
    df['effective_kt'] = df['gust'].where(df['gust'].notna(), df['sknt'])
    df = df.dropna(subset=['effective_kt', 'lat', 'lon'])
    return df

def load_wind_data_all_years(csv_dir, pattern, years, usecols):
    """Loads and concatenates one CSV per year. Missing years are skipped
    with a warning rather than raising, so a partial pull (e.g. 2026 not
    finished yet) doesn't crash the whole script."""
    frames = []
    for year in years:
        path = os.path.join(csv_dir, pattern.format(year=year))
        if not os.path.exists(path):
            print(f"WARNING: {path} not found -- skipping year {year}")
            continue
        if os.path.getsize(path) == 0:
            print(f"WARNING: {path} is empty (0 bytes) -- skipping year {year}. Re-download it.")
            continue
        try:
            df_year = load_wind_data_year(path, usecols)
        except pd.errors.EmptyDataError:
            print(f"WARNING: {path} has no data rows or header -- skipping year {year}. Re-download it.")
            continue
        except ValueError as e:
            # Usually means the expected columns are missing, e.g. the file is a
            # Git LFS pointer or an error page saved as .csv instead of real data.
            print(f"WARNING: {path} could not be read ({e}) -- skipping year {year}. "
                  f"Open the file and check it starts with the header 'station,valid,...'.")
            continue
        if len(df_year) == 0:
            print(f"WARNING: {path} has no usable wind rows -- skipping year {year}")
            continue
        df_year['data_year'] = year
        frames.append(df_year)
        print(f"  Loaded {path}: {len(df_year):,} usable rows")

    if not frames:
        raise FileNotFoundError(
            f"No wind data files found matching '{pattern}' for years {years} in '{csv_dir}'. "
            "Check WIND_CSV_DIR / WIND_CSV_PATTERN / WIND_CSV_YEARS."
        )
    return pd.concat(frames, ignore_index=True)

# If none of the yearly raw files can be read (e.g. they are Git LFS pointer
# files or failed downloads), fall back to the previously saved per-vertiport
# results in this file instead of stopping.
PRECOMPUTED_WIND_FILE = 'vertiport_wind_risk.csv'

print("Loading yearly wind data files...")
try:
    wind_raw = load_wind_data_all_years(WIND_CSV_DIR, WIND_CSV_PATTERN, WIND_CSV_YEARS, WIND_USECOLS)
    USE_PRECOMPUTED_WIND = False
    print(f"Loaded {len(wind_raw):,} total wind observations across {wind_raw['station'].nunique()} stations "
          f"and {wind_raw['data_year'].nunique()} year(s)")

    # Station coordinates taken directly from the data (median per station).
    metar_station_coords = (
        wind_raw.groupby('station')[['lat', 'lon']]
        .median()
        .apply(lambda r: (r['lat'], r['lon']), axis=1)
        .to_dict()
    )
except FileNotFoundError as e:
    if not os.path.exists(PRECOMPUTED_WIND_FILE):
        raise FileNotFoundError(
            f"{e}\nNo raw wind data could be read, and the fallback file "
            f"'{PRECOMPUTED_WIND_FILE}' was not found either."
        )
    print(f"\nWARNING: no raw wind data could be read. Using the saved results in "
          f"'{PRECOMPUTED_WIND_FILE}' instead. METAR station markers will not appear on the map.\n")
    USE_PRECOMPUTED_WIND = True
    wind_raw = None
    metar_station_coords = {}

#--------------------
#   PARAMETERS
#--------------------

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 90
COVERAGE_THRESHOLD = 0.95

# --- wind risk settings (mirrors bushfire's naming convention) ---
WIND_THRESHOLD_KT     = 20.0  # Schweiger et al. 2023: top of the Wind-Advisory band
MISSING_DATA_SEVERITY = 1.0   # severity given to a site with no wind data (worst case)

# How strongly wind risk trades against vertiport count.
# Higher = accept more vertiports to avoid windy sites.
RISK_LAMBDA = 0.5

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

def nearest_metar_station(lat, lon, station_coords):
    best_station, best_dist = None, float('inf')
    for code, (slat, slon) in station_coords.items():
        d = haversine_km(lat, lon, slat, slon)
        if d < best_dist:
            best_dist, best_station = d, code
    return best_station, best_dist

#--------------------
# WIND RISK METRICS
#--------------------

def build_station_severity(wind_df, station_coords, threshold_kt):
    rows = []
    for code in station_coords:
        s = wind_df[wind_df['station'] == code]
        if len(s) == 0:
            rows.append({'station': code, 'pct_above_threshold': np.nan, 'avg_wind_kt': np.nan, 'n_obs': 0})
            continue
        rows.append({
            'station': code,
            'pct_above_threshold': (s['effective_kt'] > threshold_kt).mean() * 100,
            'avg_wind_kt': s['effective_kt'].mean(),
            'n_obs': len(s),
        })
    return pd.DataFrame(rows).set_index('station')

def assign_vertiport_wind_risk(candidate_coords, station_coords, station_severity):
    rows = []
    for name, (lat, lon) in candidate_coords.items():
        station, dist_km = nearest_metar_station(lat, lon, station_coords)
        pct = station_severity.loc[station, 'pct_above_threshold']
        rows.append({'vertiport': name, 'nearest_station': station,
                     'station_dist_km': dist_km, 'pct_above_threshold': pct})
    return pd.DataFrame(rows).set_index('vertiport')

if USE_PRECOMPUTED_WIND:
    # Read the saved results; do NOT overwrite them.
    vertiport_wind = pd.read_csv(PRECOMPUTED_WIND_FILE).set_index('vertiport')
    missing = [j for j in candidate_coords if j not in vertiport_wind.index]
    if missing:
        print(f"WARNING: {len(missing)} vertiport(s) not in {PRECOMPUTED_WIND_FILE}; "
              f"they will be treated as having no wind data (maximum severity).")
        vertiport_wind = vertiport_wind.reindex(list(vertiport_wind.index) + missing)
    print(f"Loaded: {PRECOMPUTED_WIND_FILE} ({len(vertiport_wind)} vertiports)")
else:
    station_severity = build_station_severity(wind_raw, metar_station_coords, WIND_THRESHOLD_KT)
    vertiport_wind = assign_vertiport_wind_risk(candidate_coords, metar_station_coords, station_severity)
    vertiport_wind.to_csv(PRECOMPUTED_WIND_FILE)
    print(f"Saved: {PRECOMPUTED_WIND_FILE} ({len(vertiport_wind)} vertiports)")

FAR_STATION_WARNING_KM = 75
far = vertiport_wind[vertiport_wind['station_dist_km'] > FAR_STATION_WARNING_KM] \
    if 'station_dist_km' in vertiport_wind.columns else vertiport_wind.iloc[0:0]
if len(far):
    print(f"\nWARNING: {len(far)} vertiport(s) are >{FAR_STATION_WARNING_KM}km from their nearest METAR station:")
    print(far)

# returns wind severity of a site from the % of readings above the threshold,
# or MISSING_DATA_SEVERITY if there is no wind data for it
def wind_severity(j):
    if j not in vertiport_wind.index:
        return MISSING_DATA_SEVERITY
    pct = vertiport_wind.loc[j, 'pct_above_threshold']
    return MISSING_DATA_SEVERITY if pd.isna(pct) else min(pct / 100.0, 1.0)

# normalise values to 0-1 across the candidate sites
def _minmax(d):
    lo, hi = min(d.values()), max(d.values())
    return {k: ((v - lo) / (hi - lo) if hi > lo else 0.0) for k, v in d.items()}

wind_norm = _minmax({j: wind_severity(j) for j in candidate_coords})

# wind cost of site j (0-1)
def wind_cost(j):
    return wind_norm[j]

#--------------------
# LOAD MATRICES (reused from optim_model.py)
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

def run_model(a, demand_coords, candidate_coords, alpha, cost, lam=RISK_LAMBDA):
    I = list(demand_coords.keys())
    J = list(candidate_coords.keys())

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

cost = {j: wind_cost(j) for j in candidate_coords}

coverable = [i for i in demand_coords if any(a[i, j] == 1 for j in candidate_coords)]
print(f"\nMax achievable coverage: {len(coverable)}/{len(demand_coords)} = {len(coverable)/len(demand_coords)*100:.1f}%")
print(f"Requested coverage: {COVERAGE_THRESHOLD*100}%")

status, selected, covered, uncovered = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost)

#--------------------
# RESULTS
#--------------------

print(f"\nTime threshold          = {TIME_THRESHOLD} min")
print(f"Vehicle speed           = {UAV_SPEED_KMH} km/h")
print(f"Coverage target         = {COVERAGE_THRESHOLD*100}%")
print(f"Wind threshold          = {WIND_THRESHOLD_KT} kt")
print(f"Risk lambda             = {RISK_LAMBDA}\n")

if status == 'optimal':
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites     :")
    header = f"    {'Site':<45} {'Wind %>thresh':>14} {'Severity':>10} {'Wind cost':>10}"
    print(header)
    print('    ' + '-' * (len(header) - 4))
    total_risk = 0.0
    for port in selected:
        pct = vertiport_wind.loc[port, 'pct_above_threshold'] if port in vertiport_wind.index else np.nan
        total_risk += cost[port]
        print(f"    {port:<45} {pct:>14.1f} {wind_severity(port):>10.3f} {cost[port]:>10.3f}")
    print(f"\nTotal wind risk    : {total_risk:.3f}")
    print(f"\nRegions covered    : {len(covered)} / {len(demand_coords)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg)
else:
    print("INFEASIBLE — lower COVERAGE_THRESHOLD or increase TIME_THRESHOLD")

#--------------------
# SENSITIVITY ANALYSIS (risk lambda x wind threshold)
# Needs the raw yearly wind files; set Run_analysis = True to run it.
#--------------------
Run_analysis = False

if Run_analysis and not USE_PRECOMPUTED_WIND:
    lambdas_to_test    = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    thresholds_to_test = [15.0, 17.0, 20.0, 25.0]  # knots, Schweiger et al. categories

    sens_rows = []
    for th in thresholds_to_test:
        sev_temp   = build_station_severity(wind_raw, metar_station_coords, th)
        vwind_temp = assign_vertiport_wind_risk(candidate_coords, metar_station_coords, sev_temp)
        sev = {j: (MISSING_DATA_SEVERITY if pd.isna(vwind_temp.loc[j, 'pct_above_threshold'])
                   else min(vwind_temp.loc[j, 'pct_above_threshold'] / 100.0, 1.0))
               for j in candidate_coords}
        cost_temp = _minmax(sev)
        for lam in lambdas_to_test:
            s_, sel, cov, uncov = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost_temp, lam=lam)
            if s_ == 'optimal':
                total = sum(cost_temp[j] for j in sel)
                sens_rows.append({'lambda': lam, 'wind_threshold_kt': th,
                                  'n_vertiports': len(sel), 'total_wind_risk': round(total, 3)})
                print(f"lambda={lam} th={th}kt | vertiports={len(sel)} | total wind risk={total:.3f}")
            else:
                sens_rows.append({'lambda': lam, 'wind_threshold_kt': th,
                                  'n_vertiports': None, 'total_wind_risk': None})
                print(f"lambda={lam} th={th}kt | INFEASIBLE")

    pd.DataFrame(sens_rows).to_excel('sensitivity_wind_risk.xlsx', index=False)
    print("\nSaved: sensitivity_wind_risk.xlsx")
elif Run_analysis:
    print("Sensitivity analysis skipped: it needs the raw yearly wind files.")

#--------------------
# MAP  (same style as the shuttle V2 map)
#--------------------

m_map = folium.Map(location=[-33.0, 151.0], zoom_start=7,
                   tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
                   attr='Esri, HERE, Garmin, FAO, NOAA, USGS')
MeasureControl(position='bottomleft', primaryLengthUnit='kilometers', secondaryLengthUnit='meters',
               activeColor='orange', completedColor='green').add_to(m_map)
m_map.get_root().html.add_child(folium.Element('<style>.leaflet-interactive:focus { outline: none !important; }</style>'))

SELECTED_COL  = 'blue'
COVERED_COL   = 'red'
UNCOVERED_COL = 'orange'
STATION_COL   = 'purple'

def wind_text(j):
    pct = vertiport_wind.loc[j, 'pct_above_threshold']
    return f"{pct:.1f}% of readings &gt; {WIND_THRESHOLD_KT:.0f} kt" if not pd.isna(pct) else "no wind data"

# METAR stations used as the wind source
for code, (slat, slon) in metar_station_coords.items():
    folium.CircleMarker([slat, slon], radius=5, color=STATION_COL, fill=True, fill_opacity=1,
                        popup=folium.Popup(f"<b>METAR station {code}</b>", max_width=200),
                        tooltip=f"METAR {code}").add_to(m_map)

# demand areas: covered red, uncovered orange
for i in demand_coords:
    lat, lon = demand_coords[i]
    is_cov = i in covered
    folium.CircleMarker([lat, lon], radius=4, color=COVERED_COL if is_cov else UNCOVERED_COL,
                        fill=True, fill_opacity=1,
                        popup=folium.Popup(f"{i}<br>{'Covered' if is_cov else 'Not covered'}"
                                           f"<br>Population: {demand_population[i]:,.0f}", max_width=280),
                        tooltip=i).add_to(m_map)

# selected vertiports: blue, with wind details in the popup
for j in selected:
    lat, lon = candidate_coords[j]
    station = vertiport_wind.loc[j, 'nearest_station'] if 'nearest_station' in vertiport_wind.columns else None
    dist = vertiport_wind.loc[j, 'station_dist_km'] if 'station_dist_km' in vertiport_wind.columns else np.nan
    station_txt = (f"<br>Wind source: {station} ({dist:.0f} km away)"
                   if isinstance(station, str) and not pd.isna(dist) else "")
    folium.CircleMarker([lat, lon], radius=6, color=SELECTED_COL, fill=True, fill_opacity=1,
                        popup=folium.Popup(f"<b>{j}</b><br>Wind: {wind_text(j)}{station_txt}"
                                           f"<br>Wind cost (0-1): {cost[j]:.3f}", max_width=280),
                        tooltip=j).add_to(m_map)

dot = lambda col, op=1: (f'<i style="background:{col}; opacity:{op}; width:12px; height:12px; '
                         f'display:inline-block; border-radius:50%;"></i>')
legend_html = f'''
<div style="position: fixed; bottom: 50px; right: 50px; width: 260px; background-color: white;
     border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
<p style="margin: 5px 0;">{dot(SELECTED_COL)} Selected Vertiports ({len(selected)})</p>
<p style="margin: 5px 0;">{dot(COVERED_COL)} Covered areas ({len(covered)})</p>
<p style="margin: 5px 0;">{dot(UNCOVERED_COL)} Uncovered areas ({len(uncovered)})</p>
<p style="margin: 5px 0;">{dot(STATION_COL)} METAR stations ({len(metar_station_coords)})</p>
</div>'''
m_map.get_root().html.add_child(folium.Element(legend_html))
m_map.save('vertiports_map_wind_risk_fix.html')
print("Map saved to vertiports_map_wind_risk_fix.html")