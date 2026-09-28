"""
optim_model_wind_risk.py

Extends optim_model.py to incorporate wind operational risk into vertiport
site selection using a SOFT PENALTY, following the same pattern as
optim_model_bushfire_risk.py (penalty offset by a reliability value,
floored at 0, added to the objective alongside vertiport count).

NOTE ON SPEC DEVIATION: the project spec ("Vertiport Location Model for
Rural Medical Access", section 6.3) frames wind risk as a HARD feasibility
constraint alongside flood and bushfire risk -- "if a site fails a
feasibility test, it cannot be selected as a vertiport." This script
instead treats it as a SOFT penalty, matching the convention already used
in optim_model_bushfire_risk.py. This is a deliberate deviation from the
literal spec, made for consistency with the team's other risk models and
to preserve solver flexibility -- state this explicitly in the write-up
rather than presenting it as a literal implementation of 6.3.

Design Intent
------------------------------------------------------------------------
Vertiports exposed to frequent high wind carry a penalty, since operations
there would be grounded more often (Schweiger, Schmitz & Knabe, 2023 --
eVTOL wind-closure case study for Hamburg/Munich). But a windy site isn't
simply excluded if it has a reliable, low-wind ALTERNATE vertiport nearby
that flights can be redirected to within a short window -- this mirrors
EASA's continuous-safe-flight-and-landing requirement for an alternate
vertiport. That reliability is weighted by the population the alternate
itself serves, matching the population-weighting pattern used in the
bushfire risk model.

    net_wind_cost(j) = wind_penalty(j) - alternate_value(j)
                        (floored at 0 -- offset, never a net incentive
                         to add unnecessary vertiports)

    alternate_value(j) = ALTERNATE_VALUE_PER_1000_POP x
                          (population served by the best reachable safe
                           alternate / 1000), only if one exists within
                          ALTERNATE_TIME_THRESHOLD minutes
"""

import os
import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, sin, cos, sqrt, atan2, floor
import folium

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
        df_year = load_wind_data_year(path, usecols)
        df_year['data_year'] = year
        frames.append(df_year)
        print(f"  Loaded {path}: {len(df_year):,} usable rows")

    if not frames:
        raise FileNotFoundError(
            f"No wind data files found matching '{pattern}' for years {years} in '{csv_dir}'. "
            "Check WIND_CSV_DIR / WIND_CSV_PATTERN / WIND_CSV_YEARS."
        )
    return pd.concat(frames, ignore_index=True)

print("Loading yearly wind data files...")
wind_raw = load_wind_data_all_years(WIND_CSV_DIR, WIND_CSV_PATTERN, WIND_CSV_YEARS, WIND_USECOLS)
print(f"Loaded {len(wind_raw):,} total wind observations across {wind_raw['station'].nunique()} stations "
      f"and {wind_raw['data_year'].nunique()} year(s)")

# Station coordinates taken directly from the data (median per station).
metar_station_coords = (
    wind_raw.groupby('station')[['lat', 'lon']]
    .median()
    .apply(lambda r: (r['lat'], r['lon']), axis=1)
    .to_dict()
)

#--------------------
#   PARAMETERS
#--------------------

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 90
COVERAGE_THRESHOLD = 0.95

# --- wind risk settings (mirrors bushfire's naming convention) ---
WIND_THRESHOLD_KT            = 20.0  # Schweiger et al. 2023 "Wind-Advisory" cutoff
WIND_PENALTY_WEIGHT          = 3.0   # cost at 100% exposure, scaled down proportionally below that
ALTERNATE_TIME_THRESHOLD     = 30    # minutes (EASA-style alternate-vertiport redirect window)
ALTERNATE_VALUE_PER_1000_POP = 0.4   # cost offset earned per 1,000 people served by a reachable safe alternate
SAFE_ALTERNATE_PCT_LIMIT     = 10.0  # a vertiport counts as "safe" if wind exceeds threshold <10% of hours

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

station_severity = build_station_severity(wind_raw, metar_station_coords, WIND_THRESHOLD_KT)

def assign_vertiport_wind_risk(candidate_coords, station_coords, station_severity):
    rows = []
    for name, (lat, lon) in candidate_coords.items():
        station, dist_km = nearest_metar_station(lat, lon, station_coords)
        pct = station_severity.loc[station, 'pct_above_threshold']
        rows.append({'vertiport': name, 'nearest_station': station,
                     'station_dist_km': dist_km, 'pct_above_threshold': pct})
    return pd.DataFrame(rows).set_index('vertiport')

vertiport_wind = assign_vertiport_wind_risk(candidate_coords, metar_station_coords, station_severity)
vertiport_wind.to_csv('vertiport_wind_risk.csv')
print(f"Saved: vertiport_wind_risk.csv ({len(vertiport_wind)} vertiports)")

FAR_STATION_WARNING_KM = 75
far = vertiport_wind[vertiport_wind['station_dist_km'] > FAR_STATION_WARNING_KM]
if len(far):
    print(f"\nWARNING: {len(far)} vertiport(s) are >{FAR_STATION_WARNING_KM}km from their nearest METAR station:")
    print(far)

def wind_severity_01(name):
    pct = vertiport_wind.loc[name, 'pct_above_threshold']
    # NaN (no nearby station data) is treated as maximum severity (1.0) for a
    # soft penalty -- unlike a hard gate, this doesn't remove the site, it
    # just penalises unknown-risk sites fully rather than defaulting to zero
    # risk. Flag this choice in your write-up.
    return 1.0 if np.isnan(pct) else min(pct / 100.0, 1.0)

# "Safe" alternates identified once, same pattern as bushfire's high_risk_vertiports list.
safe_vertiports = [
    j for j in candidate_coords
    if not np.isnan(vertiport_wind.loc[j, 'pct_above_threshold'])
    and vertiport_wind.loc[j, 'pct_above_threshold'] <= SAFE_ALTERNATE_PCT_LIMIT
]
print(f"Safe (low-wind) alternate vertiports identified: {len(safe_vertiports)}")

def population_served(j, a):
    """Population of the SA2s this vertiport covers within TIME_THRESHOLD --
    identical logic to the bushfire model, reusing the coverage matrix a."""
    return sum(demand_population[i] for i in demand_coords if a[i, j] == 1)

def alternate_reach_population(j, a, pop_served):
    """Population served by the best safe alternate reachable from j by
    direct flight within ALTERNATE_TIME_THRESHOLD minutes (excluding itself)."""
    j_lat, j_lon = candidate_coords[j]
    best_pop = 0
    for target in safe_vertiports:
        if target == j:
            continue
        t_lat, t_lon = candidate_coords[target]
        flight_time = haversine_km(j_lat, j_lon, t_lat, t_lon) / UAV_SPEED_KMH * 60
        if flight_time <= ALTERNATE_TIME_THRESHOLD:
            best_pop = max(best_pop, pop_served[target])
    return best_pop

def wind_cost(j, a, pop_served):
    risk_penalty    = WIND_PENALTY_WEIGHT * wind_severity_01(j)
    alternate_value = ALTERNATE_VALUE_PER_1000_POP * (alternate_reach_population(j, a, pop_served) / 1000)
    return max(0.0, risk_penalty - alternate_value)   # floor at 0 -- offset, not incentive

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

#--------------------
# MAIN MODEL
#--------------------

a = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)

pop_served = {j: population_served(j, a) for j in candidate_coords}
cost = {j: wind_cost(j, a, pop_served) for j in candidate_coords}

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
print(f"Wind penalty weight     = {WIND_PENALTY_WEIGHT}")
print(f"Wind threshold          = {WIND_THRESHOLD_KT} kt")
print(f"Alternate time thresh   = {ALTERNATE_TIME_THRESHOLD} min")
print(f"Alternate value/1000pop = {ALTERNATE_VALUE_PER_1000_POP}\n")

if status == 'optimal':
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites     :")
    header = f"    {'Site':<45} {'Wind %>thresh':<14} {'Pop served':>14} {'Alt reach pop':>15} {'Net cost':>11}"
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for port in selected:
        pct = vertiport_wind.loc[port, 'pct_above_threshold']
        pct_disp = pct if not np.isnan(pct) else float('nan')
        alt_pop = alternate_reach_population(port, a, pop_served)
        print(f"    {port:<45} {pct_disp:<14.1f} {pop_served[port]:>14,.0f} {alt_pop:>15,.0f} {cost[port]:>11.3f}")
    print(f"\nRegions covered    : {len(covered)} / {len(demand_coords)}")
    print("Uncovered regions  :")
    for reg in uncovered:
        print('\t', reg)
else:
    print("INFEASIBLE — lower COVERAGE_THRESHOLD or increase TIME_THRESHOLD")

#--------------------
# SENSITIVITY ANALYSIS (wind penalty weight x wind threshold)
#--------------------
'''
weights_to_test    = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
thresholds_to_test = [15.0, 17.0, 20.0, 25.0]  # knots, Schweiger et al. categories

sens_rows = []
for w in weights_to_test:
    for th in thresholds_to_test:
        sev_temp = build_station_severity(wind_raw, metar_station_coords, th)
        vwind_temp = assign_vertiport_wind_risk(candidate_coords, metar_station_coords, sev_temp)
        safe_temp = [j for j in candidate_coords
                     if not np.isnan(vwind_temp.loc[j, 'pct_above_threshold'])
                     and vwind_temp.loc[j, 'pct_above_threshold'] <= SAFE_ALTERNATE_PCT_LIMIT]

        def wind_cost_temp(j, a, pop_served):
            pct = vwind_temp.loc[j, 'pct_above_threshold']
            severity = 1.0 if np.isnan(pct) else min(pct / 100.0, 1.0)
            j_lat, j_lon = candidate_coords[j]
            best_pop = 0
            for target in safe_temp:
                if target == j:
                    continue
                t_lat, t_lon = candidate_coords[target]
                ft = haversine_km(j_lat, j_lon, t_lat, t_lon) / UAV_SPEED_KMH * 60
                if ft <= ALTERNATE_TIME_THRESHOLD:
                    best_pop = max(best_pop, pop_served[target])
            return max(0.0, w * severity - ALTERNATE_VALUE_PER_1000_POP * (best_pop / 1000))

        cost_temp = {j: wind_cost_temp(j, a, pop_served) for j in candidate_coords}
        s, sel, cov, uncov = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost_temp)
        if s == 'optimal':
            avg_severity = np.mean([wind_severity_01(j) for j in sel])
            sens_rows.append({'penalty_weight': w, 'wind_threshold_kt': th,
                               'n_vertiports': len(sel), 'avg_wind_severity': avg_severity})
            print(f"w={w} th={th}kt | vertiports={len(sel)} | avg_severity={avg_severity:.3f}")
        else:
            sens_rows.append({'penalty_weight': w, 'wind_threshold_kt': th,
                               'n_vertiports': None, 'avg_wind_severity': None})
            print(f"w={w} th={th}kt | INFEASIBLE")

pd.DataFrame(sens_rows).to_excel('sensitivity_wind_risk.xlsx', index=False)
print("\nSaved: sensitivity_wind_risk.xlsx")
'''

#--------------------
# MAP
#--------------------

m_map = folium.Map(location=[-33.0, 151.0], zoom_start=7, tiles='CartoDB positron')

m_map.get_root().html.add_child(folium.Element('''
    <style>.leaflet-interactive:focus { outline: none !important; }</style>
'''))

# safe alternate vertiports -- shown first so other markers layer on top
for j in safe_vertiports:
    lat, lon = candidate_coords[j]
    folium.CircleMarker(
        location=[lat, lon], radius=5, color='darkgreen', fill=True, fill_opacity=0.4, weight=1,
        popup=folium.Popup(f"{j} (safe low-wind alternate)", max_width=220), tooltip=f"{j} (safe)"
    ).add_to(m_map)

for j in selected:
    lat, lon = candidate_coords[j]
    pct = vertiport_wind.loc[j, 'pct_above_threshold']
    alt_pop = alternate_reach_population(j, a, pop_served)
    if np.isnan(pct):
        colour = 'black'
    else:
        colour = 'green' if pct < 5 else ('orange' if pct < 15 else 'red')
    pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "no data"
    folium.CircleMarker(
        location=[lat, lon], radius=7, color=colour, fill=True, fill_opacity=1,
        popup=folium.Popup(
            f"{j}<br>Wind: {pct_str} hours &gt; {WIND_THRESHOLD_KT}kt<br>"
            f"Pop served: {pop_served[j]:,.0f}<br>Alt reach pop: {alt_pop:,.0f}<br>Net cost: {cost[j]:.3f}",
            max_width=250),
        tooltip=j
    ).add_to(m_map)

for i in covered:
    lat, lon = demand_coords[i]
    folium.CircleMarker(location=[lat, lon], radius=4, color='blue', fill=True, fill_opacity=1,
                         popup=folium.Popup(i, max_width=200), tooltip=i).add_to(m_map)

for i in uncovered:
    lat, lon = demand_coords[i]
    folium.CircleMarker(location=[lat, lon], radius=4, color='gray', fill=True, fill_opacity=1,
                         popup=folium.Popup(i, max_width=200), tooltip=i).add_to(m_map)

for code, (slat, slon) in metar_station_coords.items():
    folium.CircleMarker(location=[slat, slon], radius=5, color='purple', fill=True, fill_opacity=1,
                         popup=folium.Popup(f"METAR station: {code}", max_width=200), tooltip=code).add_to(m_map)

legend_html = '''
<div style="position: fixed; bottom: 50px; right: 50px; width: 250px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
<p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Selected — low wind risk</p>
<p style="margin: 5px 0;"><i style="background:orange; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Selected — moderate wind risk</p>
<p style="margin: 5px 0;"><i style="background:red; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Selected — high wind risk</p>
<p style="margin: 5px 0;"><i style="background:blue; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Covered Regions (SA2)</p>
<p style="margin: 5px 0;"><i style="background:gray; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Uncovered Regions (SA2)</p>
<p style="margin: 5px 0;"><i style="background:purple; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> METAR station (wind proxy)</p>
<p style="margin: 5px 0;"><i style="background:darkgreen; opacity:0.4; width:12px; height:12px;
   display:inline-block; border-radius:50%;"></i> Safe low-wind alternate</p>
</div>
'''
m_map.get_root().html.add_child(folium.Element(legend_html))
m_map.save('vertiports_map_wind_risk_fix.html')
print("Map saved to vertiports_map_wind_risk_fix.html")
