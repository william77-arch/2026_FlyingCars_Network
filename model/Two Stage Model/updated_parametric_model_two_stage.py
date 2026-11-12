"""
Objective:
    minimise  Σ y_j  +  RISK_LAMBDA * Σ risk_j * y_j

    risk_j = (w_bush * bushfire_j + w_wind * wind_j + w_rain * rain_j) / (w_bush + w_wind + w_rain)
    Each hazard is normalised to 0-1 across the candidate sites, so risk_j is on a 0-1 scale.

    risk saved > 1 / λ
"""

import os
import time
import requests
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
dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')

df_bushfire = pd.read_csv('vertiport_bushfire_risk.csv')
df_rain = pd.read_excel('vertiport_rain.xlsx')
df_wind = pd.read_csv('vertiport_wind_risk.csv').set_index('vertiport')

candidate_coords  = dict(zip(dfi['name'],     zip(dfi['latitude'], dfi['longitude'])))
demand_coords     = dict(zip(dfj['SA2_Name'], zip(dfj['latitude'], dfj['longitude'])))
hospital_coords   = dict(zip(dfh['Name'],     zip(dfh['latitude'], dfh['longitude'])))
demand_population = dict(zip(dfj['SA2_Name'], dfj['population']))
site_coords       = dict(candidate_coords)   # candidate vertiports + stage 2 new pad sites

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────

UAV_SPEED_KMH      = 250
TIME_THRESHOLD     = 75
COVERAGE_THRESHOLD = 1

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

# --- solver ---
SOLVER_NAME = os.environ.get('VERTIPORT_SOLVER', 'cbc')  

# ─────────────────────────────────────────────────────────────
# STAGE 2 PARAMETERS 
# ─────────────────────────────────────────────────────────────
RUN_STAGE2 = True

# Stage 2 runs only if the existing sites cannot reach this share of SA2s
# Coverage target for Stage 2: same as Stage 1
STAGE2_TARGET = COVERAGE_THRESHOLD

# 'unreachable' : new helipads are only credited with covering SA2s that existing sites cannot reach
# 'all'         : new helipads are credited for every SA2
STAGE2_SA2_SCOPE = 'unreachable'

# --- all-NSW hospital file (candidate helipad locations) ---
ALL_HOSPITALS_FILE   = 'All_NSW_hospitals_with_lat_lon.xlsx'
ALL_HOSP_NAME_COL    = 'Name'
ALL_HOSP_LAT_COL     = 'latitude'
ALL_HOSP_LON_COL     = 'longitude'

# A hospital can only become a flight DESTINATION if it has an emergency department.
ALL_HOSP_EMERGENCY_COL          = None
EMERGENCY_TRUE_VALUES           = {'yes', 'y', 'true', '1', '1.0'}
TREAT_ALL_AS_EMERGENCY_CAPABLE  = True

# --- cost of building new infrastructure ---
# Added to the objective for every NEW helipad, on top of the usual (1 + lambda*risk) if it is also selected
NEW_HELIPAD_PENALTY = 2.0

POP_TIEBREAK = 0.0

# --- pool construction ---
DEDUP_KM       = 0.3     # drop hospitals within this distance of an existing helipad hospital / candidate vertiport
POOL_RADIUS_KM = None    # optional extra cap: only hospitals within this distance of an unreachable SA2

# --- open-space candidate sites (parks, showgrounds, sports grounds, ovals) ---
# Adds open public land as extra TAKE-OFF vertiport candidates alongside the hospitals.
# as a flight DESTINATION
INCLUDE_OPENSPACE        = True
OPENSPACE_FILE           = 'nsw_openspace.csv'   
# Open-space pads carry NO penalty of any kind: no build penalty and no risk cost.
OPENSPACE_BUILD_PENALTY  = 0.0
OPENSPACE_DEFAULT_RISK   = 0.0
# An open space is allowed as a destination only if it is within this road distance
# of an emergency hospital
OPENSPACE_DEST_DIST_KM   = 0.5

# --- road times for new sites ---
# New-site road times come from OSRM
OSRM_URL              = 'http://router.project-osrm.org'
STAGE2_ROAD_CACHE     = 'd_ij_stage2_cache.xlsx'
MAX_ROAD_SPEED_KMH    = 110    # used ONLY as a lower bound to skip pairs that cannot possibly be in range
EST_ROAD_SPEED_KMH    = 70     # fallback estimate (only if OSRM fails): average road speed ...
EST_ROAD_CIRCUITY     = 1.3    # ... and road-distance / straight-line ratio

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
    return haversine_km(*site_coords[j], *site_coords[k]) / UAV_SPEED_KMH * 60

# ─────────────────────────────────────────────────────────────
# LOAD MATRICES
# ─────────────────────────────────────────────────────────────

dfh = pd.read_excel('Emergency_hospitals_with_lat_lon.xlsx')

d_ih = pd.read_excel('vertiport_hospital_distances.xlsx')
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
    for i in candidate_coords:          # i = vertiport
        for j in demand_coords:         # j = SA2
            a[i, j] = 1 if road_time_lookup[j, i] + air_time[i] <= T else 0
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
    I = list(candidate_coords.keys())   # vertiports
    J = list(demand_coords.keys())      # SA2s
    n_req = ceil(alpha * len(J) - 1e-9)

    mod = ConcreteModel()
    mod.I = Set(initialize=I)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.I, domain=Binary)   # vertiport i is built
    mod.z = Var(mod.J, domain=Binary)   # SA2 j is covered

    mod.obj = Objective(
        expr=sum(mod.y[i] for i in I) + lam * sum(cost[i] * mod.y[i] for i in I),
        sense=minimize
    )

    # SA2 j can only be covered if a selected vertiport reaches it in time
    mod.cov = Constraint(mod.J, rule=lambda mod, j:
        mod.z[j] <= sum(a[i, j] * mod.y[i] for i in I))

    mod.demand = Constraint(expr=sum(mod.z[j] for j in J) >= n_req)

    result = SolverFactory(SOLVER_NAME).solve(mod)

    if result.solver.termination_condition == TerminationCondition.optimal:
        selected  = [i for i in I if value(mod.y[i]) > 0.5]
        covered   = [j for j in J if any(a[i, j] == 1 for i in selected)]
        cov_set   = set(covered)
        uncovered = [j for j in J if j not in cov_set]
        return 'optimal', selected, covered, uncovered
    else:
        return 'infeasible', [], [], []

# ─────────────────────────────────────────────────────────────
# STAGE 2: HELPERS FOR HOSPITAL HELIPADS
# ─────────────────────────────────────────────────────────────
# Vectorised haversine. lat1/lon1 have length n, lat2/lon2 length m -> (n, m) array of km
def hav_matrix_km(lat1, lon1, lat2, lon2):  
    la1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lo1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    la2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lo2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))

# All-NSW hospitals -> de-duplicated candidate helipad sites (DataFrame: site, hospital, lat, lon, dest_ok)
def load_hospital_pool():
    path = ALL_HOSPITALS_FILE
    dfa = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path)
    need = [ALL_HOSP_NAME_COL, ALL_HOSP_LAT_COL, ALL_HOSP_LON_COL]
    miss = [c for c in need if c not in dfa.columns]
    if miss:
        raise KeyError(f"{path}: column(s) {miss} not found. Columns present: {list(dfa.columns)}. "
                       f"Edit the ALL_HOSP_*_COL settings at the top of the script.")

    dfa = dfa.copy()
    dfa[ALL_HOSP_LAT_COL] = pd.to_numeric(dfa[ALL_HOSP_LAT_COL], errors='coerce')
    dfa[ALL_HOSP_LON_COL] = pd.to_numeric(dfa[ALL_HOSP_LON_COL], errors='coerce')
    n_raw = len(dfa)
    dfa = dfa.dropna(subset=[ALL_HOSP_NAME_COL, ALL_HOSP_LAT_COL, ALL_HOSP_LON_COL])
    dfa[ALL_HOSP_NAME_COL] = dfa[ALL_HOSP_NAME_COL].astype(str).str.strip()
    dfa = dfa.drop_duplicates(subset=[ALL_HOSP_NAME_COL, ALL_HOSP_LAT_COL, ALL_HOSP_LON_COL]).reset_index(drop=True)

    if ALL_HOSP_EMERGENCY_COL:
        if ALL_HOSP_EMERGENCY_COL not in dfa.columns:
            raise KeyError(f"{path}: emergency column '{ALL_HOSP_EMERGENCY_COL}' not found. Columns: {list(dfa.columns)}")
        dest_ok = dfa[ALL_HOSP_EMERGENCY_COL].astype(str).str.strip().str.lower().isin(EMERGENCY_TRUE_VALUES)
    else:
        dest_ok = pd.Series(bool(TREAT_ALL_AS_EMERGENCY_CAPABLE), index=dfa.index)
        if not TREAT_ALL_AS_EMERGENCY_CAPABLE:
            print("\n  NOTE: no emergency-department column set -> new helipads can be take-off sites only, "
                  "not flight destinations (see ALL_HOSP_EMERGENCY_COL).")
        else:
            print("\n  WARNING: every hospital in the file is treated as emergency-capable (TREAT_ALL_AS_EMERGENCY_CAPABLE).")

    # drop hospitals that already are an existing helipad hospital or candidate vertiport
    ex = list(hospital_coords.values()) + list(candidate_coords.values())
    D = hav_matrix_km(dfa[ALL_HOSP_LAT_COL], dfa[ALL_HOSP_LON_COL], [c[0] for c in ex], [c[1] for c in ex])
    dup = D.min(axis=1) <= DEDUP_KM
    dfa, dest_ok = dfa[~dup].reset_index(drop=True), dest_ok[~dup].reset_index(drop=True)

    taken = set(candidate_coords) | set(hospital_coords)
    sites, seen = [], set()
    for nm in dfa[ALL_HOSP_NAME_COL]:
        cand, k = nm, 1
        while cand in taken or cand in seen:
            k += 1
            cand = f"{nm} [new helipad]" if k == 2 else f"{nm} [new helipad {k - 1}]"
        seen.add(cand)
        sites.append(cand)

    return pd.DataFrame({
        'site':     sites,
        'hospital': dfa[ALL_HOSP_NAME_COL].values,
        'lat':      dfa[ALL_HOSP_LAT_COL].values,
        'lon':      dfa[ALL_HOSP_LON_COL].values,
        'dest_ok':  dest_ok.values.astype(bool),
    })


# ─────────────────────────────────────────────────────────────
# STAGE 2 HELPERS FOR OPEN-SPACE CANDIDATE POOL
# ─────────────────────────────────────────────────────────────
# Read from the downloaded OPENSPACE_FILE.
# dest_hosp_coords = list of (lat, lon) of hospitals an open space may count as a destination if near.
def load_openspace_pool(dest_hosp_coords):
    if not os.path.exists(OPENSPACE_FILE):
        print(f"  WARNING: OPENSPACE_FILE '{OPENSPACE_FILE}' not found -> skipping open-space sites.")
        return pd.DataFrame(columns=['site', 'hospital', 'lat', 'lon', 'dest_ok'])

    f = pd.read_csv(OPENSPACE_FILE) if OPENSPACE_FILE.lower().endswith('.csv') else pd.read_excel(OPENSPACE_FILE)
    # accept overpass-turbo headers (@lat/@lon) and common variants, case-insensitively
    f.columns = [str(c).strip().lstrip('@').lower() for c in f.columns]
    f = f.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon'})
    if 'name' not in f.columns:
        f['name'] = ''
    f['name'] = f['name'].fillna('').replace('', 'open space')
    raw = f[['name', 'lat', 'lon']].copy()

    raw = raw.dropna(subset=['lat', 'lon']).drop_duplicates(subset=['name', 'lat', 'lon']).reset_index(drop=True)
    if not len(raw):
        return pd.DataFrame(columns=['site', 'hospital', 'lat', 'lon', 'dest_ok'])

    # drop any that sit on top of an existing candidate vertiport (keep ones near hospitals)
    ex = list(candidate_coords.values())
    D = hav_matrix_km(raw['lat'], raw['lon'], [c[0] for c in ex], [c[1] for c in ex])
    dup = D.min(axis=1) <= DEDUP_KM
    raw = raw[~dup].reset_index(drop=True)

    # unique site names, keeping clear of hospital-pool / existing names
    taken = set(candidate_coords) | set(hospital_coords)
    sites, seen = [], set()
    for nm in raw['name'].astype(str).str.strip():
        base, cand, k = f"{nm}", f"{nm}", 1
        while cand in taken or cand in seen:
            k += 1
            cand = f"{base} {k}"
        seen.add(cand)
        sites.append(cand)

    # destination eligibility: an open space within OPENSPACE_DEST_DIST_KM road-km of a hospital that can
    # act as a destination may also be a flight destination
    dest_ok = np.zeros(len(raw), dtype=bool)
    if dest_hosp_coords:
        hmat = hav_matrix_km(raw['lat'], raw['lon'],
                             [c[0] for c in dest_hosp_coords], [c[1] for c in dest_hosp_coords])
        hav_near = hmat.min(axis=1)
        cand = np.nonzero(np.asarray(hav_near) <= OPENSPACE_DEST_DIST_KM)[0]
        if len(cand):
            nearest = hmat[cand].argmin(axis=1)                       # nearest hospital per candidate
            uniq = sorted({int(x) for x in nearest})
            dst = [dest_hosp_coords[u] for u in uniq]                 # only the hospitals actually nearby
            try:
                osrm_km = _osrm_distance_km(list(cand),
                                            [(raw['lat'].iloc[i], raw['lon'].iloc[i]) for i in cand], dst)
                for i in cand:
                    if osrm_km.get(i, np.inf) <= OPENSPACE_DEST_DIST_KM:
                        dest_ok[i] = True
                print(f"  Open space: {int(dest_ok.sum())} within {OPENSPACE_DEST_DIST_KM} km (road) of a "
                      f"hospital -> usable as destinations.")
            except Exception as e:
                print(f"  WARNING: OSRM distance check for open-space destinations failed ({e}); "
                      f"none enabled this run.")

    return pd.DataFrame({
        'site':     sites,
        'hospital': raw['name'].astype(str).str.strip().values,   # descriptive label only; not an emergency dept
        'lat':      raw['lat'].values,
        'lon':      raw['lon'].values,
        'dest_ok':  dest_ok,                                       # True = near a hospital, can be a destination
    })

# OSRM driving-time table (minutes) from each SA2 centroid to each new site
def _osrm_table(sa2_keys, sa2_vals, site_keys, site_vals, chunk_size=25):
    out = {}
    for i0 in range(0, len(sa2_keys), chunk_size):
        ik, iv = sa2_keys[i0:i0 + chunk_size], sa2_vals[i0:i0 + chunk_size]
        for j0 in range(0, len(site_keys), chunk_size):
            jk, jv = site_keys[j0:j0 + chunk_size], site_vals[j0:j0 + chunk_size]
            all_coords = list(iv) + list(jv)
            coords_str = ';'.join(f"{lon},{lat}" for lat, lon in all_coords)
            n_i = len(ik)
            sources      = ';'.join(str(k) for k in range(n_i))
            destinations = ';'.join(str(k) for k in range(n_i, n_i + len(jk)))
            url = (f"{OSRM_URL}/table/v1/driving/{coords_str}"
                   f"?sources={sources}&destinations={destinations}&annotations=duration")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            durations = resp.json()['durations']
            for a, an in enumerate(ik):
                for b, bn in enumerate(jk):
                    d = durations[a][b]
                    out[(an, bn)] = np.inf if d is None else d / 60.0
            time.sleep(0.3)   # be gentle on the public OSRM server
    return out

# OSRM driving DISTANCE (km): min road distance from each source to its nearest destination.
def _osrm_distance_km(src_keys, src_vals, dst_vals, chunk_size=25):
    best = {k: np.inf for k in src_keys}
    for i0 in range(0, len(src_keys), chunk_size):
        ik, iv = src_keys[i0:i0 + chunk_size], src_vals[i0:i0 + chunk_size]
        for j0 in range(0, len(dst_vals), chunk_size):
            jv = dst_vals[j0:j0 + chunk_size]
            all_coords = list(iv) + list(jv)
            coords_str = ';'.join(f"{lon},{lat}" for lat, lon in all_coords)
            n_i = len(ik)
            sources      = ';'.join(str(k) for k in range(n_i))
            destinations = ';'.join(str(k) for k in range(n_i, n_i + len(jv)))
            url = (f"{OSRM_URL}/table/v1/driving/{coords_str}"
                   f"?sources={sources}&destinations={destinations}&annotations=distance")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dists = resp.json()['distances']   # metres
            for a, k in enumerate(ik):
                row = [d for d in dists[a] if d is not None]
                if row:
                    best[k] = min(best[k], min(row) / 1000.0)
            time.sleep(0.3)
    return best

# Road time (min) from each SA2 centroid to each new site
def get_road_times(needed, coords_by_site, sa2_pts, client=None):
    cache = {}
    if os.path.exists(STAGE2_ROAD_CACHE):
        c = pd.read_excel(STAGE2_ROAD_CACHE)
        cache = {(r.sa2, r.vertiport): r.time_min for r in c.itertuples()}

    times, source, missing = {}, {}, {}
    for s, js in needed.items():
        for j in js:
            if (j, s) in cache:
                v = cache[(j, s)]
                times[(j, s)] = np.inf if pd.isna(v) else float(v)
                source[(j, s)] = 'osrm (cached)'
            else:
                missing.setdefault(s, []).append(j)

    if missing:
        sites = list(missing.keys())
        sa2s  = sorted({j for js in missing.values() for j in js})
        try:
            tbl = _osrm_table(sa2s, [sa2_pts[j] for j in sa2s], sites, [coords_by_site[s] for s in sites])
            rows = []
            for s, js in missing.items():
                for j in js:
                    t = tbl.get((j, s), np.inf)
                    times[(j, s)] = t
                    source[(j, s)] = 'osrm'
                    rows.append({'sa2': j, 'vertiport': s,
                                 'time_min': (np.nan if np.isinf(t) else t), 'source': 'osrm'})
            if rows:
                old = pd.read_excel(STAGE2_ROAD_CACHE) if os.path.exists(STAGE2_ROAD_CACHE) else pd.DataFrame()
                pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_excel(STAGE2_ROAD_CACHE, index=False)
        except Exception as e:
            n_el = sum(len(v) for v in missing.values())
            print(f"  WARNING: OSRM road-time lookup failed ({e}) -> {n_el} road times ESTIMATED "
                  f"(straight line x {EST_ROAD_CIRCUITY} at {EST_ROAD_SPEED_KMH} km/h). Treat stage-2 results as indicative.")
            for s, js in missing.items():
                for j in js:
                    km = haversine_km(*sa2_pts[j], *coords_by_site[s])
                    times[(j, s)] = km * EST_ROAD_CIRCUITY / EST_ROAD_SPEED_KMH * 60
                    source[(j, s)] = 'estimate'
    return times, source


# risk cost of a NEW site: same formulas as stage 1, normalised with the stage-1 bounds
_raw_bf   = {j: bfpl_severity(j) for j in candidate_coords}
_raw_wind = {j: wind_severity(j) for j in candidate_coords}
_raw_rain = {j: rain_severity(j) for j in candidate_coords}
_bounds = {k: (min(v.values()), max(v.values())) for k, v in
           {'bf': _raw_bf, 'wind': _raw_wind, 'rain': _raw_rain}.items()}

def _norm_bounded(v, key):
    lo, hi = _bounds[key]
    return min(max((v - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0

def new_site_cost_parts(s):
    """(bushfire, wind, rain) contributions to the 0-1 combined risk of a new site (already weighted)."""
    disc = min(MAX_BUSHFIRE_DISCOUNT, EMERGENCY_VALUE_PER_SITE * emergency_reach_count(s))
    bfc  = _norm_bounded(bfpl_severity(s), 'bf') * (1 - disc)
    has_backup = any(flight_min(s, k) <= ALTERNATE_TIME_THRESHOLD for k in safe_vertiports if k != s)
    wc   = _norm_bounded(wind_severity(s), 'wind') * (1 - (WIND_BACKUP_DISCOUNT if has_backup else 0.0))
    rc   = _norm_bounded(rain_severity(s), 'rain')
    return (WEIGHT_BUSHFIRE * bfc / W_SUM, WEIGHT_WIND * wc / W_SUM, WEIGHT_RAIN * rc / W_SUM)

def new_site_cost(s):
    return sum(new_site_cost_parts(s))

def risk_parts(port):
    """(bushfire, wind, rain) weighted contributions for any site, existing or new."""
    if port in candidate_coords:
        return (WEIGHT_BUSHFIRE * bushfire_cost(port) / W_SUM,
                WEIGHT_WIND     * wind_cost(port)     / W_SUM,
                WEIGHT_RAIN     * rain_cost(port)     / W_SUM)
    return new_site_cost_parts(port)

def has_missing_risk_data(s):
    return (s not in df_bushfire.index) or (s not in df_wind.index) or (s not in df_rain.index)


# ─────────────────────────────────────────────────────────────
# STAGE 2: EXTENDED PYOMO MODEL
# ─────────────────────────────────────────────────────────────
# New decision variables on top of stage 1:
#   q[p] = 1  a new helipad is built at hospital p        (cost NEW_HELIPAD_PENALTY)
#   y[p] <= q[p]  p can only be operated as a vertiport if its helipad is built
#   x[i,h] <= y[i], x[i,h] <= q[h]   vertiport i can fly to NEW destination helipad h
# SA2 j is covered if
#   (a) a selected site reaches it via an EXISTING destination hospital (stage-1 rule), or
#   (b) a selected site i has road(j,i) + flight(i,h) <= T to a newly built destination helipad h.
def run_model_extended(J, I_all, P, base, pair_cov, cost_all, popshare, n_req, lam, q_pen, pop_eps):
    pairs = list(pair_cov)
    pairs_for = {j: [] for j in J}
    for pr, js in pair_cov.items():
        for j in js:
            pairs_for[j].append(pr)

    mod = ConcreteModel()
    mod.I  = Set(initialize=I_all)
    mod.P  = Set(initialize=P)
    mod.J  = Set(initialize=J)
    mod.PR = Set(initialize=pairs, dimen=2)
    mod.y = Var(mod.I,  domain=Binary)
    mod.q = Var(mod.P,  domain=Binary)
    mod.z = Var(mod.J,  domain=Binary)
    mod.x = Var(mod.PR, bounds=(0, 1))

    pen = q_pen if isinstance(q_pen, dict) else {p: q_pen for p in P}
    mod.obj = Objective(
        expr=sum(mod.y[i] * (1 + lam * cost_all[i]) for i in I_all)
             + sum(pen[p] * mod.q[p] for p in P)
             - pop_eps * sum(popshare[j] * mod.z[j] for j in J),
        sense=minimize)

    mod.link = Constraint(mod.P,  rule=lambda m, p: m.y[p] <= m.q[p])
    mod.xy   = Constraint(mod.PR, rule=lambda m, i, h: m.x[i, h] <= m.y[i])
    mod.xq   = Constraint(mod.PR, rule=lambda m, i, h: m.x[i, h] <= m.q[h])
    mod.cov  = Constraint(mod.J,  rule=lambda m, j:
        m.z[j] <= sum(m.y[i] for i in base[j]) + sum(m.x[pr] for pr in pairs_for[j]))
    mod.demand = Constraint(expr=sum(mod.z[j] for j in J) >= n_req)

    result = SolverFactory(SOLVER_NAME).solve(mod)
    if result.solver.termination_condition != TerminationCondition.optimal:
        return None

    val = lambda v: 0.0 if v.value is None else v.value
    sel   = [i for i in I_all if val(mod.y[i]) > 0.5]
    built = [p for p in P     if val(mod.q[p]) > 0.5]
    dest_used = {h for (i, h) in pairs if val(mod.x[i, h]) > 0.5}
    return sel, built, dest_used

# ─────────────────────────────────────────────────────────────
# MAIN MODEL  -  STAGE 1
# ─────────────────────────────────────────────────────────────
a = build_coverage_matrix(demand_coords, candidate_coords, road_time_lookup, d_ih, UAV_SPEED_KMH, TIME_THRESHOLD)

cost = {j: combined_cost(j) for j in candidate_coords}
coverable = [j for j in demand_coords if any(a[i, j] == 1 for i in candidate_coords)]

print(f"\nMax achievable coverage: {len(coverable)}/{len(demand_coords)} = {len(coverable)/len(demand_coords)*100:.1f}%")
print(f"Requested coverage: {COVERAGE_THRESHOLD*100}%")

status, selected, covered, uncovered = run_model(a, demand_coords, candidate_coords, COVERAGE_THRESHOLD, cost)
stage1_met = (status == 'optimal')
if not stage1_met:
    # target not reachable with existing sites: solve for the best achievable coverage instead,
    # so there is an output to inspect (and to hand to stage 2)
    _, selected, covered, uncovered = run_model(
        a, demand_coords, candidate_coords, len(coverable) / len(demand_coords), cost)

# ─────────────────────────────────────────────────────────────
# MAIN MODEL : STAGE 2
# ─────────────────────────────────────────────────────────────
# Build a candidate pool of new helipads from the all-NSW hospital list, and re-optimise
def run_stage2(a, coverable, client=None):
    J, I0, T = list(demand_coords), list(candidate_coords), TIME_THRESHOLD
    cov_set  = set(coverable)
    total_pop = float(sum(demand_population[j] for j in J))
    target = sorted((j for j in J if j not in cov_set), key=lambda j: -demand_population[j])
    S = target if STAGE2_SA2_SCOPE == 'unreachable' else list(J)
    S_idx = {j: m for m, j in enumerate(S)}

    # candidate pool 
    os_sites = set()
    pool_all = load_hospital_pool()
    if INCLUDE_OPENSPACE:
        # hospitals an open space may count as a destination if nearby
        # plus the all-NSW hospital pool (the hospitals the model can build as new destinations)
        dest_hosp = list(hospital_coords.values()) + list(zip(
            pool_all.loc[pool_all.dest_ok, 'lat'], pool_all.loc[pool_all.dest_ok, 'lon']))
        openspace = load_openspace_pool(dest_hosp)
        if len(openspace):
            os_sites = set(openspace['site'])
            pool_all = pd.concat([pool_all, openspace], ignore_index=True)
            pool_all = pool_all.drop_duplicates(subset=['site']).reset_index(drop=True)
    pool_all.to_csv('stage2_candidate_pool.csv', index=False)  

    vmax = max(UAV_SPEED_KMH, MAX_ROAD_SPEED_KMH)
    near = hav_matrix_km(pool_all.lat, pool_all.lon,
                         [demand_coords[j][0] for j in target], [demand_coords[j][1] for j in target]).min(axis=1)
    keep = near <= vmax * T / 60            # a site farther than this from every unreachable SA2 cannot help in T minutes
    if POOL_RADIUS_KM is not None:
        keep &= near <= POOL_RADIUS_KM
    pool = pool_all[keep].reset_index(drop=True)
    P = pool['site'].tolist()
    P_idx = {s: k for k, s in enumerate(P)}
    if not P:
        print("ERROR: no candidate sites in range -> stage 2 cannot add anything.")
        return None
    for s, la, lo in zip(pool.site, pool.lat, pool.lon):
        site_coords[s] = (la, lo)

    # flight times 
    h_xy   = list(hospital_coords.values())
    air0_p = hav_matrix_km(pool.lat, pool.lon, [c[0] for c in h_xy], [c[1] for c in h_xy]).min(axis=1) / UAV_SPEED_KMH * 60

    Hd   = [k for k in range(len(P)) if pool.dest_ok[k]]            # destination-eligible pool sites
    Hn   = [P[k] for k in Hd]
    pxy  = list(zip(pool.lat, pool.lon))
    F_pp = hav_matrix_km(pool.lat, pool.lon, [pool.lat[k] for k in Hd], [pool.lon[k] for k in Hd]) / UAV_SPEED_KMH * 60
    F_0  = hav_matrix_km([candidate_coords[i][0] for i in I0], [candidate_coords[i][1] for i in I0],
                         [pool.lat[k] for k in Hd], [pool.lon[k] for k in Hd]) / UAV_SPEED_KMH * 60

    # road times for new sites
    D_ps   = hav_matrix_km(pool.lat, pool.lon, [demand_coords[j][0] for j in S], [demand_coords[j][1] for j in S])
    air_lb = air0_p.copy()
    if Hd:
        air_lb = np.minimum(air_lb, F_pp.min(axis=1))               # includes 0 for a site that is itself a destination
    in_range = (D_ps / MAX_ROAD_SPEED_KMH * 60 + air_lb[:, None]) <= T
    needed = {P[k]: [S[m] for m in np.nonzero(in_range[k])[0]] for k in range(len(P)) if in_range[k].any()}
    road, road_src = get_road_times(needed, {s: site_coords[s] for s in needed}, {j: demand_coords[j] for j in S}, client)

    Rp = np.full((len(P), len(S)), np.inf)
    for (j, s), t in road.items():
        Rp[P_idx[s], S_idx[j]] = t
    A0p = (Rp + air0_p[:, None]) <= T                                # new site -> EXISTING destination

    R0  = np.array([[road_time_lookup[(j, i)] for j in S] for i in I0], dtype=float)
    A0e = np.array([[a[i, j] for j in S] for i in I0], dtype=bool)

    # coverage via NEW destination helipads 
    pair_cov = {}
    for m, h in enumerate(Hn):
        new_e = (R0 + F_0[:, [m]] <= T) & ~A0e
        for r in np.nonzero(new_e.any(axis=1))[0]:
            pair_cov.setdefault((I0[r], h), []).extend(S[c] for c in np.nonzero(new_e[r])[0])
        new_p = (Rp + F_pp[:, [m]] <= T) & ~A0p
        for r in np.nonzero(new_p.any(axis=1))[0]:
            pair_cov.setdefault((P[r], h), []).extend(S[c] for c in np.nonzero(new_p[r])[0])

    # ceiling with new infrastructure
    ext = set(cov_set)
    ext.update(S[m] for m in range(len(S)) if A0p[:, m].any())
    for js in pair_cov.values():
        ext.update(js)
    n_target = ceil(max(COVERAGE_THRESHOLD, STAGE2_TARGET) * len(J) - 1e-9)
    n_req = min(n_target, len(ext))
    if n_req < n_target:
        print(f"WARNING: target of {n_target} SA2s cannot be reached even with every candidate site; "
              f"solving for the achievable maximum ({n_req}).")

    # costs and solve 
    # open-space sites carry no risk cost 
    def _new_cost(s):
        if s in os_sites and OPENSPACE_DEFAULT_RISK is not None:
            return OPENSPACE_DEFAULT_RISK
        return new_site_cost(s)
    cost_new = {s: _new_cost(s) for s in P}
    cost_all = {**cost, **cost_new}
    q_pen = {s: (OPENSPACE_BUILD_PENALTY if s in os_sites else NEW_HELIPAD_PENALTY) for s in P}
    base = {j: [] for j in J}
    for i in I0:
        for j in J:
            if a[i, j]:
                base[j].append(i)
    for k, s in enumerate(P):
        for m, j in enumerate(S):
            if A0p[k, m]:
                base[j].append(s)
    popshare = {j: demand_population[j] / total_pop for j in J}

    out = run_model_extended(J, I0 + P, P, base, pair_cov, cost_all, popshare, n_req,
                             RISK_LAMBDA, q_pen, POP_TIEBREAK)
    if out is None:
        print("STAGE 2 INFEASIBLE / solver did not reach optimality.")
        return None
    sel, built, dest_used = out
    
    used = set(sel) | set(dest_used)
    built = [p for p in built if p in used]
    sel_set, built_set = set(sel), set(built)

    # recompute coverage directly
    pairs_for = {j: [] for j in J}
    for pr, js in pair_cov.items():
        for j in js:
            pairs_for[j].append(pr)
    served_by = {}
    for j in J:
        who = {i for i in base[j] if i in sel_set}
        for (i, h) in pairs_for[j]:
            if i in sel_set and h in built_set:
                who.update([h] + ([i] if i in P_idx else []))
        if who:
            served_by[j] = who
    covered   = [j for j in J if j in served_by]
    uncovered = [j for j in J if j not in served_by]

    new_cov = [j for j in covered if j not in cov_set]              # SA2s that only the new infrastructure reaches
    newly_pop = sum(demand_population[j] for j in new_cov)

    # report 
    rows = []
    for p in built:
        role = ' + '.join(r for r, on in (('take-off vertiport', p in sel_set), ('destination', p in dest_used)) if on) or 'helipad'
        mine = [j for j in new_cov if p in served_by[j]]
        k = P_idx[p]
        srcs = {road_src[(j, p)] for j in S if (j, p) in road_src}
        rows.append({
            'site': p, 'type': 'park' if p in os_sites else 'helipad',
            'hospital': pool.hospital[k], 'lat': pool.lat[k], 'lon': pool.lon[k], 'role': role,
            'risk_cost_0_1': round(cost_new[p], 3),
            'risk_data_missing': has_missing_risk_data(p),
            'road_time_source': ', '.join(sorted(srcs)) if srcs else 'n/a',
            'new_sa2_served': len(mine),
            'new_population_served': int(sum(demand_population[j] for j in mine)),
        })
    new_df = pd.DataFrame(rows)

    def _print_table(headers, table_rows, aligns=None):
        aligns = aligns or ['<'] * len(headers)   # '<' left, '>' right (per column)
        w = [max([len(str(h))] + [len(str(r[c])) for r in table_rows]) for c, h in enumerate(headers)]
        line = lambda cells: '  ' + '  '.join(f"{str(cells[c]):{aligns[c]}{w[c]}}" for c in range(len(headers)))
        print(line(headers))
        print('  ' + '-' * (sum(w) + 2 * (len(headers) - 1)))
        for r in table_rows:
            print(line(r))

    if len(new_df):
        new_df = new_df.sort_values('new_population_served', ascending=False).reset_index(drop=True)
        new_df.to_csv('stage2_new_helipads.csv', index=False)

        # Table 1: new sites (helipads + parks) and their role
        print(f"\nStage 2 added {len(built)} new sites -> coverage {len(covered)}/{len(J)} SA2s "
              f"({sum(demand_population[j] for j in covered) / total_pop * 100:.1f}% of population)\n")
        print("New sites to build:")
        _print_table(['Site', 'Type', 'Role'],
                     [[r['site'], r['type'], r['role']] for r in new_df.to_dict('records')])

        # Table 2: SA2s newly covered by the stage-2 sites, with population and which new site serves them
        print("\nSA2 regions newly covered:")
        cov_rows = []
        for j in sorted(new_cov, key=lambda j: -demand_population[j]):
            by = ', '.join(sorted(s for s in served_by[j] if s in set(built))) or '-'
            cov_rows.append([j, f"{demand_population[j]:,}", by])
        _print_table(['SA2 region', 'Population', 'Covered by'], cov_rows, aligns=['<', '>', '<'])

        n_miss = int(new_df.risk_data_missing.sum())
        if n_miss:
            print(f"\nNOTE: {n_miss} new site(s) have no bushfire/wind/rain data (worst-case default applied; "
                  f"append them to the risk files by site name to fix).")
        if (new_df.road_time_source == 'estimate').any():
            print("NOTE: some new-site road times are straight-line estimates, not OSRM routing.")

    pd.DataFrame([{'sa2': j, 'population': demand_population[j],
                   'lat': demand_coords[j][0], 'lon': demand_coords[j][1],
                   'covered_stage2': j in served_by,
                   'reachable_from_existing_sites': j in cov_set,
                   'served_by': '; '.join(sorted(served_by.get(j, [])))}
                  for j in sorted(J, key=lambda j: -demand_population[j])]
                 ).to_csv('stage2_sa2_coverage.csv', index=False)
    print("\nSaved: stage2_new_helipads.csv, stage2_sa2_coverage.csv, stage2_candidate_pool.csv")

    if uncovered:
        print(f"\nStill uncovered ({len(uncovered)}): "
              + ', '.join(sorted(uncovered, key=lambda j: -demand_population[j])))

    roles = {r['site']: r['role'] for r in rows}
    return {'selected': sel, 'built': built, 'roles': roles, 'covered': covered, 'uncovered': uncovered,
            'new_covered': new_cov, 'new_df': new_df, 'openspace': os_sites & set(built)}

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
TOTAL_POP = float(sum(demand_population.values()))

def report_solution(sel, cov, unc, title, list_uncovered=True):
    print(f"\n{title}")
    header = f"  {'Site':<55} {'Bushfire':>10} {'Wind':>10} {'Rain':>10} {'Total':>10}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    total_risk = 0.0
    for port in sel:
        bc, wc, rc = risk_parts(port)
        total_risk += bc + wc + rc
        print(f"  {port:<55} {bc:>10.3f} {wc:>10.3f} {rc:>10.3f} {bc+wc+rc:>10.3f}")
    pop_cov = sum(demand_population[j] for j in cov)
    print(f"\nTotal risk         : {total_risk:.3f}")
    print(f"Regions covered    : {len(cov)} / {len(demand_coords)}   (population {pop_cov / TOTAL_POP * 100:.1f}%)")
    if list_uncovered:
        print("Uncovered regions  :")
        for reg in unc:
            print('\t', reg)

print(f"\nTime threshold  = {TIME_THRESHOLD} min")
print(f"Vehicle speed   = {UAV_SPEED_KMH} km/h")
print(f"Coverage target = {COVERAGE_THRESHOLD*100}%")
print(f"Risk weights    = Bushfire:{WEIGHT_BUSHFIRE} Wind:{WEIGHT_WIND} Rain:{WEIGHT_RAIN} | lambda={RISK_LAMBDA}\n")

if stage1_met:
    print(f"Minimum vertiports : {len(selected)}")
    print("Selected sites:")
    report_solution(selected, covered, uncovered, "STAGE 1 (existing sites)")
else:
    print("INFEASIBLE: the coverage target cannot be met with the existing candidate sites.")
    print(f"Showing the best achievable instead ({len(coverable)} SA2s).")
    report_solution(selected, covered, uncovered, "STAGE 1 (existing sites, best achievable)", list_uncovered=False)
    print(f"Unreachable regions: {len(uncovered)} (listed in stage 2 below if it runs)")

stage2 = None
ceiling = len(coverable) / len(demand_coords)
if RUN_STAGE2:
    if ceiling < STAGE2_TARGET - 1e-9:
        stage2 = run_stage2(a, coverable)
    else:
        print(f"\nStage 2 not needed: existing sites already reach {ceiling*100:.1f}% "
              f"(>= STAGE2_TARGET {STAGE2_TARGET*100:.1f}%).")
        
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
                'lambda':       lam,
                'n_vertiports': len(sel),
                'total_risk':   round(total_risk, 3),
                'avg_risk':     round(avg_risk, 3),
                'status':       'optimal'
            })
            print(f"{lam:<10} {len(sel):>12} {total_risk:>12.3f} {avg_risk:>15.3f}")
        else:
            lambda_results.append({'lambda': lam, 'n_vertiports': None, 'total_risk': None, 'avg_risk': None, 'status': 'infeasible'})
            print(f"{lam:<10} {'INFEASIBLE':>12}")

    # collect feasible results
    feasible = [r for r in lambda_results if r['status'] == 'optimal']
    if not feasible:
        print('No feasible lambda results (coverage target unreachable with existing sites) - plot skipped.')
    else:
        lam_vals = [r['lambda']       for r in feasible]
        n_ports  = [r['n_vertiports'] for r in feasible]
        t_risk   = [r['total_risk']   for r in feasible]

        x      = np.arange(len(lam_vals))          # even spacing kills low-lambda crowding
        labels = [f'{l:g}' for l in lam_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # --- Panel 1: count + risk vs lambda on a categorical x-axis ---
        ax1b = ax1.twinx()
        l1, = ax1.plot(x, n_ports, 'o-',  color='steelblue', zorder=3, label='Vertiports')
        l2, = ax1b.plot(x, t_risk, 's--', color='firebrick', zorder=3, label='Total risk')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_xlabel('Risk weight λ (0 = ignore risk)')
        ax1.set_ylabel('Minimum vertiports', color='steelblue')
        ax1b.set_ylabel('Total operational risk (0-1 per site, summed)', color='firebrick')
        ax1.tick_params(axis='y', colors='steelblue')
        ax1b.tick_params(axis='y', colors='firebrick')
        ax1.set_ylim(min(n_ports) - 1, max(n_ports) + 1)   # pad so a flat line isn't glued to the frame

        # label the count only when it changes -> no stacked duplicates
        prev = None
        for xi, p in zip(x, n_ports):
            if p != prev:
                ax1.annotate(str(p), (xi, p), textcoords='offset points',
                             xytext=(0, 10), ha='center', color='steelblue', fontsize=9)
            prev = p

        ax1.set_title('Vertiport count and total risk vs λ')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(handles=[l1, l2], loc='center left')

        # --- Panel 2: trade-off, one label per distinct (count, risk) outcome ---
        groups = {}
        for l, p, rk in zip(lam_vals, n_ports, t_risk):
            groups.setdefault((p, round(rk, 3)), []).append(l)

        pts = sorted(groups.items(), key=lambda kv: (kv[0][0], -kv[0][1]))   # by count, then risk high->low
        gx = [p  for (p, rk), _ in pts]
        gy = [rk for (p, rk), _ in pts]
        ax2.plot(gx, gy, 'o-', color='seagreen', zorder=3)

        for (p, rk), ls in pts:
            tag = f'λ={ls[0]:g}' if len(ls) == 1 else f'λ={min(ls):g}–{max(ls):g}'
            ax2.annotate(tag, (p, rk), textcoords='offset points',
                         xytext=(8, 8), ha='left', fontsize=9)

        ax2.set_xlabel('Vertiports')
        ax2.set_ylabel('Total operational risk')
        ax2.set_title('Trade-off: risk vs vertiport count')
        ax2.margins(x=0.25, y=0.25)                 # room for corner labels
        ax2.grid(True, linestyle='--', alpha=0.4)

        plt.suptitle(f'λ sensitivity | T={TIME_THRESHOLD} min | coverage={COVERAGE_THRESHOLD*100:.0f}%', fontsize=12)
        plt.tight_layout()
        plt.savefig('lambda_sensitivity.png', dpi=150, bbox_inches='tight')
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
    ax2.legend(title='Time Threshold', loc='center left', bbox_to_anchor=(1.02, 0.5))    
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sensitivity_analysis_combined.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to sensitivity_analysis_combined.png")


# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
SA2_SHAPEFILE = 'SA2_2021_AUST_GDA2020.shp'   # path to the ABS boundary file
SA2_NAME_COL  = 'SA2_NAME21'                   # SA2 name column in that file

_sa2_gdf = None
def _load_sa2():
    global _sa2_gdf
    if _sa2_gdf is None:
        _sa2_gdf = gpd.read_file(SA2_SHAPEFILE).to_crs(epsg=4326)
    return _sa2_gdf

def match_polygons(names, gdf):
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

def make_map(sel, cov, unc, outfile, new_helipads=None, new_roles=None, new_covered=None, openspace_sites=None):
    gdf = _load_sa2()
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

    new_cov_set = set(new_covered or [])
    covered_gdf     = match_polygons([j for j in cov if j not in new_cov_set], gdf)
    new_covered_gdf = match_polygons([j for j in cov if j in new_cov_set], gdf)
    uncovered_gdf   = match_polygons(unc, gdf)

    # uncovered regions
    if uncovered_gdf is not None:
        folium.GeoJson(
            uncovered_gdf[[SA2_NAME_COL, 'geometry']],
            style_function=lambda f: {'fillColor': '#000000', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.6},
            tooltip=folium.GeoJsonTooltip(fields=[SA2_NAME_COL], aliases=['Uncovered:'])
        ).add_to(m_map)

    # covered regions
    if covered_gdf is not None:
        folium.GeoJson(
            covered_gdf[[SA2_NAME_COL, 'geometry']],
            style_function=lambda f: {'fillColor': '#7fb8e0', 'color': '#2b7bb9', 'weight': 1.5, 'fillOpacity': 0.6},
            tooltip=folium.GeoJsonTooltip(fields=[SA2_NAME_COL], aliases=['Region:'])
        ).add_to(m_map)

    # regions covered only thanks to new helipads (stage 2)
    if new_covered_gdf is not None:
        folium.GeoJson(
            new_covered_gdf[[SA2_NAME_COL, 'geometry']],
            style_function=lambda f: {'fillColor': '#f4a261', 'color': '#e76f51', 'weight': 1.5, 'fillOpacity': 0.6},
            tooltip=folium.GeoJsonTooltip(fields=[SA2_NAME_COL], aliases=['Covered via new helipad:'])
        ).add_to(m_map)

    # selected vertiports (existing)
    new_set = set(new_helipads or [])
    for j in sel:
        if j in new_set:
            continue
        lat, lon = site_coords[j]
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(j, max_width=200),
            tooltip=j,
            icon=folium.Icon(color='red', icon='plane', prefix='fa')
        ).add_to(m_map)

    # new sites (stage 2): open-space pads get a park pin, hospital helipads keep the H pin
    os_set = set(openspace_sites or [])
    for j in (new_helipads or []):
        lat, lon = site_coords[j]
        role = (new_roles or {}).get(j, 'new helipad')
        if j in os_set:
            label, icon = 'pad', folium.Icon(color='darkgreen', icon='tree', prefix='fa')
        else:
            label, icon = 'helipad', folium.Icon(color='green', icon='h-square', prefix='fa')
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(f"NEW {label}: {j} ({role})", max_width=250),
            tooltip=f"NEW {label}: {j} ({role})",
            icon=icon
        ).add_to(m_map)

    # reachable hospitals
    for h, (h_lat, h_lon) in hospital_coords.items():
        reachable = any(d_ih.loc[j, h] / UAV_SPEED_KMH * 60 <= TIME_THRESHOLD for j in sel if j in d_ih.index)
        if reachable:
            folium.CircleMarker(
                location=[h_lat, h_lon], radius=2, color='green', fill=True, fill_color='green', fill_opacity=0.2,
                popup=folium.Popup(h, max_width=200), tooltip=h
            ).add_to(m_map)

    new_rows = '''
    <p style="margin: 5px 0;"><i style="background:#f4a261; width:12px; height:12px;
       display:inline-block; border:1px solid #e76f51;"></i> Covered via new site</p>
    <p style="margin: 5px 0;"><i class="fa fa-h-square" style="color:green; font-size:16px;
       width:12px; text-align:center;"></i> New helipad (hospital)</p>
    <p style="margin: 5px 0;"><i class="fa fa-tree" style="color:darkgreen; font-size:16px;
       width:12px; text-align:center;"></i> New pad (park)</p>
    ''' if new_helipads else ''
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 250px;
         background-color: white; border:2px solid grey; z-index:9999;
         font-size:14px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
    <p style="margin: 5px 0;"><i style="background:#7fb8e0; width:12px; height:12px;
       display:inline-block; border:1px solid #2b7bb9;"></i> Covered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i style="background:black; width:12px; height:12px;
       display:inline-block; border:1px solid black;"></i> Uncovered Regions (SA2)</p>
    <p style="margin: 5px 0;"><i class="fa fa-map-marker" style="color:red; font-size:16px;
       width:12px; text-align:center;"></i> Selected Vertiports</p>
    {new_rows}
    <p style="margin: 5px 0;"><i style="background:green; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> Reachable Hospitals</p>
    </div>
    '''
    m_map.get_root().html.add_child(folium.Element(legend_html))
    m_map.save(outfile)
    print(f"Map saved to {outfile}")

if os.path.exists(SA2_SHAPEFILE):
    make_map(selected, covered, uncovered, 'vertiports_map.html')
    if stage2 is not None:
        make_map(stage2['selected'], stage2['covered'], stage2['uncovered'], 'vertiports_map_stage2.html',
                 new_helipads=stage2['built'], new_roles=stage2['roles'], new_covered=stage2['new_covered'],
                 openspace_sites=stage2['openspace'])
else:
    print(f"Map skipped: shapefile '{SA2_SHAPEFILE}' not found.")