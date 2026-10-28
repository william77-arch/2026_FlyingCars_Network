"""
eVTOL AIRPORT-SHUTTLE vertiport location model (Greater Sydney), single direct journey

Journey:  area i --(road)--> vertiport j --(direct flight)--> Western Sydney International (WSI)

Sets
    I  demand areas in Greater Sydney: suburbs (prepare_suburb_demand.py) or SA2 regions
    J  candidate vertiports (existing sites + new-build sites)
    H  destination airport(s): WSI only (Kingsford Smith can be switched back on in AIRPORTS)

Coverage
    a_ij = 1 if, for some airport h ('any') or both airports ('both'):
             road_time(i, j) <= ROAD_MAX_MIN,
             road_time(i, j) + flight_time(j, h) <= TIME_THRESHOLD, and
             road_time(i, j) + flight_time(j, h) <= drive_time(i, h) - MIN_TIME_SAVING
           (a shuttle is only worth it if it is faster than driving to the airport)

Model
    min   sum_j (ci_j + hazard_j) * y_j
    s.t.  z_i <= sum_j a_ij * y_j                                        area i covered only if a chosen
                                                                          vertiport can serve it
          sum_i pop_i z_i            >= COVERAGE_THRESHOLD * coverable population (study area)
          sum_{i in r} pop_i z_i     >= REGION_COVERAGE   * coverable population of region r  (each r)
          y_j, z_i binary
    ci_j     cost index (1 = a typical new-build pad; existing pads are cheaper)
    hazard_j W_BUSH*bushfire_j + W_WIND*wind_j + W_RAIN*rain_j, each scored 0..1

Inputs (same folder)
    suburbs_with_coord.xlsx (or sa2_names_with_coord.xlsx)
                                                 name, latitude, longitude, population [+ region, GCC_NAME21]
    vertiport_shuttle_existing_candidates.xlsx  sheet 'candidates' (build_shuttle_vertiports_existing_sites.py)
    vertiport_shuttle_new_build_candidates.xlsx (make_new_build_candidates.py)            [optional]
    d_ij.xlsx                                    sa2, vertiport, time_min  road times      [optional]
    vertiport_bushfire_risk.csv, vertiport_wind_risk.csv, vertiport_rain.xlsx            [optional]
    vertiport_name_with_coord.xlsx               original site list, used to place the hazard data [optional]
Missing road times are estimated from straight-line distance. Missing hazard data is borrowed
from the nearest site that has data, within MISSING_BORROW_KM.

Outputs
    shuttle_selected_sites.csv, shuttle_area_coverage.csv, shuttle_region_coverage.csv,
    shuttle_solution_map.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import MeasureControl
from pyomo.environ import (ConcreteModel, Set, Var, Binary, Objective, Constraint, ConstraintList,
                           minimize, value, SolverFactory, TerminationCondition)

# ─────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────
DEMAND_FILE    = 'suburbs_with_coord.xlsx'      # falls back to the SA2 file if this is missing
SA2_FILE       = 'sa2_names_with_coord.xlsx'
NAME_COL       = None          # area-name column; None = auto (SAL_NAME21, Suburb, SA2_Name, name)
EXISTING_FILE  = 'vertiport_shuttle_existing_candidates.xlsx'
NEW_BUILD_FILE = 'vertiport_shuttle_new_build_candidates.xlsx'
ROAD_TIME_FILE = 'd_ij.xlsx'
BUSHFIRE_FILE  = 'vertiport_bushfire_risk.csv'
WIND_FILE      = 'vertiport_wind_risk.csv'
RAIN_FILE      = 'vertiport_rain.xlsx'
ORIGINAL_SITES = 'vertiport_name_with_coord.xlsx'

ALL_AIRPORTS = {   # key: (name, lat, lon)
    'WSI': ('Western Sydney International (Nancy Bird Walton) Airport', -33.879444, 150.739722),
    'SYD': ('Sydney Kingsford Smith International Airport',             -33.946098, 151.177002),
}
DESTINATIONS = ['WSI']        # shuttle destination(s); ['WSI', 'SYD'] restores the two-airport model
AIRPORTS = {h: ALL_AIRPORTS[h] for h in DESTINATIONS}
DEST_LABEL = ' or '.join(DESTINATIONS)
AIRPORT_RULE = 'any'          # with two destinations: 'any' = at least one within the limit, 'both' = both

UAV_SPEED_KMH       = 250     # cruise speed
TAKEOFF_LANDING_MIN = 4       # added to every flight (climb, descent, approach)
RANGE_KM            = 150     # longest direct flight (typical eVTOL range with reserve)

TIME_THRESHOLD      = 60      # door-to-airport minutes: road to vertiport + flight
COVERAGE_THRESHOLD  = 0.85    # share of the study area's coverable population that must be covered
REGION_COVERAGE     = 0.50    # share of each region's coverable population (None = off)
REGION_COL          = None    # demand-file column holding the region; None = auto-detect SA4 / SA3
GREATER_SYDNEY_ONLY = True    # only areas in Greater Sydney (GCC_NAME21 column, else the box below)
GREATER_SYDNEY_BOX  = (149.97, -34.33, 151.63, -32.99)   # lon/lat box around the Greater Sydney GCCSA

# road time estimate where d_ij has no value
ROAD_DETOUR_FACTOR  = 1.3     # road distance / straight-line distance
ROAD_SPEED_KMH      = 40      # average driving speed to the vertiport
ROAD_MAX_MIN        = 15      # longest drive to a vertiport (catchment of a vertiport)
ONE_SITE_PER_CENTRE = True    # new-build sites: keep only the best (cost + hazard) site per town centre
DROP_CLOSER_TO_SYD  = True    # drop vertiports whose flight to Kingsford Smith is shorter than to WSI
EXCLUDE_NAME_WORDS  = ['medical', 'hospital', 'health', 'careflight', 'ambulance',
                       'rescue', 'police', 'military', 'raaf', 'naval']   # never used as shuttle sites
MIN_TIME_SAVING     = 10      # eVTOL trip must beat driving straight to that airport by this many minutes

# cost index: new-build sites carry their own (1.0 park, 1.5 car park, 2.0 rooftop)
COST_INDEX_EXISTING = {'helipad / HLS': 0.5, 'heliport': 0.5, 'park / oval HLS': 0.6,
                       'NSW POI helipad': 0.6, 'GA aerodrome': 0.5, 'passenger airport': 0.5}
COST_INDEX_DEFAULT  = 1.0

# hazards (each 0..1), weighted relative to the cost index
W_BUSH, W_WIND, W_RAIN = 0.5, 0.5, 0.25
BFPL_SEVERITY = {'Vegetation Category 1': 1.0, 'Vegetation Category 3': 0.6,
                 'Vegetation Category 2': 0.3, 'Vegetation Buffer': 0.15, 'Not bush fire prone': 0.0}
RAIN_MAX_AVG_MM_DAY, RAIN_MAX_HEAVY_DAYS = 5.0, 40
MISSING_BORROW_KM = 15
MISSING_DEFAULT   = {'bush': 0.0, 'wind': 0.5, 'rain': 0.5}

# Solver: Pyomo builds the model; a separate solver program solves it.
#   SOLVER = 'auto' tries CBC, then HiGHS, GLPK, Gurobi, CPLEX - whichever is installed.
#   If none is found, install one in the SAME Python environment you run this script from:
#     1. HiGHS (easiest):  pip install highspy
#     2. CBC via conda:    conda install -c conda-forge coincbc
#     3. Existing cbc.exe: set CBC_EXECUTABLE below to its full path
#   Check which environment you are in with:  import sys; print(sys.executable)
SOLVER         = 'auto'       # or 'cbc', 'appsi_highs', 'glpk', 'gurobi', 'cplex'
CBC_EXECUTABLE = None         # e.g. r'C:\path\to\cbc.exe'  (None = look on PATH)
TIME_LIMIT = 300              # seconds
MIP_GAP    = 0.005


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def hav_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def flight_min(km):
    return km / UAV_SPEED_KMH * 60 + TAKEOFF_LANDING_MIN


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
dfj = pd.read_excel(DEMAND_FILE if Path(DEMAND_FILE).exists() else SA2_FILE)
if NAME_COL is None:
    NAME_COL = next(c for c in ['SAL_NAME21', 'Suburb', 'SA2_Name', 'name'] if c in dfj.columns)
dfj = dfj.rename(columns={NAME_COL: 'area'})
if REGION_COL is None:
    REGION_COL = next((c for c in ['SA4_NAME21', 'SA4_NAME', 'SA4_Name', 'SA3_NAME21', 'SA3_NAME',
                                   'SA3_Name', 'Region', 'region'] if c in dfj.columns), None)
if 'STE_NAME21' in dfj.columns:
    dfj = dfj[dfj['STE_NAME21'] == 'New South Wales']
if GREATER_SYDNEY_ONLY:
    if 'GCC_NAME21' in dfj.columns:
        dfj = dfj[dfj['GCC_NAME21'] == 'Greater Sydney']
    else:
        x0, y0, x1, y1 = GREATER_SYDNEY_BOX
        dfj = dfj[dfj['longitude'].between(x0, x1) & dfj['latitude'].between(y0, y1)]
        print('No GCC_NAME21 column: Greater Sydney approximated by a lon/lat box')
dfj = dfj.dropna(subset=['latitude', 'longitude']).drop_duplicates('area').reset_index(drop=True)
dfj['population'] = dfj['population'].fillna(0)

AREA_LABEL = 'Greater Sydney' if GREATER_SYDNEY_ONLY else 'NSW'
demand_coords     = dict(zip(dfj['area'], zip(dfj['latitude'], dfj['longitude'])))
demand_population = dict(zip(dfj['area'], dfj['population']))
region = dict(zip(dfj['area'], dfj[REGION_COL])) if REGION_COL else {i: 'Greater Sydney' for i in demand_coords}

cand = pd.read_excel(EXISTING_FILE, sheet_name='candidates')
cand['cost_index'] = cand['site_type'].map(COST_INDEX_EXISTING).fillna(COST_INDEX_DEFAULT)
cand['new_build'] = False
if Path(NEW_BUILD_FILE).exists():
    nb = pd.read_excel(NEW_BUILD_FILE)
    cand = pd.concat([cand, nb[nb['new_build'] == True]], ignore_index=True)
cand['cost_index'] = cand['cost_index'].fillna(COST_INDEX_DEFAULT)

all_coords = dict(zip(cand['name'], zip(cand['latitude'], cand['longitude'])))   # for hazard borrowing
if Path(ORIGINAL_SITES).exists():
    src = pd.read_excel(ORIGINAL_SITES)
    all_coords = {**dict(zip(src['name'], zip(src['latitude'], src['longitude']))), **all_coords}

cand = cand[~cand['name'].isin([v[0] for v in ALL_AIRPORTS.values()])]      # airports are never vertiport sites
_bad = cand['name'].str.lower().str.contains('|'.join(EXCLUDE_NAME_WORDS), na=False)
if _bad.any():
    print(f'excluded {int(_bad.sum())} medical / emergency / defence sites: '
          + ', '.join(cand.loc[_bad, 'name'].head(5)) + (' ...' if _bad.sum() > 5 else ''))
cand = cand[~_bad]
cand = cand.dropna(subset=['latitude', 'longitude']).drop_duplicates('name').reset_index(drop=True)
for h, (_, la, lo) in AIRPORTS.items():
    cand[f'd_{h}'] = hav_km(cand['latitude'], cand['longitude'], la, lo)
    cand[f't_{h}'] = np.where(cand[f'd_{h}'] <= RANGE_KM, flight_min(cand[f'd_{h}']), np.inf)
cand = cand[np.isfinite(cand[[f't_{h}' for h in AIRPORTS]]).any(axis=1)].reset_index(drop=True)

if DROP_CLOSER_TO_SYD and 'WSI' in AIRPORTS:
    _, s_la, s_lo = ALL_AIRPORTS['SYD']
    t_syd = flight_min(hav_km(cand['latitude'], cand['longitude'], s_la, s_lo))
    closer = t_syd < cand['t_WSI']
    print(f'dropped {int(closer.sum())} vertiports that are quicker to fly to Kingsford Smith than to WSI')
    cand = cand[~closer].reset_index(drop=True)

print(f'{len(demand_coords)} demand areas ({NAME_COL}), {len(cand)} candidate vertiports within {RANGE_KM} km of {DEST_LABEL}')

# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX  a_ij  (road + direct flight <= T)
# ─────────────────────────────────────────────────────────────
road_lookup = {}
if Path(ROAD_TIME_FILE).exists():
    d_ij = pd.read_excel(ROAD_TIME_FILE)
    road_lookup = dict(zip(zip(d_ij['sa2'], d_ij['vertiport']), d_ij['time_min']))


def build_coverage_matrix(T):
    """Returns {i: [(j, door-to-airport minutes, airport), ...]} for every (i, j) within T."""
    I = list(demand_coords)
    D = hav_km(dfj['latitude'].values[:, None], dfj['longitude'].values[:, None],
               cand['latitude'].values[None, :], cand['longitude'].values[None, :])
    road = D * ROAD_DETOUR_FACTOR / ROAD_SPEED_KMH * 60
    names = cand['name'].values
    t_air = cand[[f't_{h}' for h in AIRPORTS]].values               # (J, H)
    drive = np.column_stack([hav_km(dfj['latitude'].values, dfj['longitude'].values, la, lo)
                             * ROAD_DETOUR_FACTOR / ROAD_SPEED_KMH * 60 for _, la, lo in AIRPORTS.values()])
    a = {}
    for ii, i in enumerate(I):
        opts = []
        for jj in np.where(road[ii] <= ROAD_MAX_MIN * 1.5)[0]:       # estimate first, then exact lookup
            r = road_lookup.get((i, names[jj]))
            r = road[ii, jj] if r is None or pd.isna(r) else r
            if r > ROAD_MAX_MIN:
                continue
            total = r + t_air[jj]
            ok = (total <= T) & (total <= drive[ii] - MIN_TIME_SAVING)
            if (AIRPORT_RULE == 'both' and ok.all()) or (AIRPORT_RULE == 'any' and ok.any()):
                best = int(np.argmin(np.where(ok, total, np.inf)))     # fastest airport that passes
                opts.append((names[jj], round(float(total[best]), 1), list(AIRPORTS)[best]))
        a[i] = opts
    return a


# ─────────────────────────────────────────────────────────────
# HAZARD COSTS (0..1), nearest-site fill for missing data
# ─────────────────────────────────────────────────────────────
def fill_nearest(known, kind):
    pts = [(n, v) + all_coords[n] for n, v in known.items() if n in all_coords and not pd.isna(v)]
    out = {}
    if pts:
        kl, kn = np.array([p[2] for p in pts]), np.array([p[3] for p in pts])
    for _, r in cand.iterrows():
        j = r['name']
        if j in known and not pd.isna(known[j]):
            out[j] = (known[j], 'site data')
        elif pts:
            d = hav_km(r['latitude'], r['longitude'], kl, kn)
            k = int(d.argmin())
            out[j] = ((pts[k][1], f'nearest site {d[k]:.1f} km') if d[k] <= MISSING_BORROW_KM
                      else (MISSING_DEFAULT[kind], 'default'))
        else:
            out[j] = (MISSING_DEFAULT[kind], 'default')
    return out


bush_known, wind_known, rain_known = {}, {}, {}
if Path(BUSHFIRE_FILE).exists():
    b = pd.read_csv(BUSHFIRE_FILE).dropna(subset=['name']).drop_duplicates('name')
    bush_known = {n: BFPL_SEVERITY.get(str(l).strip(), 0.0) for n, l in zip(b['name'], b['bfpl_category_label'])}
if Path(WIND_FILE).exists():
    w = pd.read_csv(WIND_FILE).dropna(subset=['pct_above_threshold'])
    wind_known = {n: min(p / 100, 1.0) for n, p in zip(w['vertiport'], w['pct_above_threshold'])}
if Path(RAIN_FILE).exists():
    r = pd.read_excel(RAIN_FILE).rename(columns={'name': 'vertiport'}).drop_duplicates('vertiport')
    rain_known = {n: (min(a / RAIN_MAX_AVG_MM_DAY, 1) + min(d / RAIN_MAX_HEAVY_DAYS, 1)) / 2
                  for n, a, d in zip(r['vertiport'], r['Avg_Rainfall'], r['Heavy_Rain_Days'])}
bush, wind, rain = fill_nearest(bush_known, 'bush'), fill_nearest(wind_known, 'wind'), fill_nearest(rain_known, 'rain')
hazard = {j: W_BUSH * bush[j][0] + W_WIND * wind[j][0] + W_RAIN * rain[j][0] for j in cand['name']}
cost = {j: ci + hazard[j] for j, ci in zip(cand['name'], cand['cost_index'])}

if ONE_SITE_PER_CENTRE and 'centre' in cand.columns:
    nbm = cand['new_build'] & cand['centre'].fillna('').astype(str).ne('')
    best = (cand[nbm].assign(_c=cand.loc[nbm, 'name'].map(cost))
            .sort_values('_c').drop_duplicates('centre')['name'])
    before = int(nbm.sum())
    cand = cand[~nbm | cand['name'].isin(best)].reset_index(drop=True)
    print(f'one site per town centre: {before} new-build sites -> {len(best)}')


# ─────────────────────────────────────────────────────────────
# PYOMO MODEL
# ─────────────────────────────────────────────────────────────
_SOLVER_CHOICE = None


def pick_solver():
    """First MIP solver that Pyomo can find on this computer."""
    global _SOLVER_CHOICE
    if _SOLVER_CHOICE:
        return _SOLVER_CHOICE
    order = [SOLVER] if SOLVER != 'auto' else ['cbc', 'appsi_highs', 'glpk', 'gurobi', 'cplex']
    for name in order:
        try:
            if make_solver(name).available(exception_flag=False):
                _SOLVER_CHOICE = name
                return name
        except Exception:
            continue
    import sys
    raise SystemExit(
        'No MIP solver found (tried: ' + ', '.join(order) + ').\n'
        f'Python environment: {sys.executable}\n'
        'Install one in this environment:\n'
        '  1. pip install highspy                        (HiGHS, easiest)\n'
        '  2. conda install -c conda-forge coincbc       (CBC)\n'
        '  3. set CBC_EXECUTABLE = r"C:\\path\\to\\cbc.exe"  (an existing CBC)')


def make_solver(name):
    if name == 'cbc' and CBC_EXECUTABLE:
        return SolverFactory('cbc', executable=CBC_EXECUTABLE)
    return SolverFactory(name)


def run_model(a, alpha, alpha_region):
    I = [i for i in demand_coords if a[i]]                               # coverable areas
    J = sorted({j for i in I for j, _, _ in a[i]})
    pop = demand_population
    cov_pop = sum(pop[i] for i in I)

    mod = ConcreteModel()
    mod.I = Set(initialize=I)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.J, domain=Binary)
    mod.z = Var(mod.I, domain=Binary)

    mod.obj = Objective(expr=sum(cost[j] * mod.y[j] for j in J), sense=minimize)
    mod.cov = Constraint(mod.I, rule=lambda m, i: m.z[i] <= sum(m.y[j] for j, _, _ in a[i]))
    mod.demand = Constraint(expr=sum(pop[i] * mod.z[i] for i in I) >= alpha * cov_pop)
    mod.reg = ConstraintList()
    if alpha_region:
        for r_ in sorted({region[i] for i in I}):
            members = [i for i in I if region[i] == r_]
            mod.reg.add(sum(pop[i] * mod.z[i] for i in members) >= alpha_region * sum(pop[i] for i in members))

    name = pick_solver()
    opt = make_solver(name)
    if name == 'cbc':
        opt.options.update({'seconds': TIME_LIMIT, 'ratioGap': MIP_GAP})
    elif name == 'appsi_highs':
        opt.config.time_limit, opt.config.mip_gap = TIME_LIMIT, MIP_GAP
    elif name == 'glpk':
        opt.options.update({'tmlim': TIME_LIMIT, 'mipgap': MIP_GAP})
    elif name in ('gurobi', 'gurobi_direct'):
        opt.options.update({'TimeLimit': TIME_LIMIT, 'MIPGap': MIP_GAP})
    elif name == 'cplex':
        opt.options.update({'timelimit': TIME_LIMIT, 'mip_tolerances_mipgap': MIP_GAP})
    res = opt.solve(mod, load_solutions=False)
    tc = res.solver.termination_condition
    if tc in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
        return 'infeasible', [], [], name
    try:
        mod.solutions.load_from(res)          # also loads a time-limited (non-optimal) solution
    except Exception:
        return 'infeasible', [], [], name
    status = 'optimal' if tc == TerminationCondition.optimal else f'feasible ({tc})'
    selected = [j for j in J if value(mod.y[j]) > 0.5]
    covered = [i for i in I if value(mod.z[i]) > 0.5]
    return status, selected, covered, name


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
a = build_coverage_matrix(TIME_THRESHOLD)
pop = demand_population
coverable = [i for i in demand_coords if a[i]]
cov_pop, all_pop = sum(pop[i] for i in coverable), sum(pop.values())
print(f'Coverable: {len(coverable)}/{len(demand_coords)} areas, {cov_pop:,.0f}/{all_pop:,.0f} people '
      f'({cov_pop / max(all_pop, 1) * 100:.1f}%) can reach {DEST_LABEL} within {TIME_THRESHOLD} min '
      f'and at least {MIN_TIME_SAVING} min faster than driving')

status, selected, covered, solver_used = run_model(a, COVERAGE_THRESHOLD, REGION_COVERAGE)
uncovered = [i for i in demand_coords if i not in covered]

print(f'\nTime threshold  = {TIME_THRESHOLD} min (road + {UAV_SPEED_KMH} km/h flight + {TAKEOFF_LANDING_MIN} min)')
print(f'Coverage target = {COVERAGE_THRESHOLD:.0%} of coverable {AREA_LABEL} population'
      + (f', {REGION_COVERAGE:.0%} per region' if REGION_COVERAGE else ''))
print(f'Solver          = {solver_used}, status {status}')

if status == 'infeasible':
    raise SystemExit('INFEASIBLE: lower COVERAGE_THRESHOLD / REGION_COVERAGE or raise TIME_THRESHOLD.')

cj = cand.set_index('name')
# which selected vertiport each covered area uses (fastest door-to-airport)
use = {}
for i in covered:
    j, t, h = min(((j, t, h) for j, t, h in a[i] if j in set(selected)), key=lambda x: x[1])
    use[i] = (j, t, h)

sites = pd.DataFrame([{
    'name': j, 'site_type': cj.at[j, 'site_type'], 'new_build': bool(cj.at[j, 'new_build']),
    'latitude': cj.at[j, 'latitude'], 'longitude': cj.at[j, 'longitude'],
    'cost_index': cj.at[j, 'cost_index'], 'hazard': round(hazard[j], 3),
    'bushfire': round(bush[j][0], 2), 'wind': round(wind[j][0], 2), 'rain': round(rain[j][0], 2),
    'hazard_data': '; '.join(sorted({bush[j][1], wind[j][1], rain[j][1]})),
    **{f'flight_{h}_min': round(cj.at[j, f't_{h}'], 1) for h in AIRPORTS},
    'areas_served': sum(1 for i in use if use[i][0] == j),
    'population_served': sum(pop[i] for i in use if use[i][0] == j),
} for j in selected]).sort_values('population_served', ascending=False)

_drive = {i: {h: float(hav_km(*demand_coords[i], la, lo) * ROAD_DETOUR_FACTOR / ROAD_SPEED_KMH * 60)
              for h, (_, la, lo) in AIRPORTS.items()} for i in demand_coords}
sa2 = pd.DataFrame([{'area': i, 'region': region[i], 'population': pop[i], 'coverable': bool(a[i]),
                     'covered': i in use, 'vertiport': use.get(i, ('', None, ''))[0],
                     'airport': use.get(i, ('', None, ''))[2],
                     'door_to_airport_min': use.get(i, ('', None, ''))[1],
                     'drive_to_airport_min': round(_drive[i][use[i][2]], 1) if i in use else None}
                    for i in demand_coords])
reg = (sa2.assign(coverable_pop=sa2['population'] * sa2['coverable'],
                  covered_pop=sa2['population'] * sa2['covered'])
       .groupby('region').agg(areas=('area', 'size'), population=('population', 'sum'),
                              coverable_pop=('coverable_pop', 'sum'), covered_pop=('covered_pop', 'sum'))
       .reset_index())
reg['covered_of_coverable'] = (reg['covered_pop'] / reg['coverable_pop'].replace(0, np.nan)).round(3)
reg['covered_of_total'] = (reg['covered_pop'] / reg['population'].replace(0, np.nan)).round(3)

sites.to_csv('shuttle_selected_sites.csv', index=False)
sa2.to_csv('shuttle_area_coverage.csv', index=False)
reg.to_csv('shuttle_region_coverage.csv', index=False)

cov = sum(pop[i] for i in covered)
print(f'\nMinimum vertiports : {len(selected)}  '
      f'({sites["new_build"].sum()} new-build, {(~sites["new_build"]).sum()} existing)')
print(f'Covered            : {len(covered)}/{len(demand_coords)} areas, {cov:,.0f} people '
      f'({cov / max(cov_pop, 1) * 100:.1f}% of coverable, {cov / max(all_pop, 1) * 100:.1f}% of {AREA_LABEL})')
print('\nSelected sites:')
print(sites[['name', 'site_type', 'cost_index', 'hazard', *[f'flight_{h}_min' for h in AIRPORTS],
             'areas_served', 'population_served']].to_string(index=False))
print(f'\nCoverage by region ({REGION_COL or "no region column: whole of Greater Sydney"}):')
print(reg.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS  (set RUN_SENSITIVITY = True)
# ─────────────────────────────────────────────────────────────
RUN_SENSITIVITY = False
if RUN_SENSITIVITY:
    import matplotlib.pyplot as plt
    time_thresholds  = [30, 45, 60, 75, 90]
    coverage_targets = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = {}
    for T in time_thresholds:
        a_T = build_coverage_matrix(T)
        for alpha in coverage_targets:
            s_, sel_, _, _ = run_model(a_T, alpha, REGION_COVERAGE)
            results[T, alpha] = len(sel_) if s_ != 'infeasible' else np.nan
            print(f'T={T:3d} min | coverage={alpha:.0%} | vertiports={results[T, alpha]}')
    fig, ax = plt.subplots(figsize=(9, 6))
    for alpha in coverage_targets:
        ax.plot(time_thresholds, [results[T, alpha] for T in time_thresholds], marker='o', label=f'{alpha:.0%}')
    ax.set_xlabel('Door-to-airport time limit (minutes)')
    ax.set_ylabel('Vertiports required')
    ax.legend(title='Coverage target')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('shuttle_sensitivity.png', dpi=150)

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
m_map = folium.Map(location=[-33.87, 150.95], zoom_start=10,
                   tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
                   attr='Esri, HERE, Garmin, FAO, NOAA, USGS')
MeasureControl(position='bottomleft', primaryLengthUnit='kilometers', secondaryLengthUnit='meters',
               activeColor='orange', completedColor='green').add_to(m_map)
m_map.get_root().html.add_child(folium.Element('<style>.leaflet-interactive:focus { outline: none !important; }</style>'))

FLIGHT_COL = {'SYD': '#d62728', 'WSI': '#1f77b4'}
AIRPORT_LABEL = {'SYD': 'Kingsford Smith', 'WSI': 'Western Sydney'}
for _, s in sites.iterrows():                  # flight line from each vertiport to the airport it mostly serves
    hs = [use[i][2] for i in use if use[i][0] == s['name']]
    h = max(set(hs), key=hs.count) if hs else min(AIRPORTS, key=lambda k: cj.at[s['name'], f't_{k}'])
    folium.PolyLine([[s['latitude'], s['longitude']], [AIRPORTS[h][1], AIRPORTS[h][2]]],
                    color=FLIGHT_COL[h], weight=2, opacity=0.6).add_to(m_map)
for i in demand_coords:
    lat, lon = demand_coords[i]
    folium.CircleMarker([lat, lon], radius=4, color='red' if i in use else 'orange', fill=True, fill_opacity=1,
                        popup=folium.Popup(i + (f'<br>via {use[i][0]} to {use[i][2]}, {use[i][1]} min' if i in use else ''),
                                           max_width=250), tooltip=i).add_to(m_map)
for _, s in sites.iterrows():
    folium.CircleMarker([s['latitude'], s['longitude']], radius=6, color='blue', fill=True, fill_opacity=1,
                        popup=folium.Popup(f"<b>{s['name']}</b><br>{s['site_type']}<br>"
                                           f"{s['areas_served']} areas, {s['population_served']:,.0f} people", max_width=250),
                        tooltip=s['name']).add_to(m_map)
cross = ('<div style="color:red; font-size:30px; font-weight:bold; line-height:30px; text-align:center; '
         'text-shadow:0 0 3px white, 0 0 3px white;">&#10006;</div>')
for h, (n, la, lo) in AIRPORTS.items():
    folium.Marker([la, lo], icon=folium.DivIcon(html=cross, icon_size=(30, 30), icon_anchor=(15, 15)),
                  tooltip=n).add_to(m_map)

dot = lambda col: f'<i style="background:{col}; width:12px; height:12px; display:inline-block; border-radius:50%;"></i>'
line = lambda col: f'<i style="background:{col}; width:16px; height:3px; display:inline-block;"></i>'
legend_html = f'''
<div style="position: fixed; bottom: 50px; right: 50px; width: 260px; background-color: white;
     border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
<p style="margin: 5px 0;"><span style="color:red; font-weight:bold;">&#10006;</span> Airport ({', '.join(AIRPORTS)})</p>
<p style="margin: 5px 0;">{dot('blue')} Selected Vertiports ({len(selected)})</p>
<p style="margin: 5px 0;">{dot('red')} Covered areas ({len(covered)})</p>
<p style="margin: 5px 0;">{dot('orange')} Uncovered areas ({len(uncovered)})</p>
{''.join(f'<p style="margin: 5px 0;">{line(FLIGHT_COL[h])} Flight to {AIRPORT_LABEL[h]}</p>' for h in AIRPORTS)}
</div>'''
m_map.get_root().html.add_child(folium.Element(legend_html))
m_map.save('shuttle_solution_map.html')
print('\nSaved shuttle_selected_sites.csv, shuttle_area_coverage.csv, shuttle_region_coverage.csv, '
      'shuttle_solution_map.html')