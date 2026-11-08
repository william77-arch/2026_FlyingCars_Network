"""
eVTOL AIRPORT-SHUTTLE vertiport location model, V2 (all of NSW) - Western Sydney International (WSI)

Journey:  area i --(drive)--> vertiport j --(direct eVTOL flight)--> WSI

Sets
    I  demand areas in NSW: suburbs (prepare_suburb_demand.py) or SA2 regions
    J  candidate vertiports (existing sites + new-build sites)

Coverage
    a_ij = 1 if  distance(j, WSI) <= RANGE_KM            (the eVTOL can make the flight), and
                 drive_time(i, j) <= ROAD_MAX_MIN            (area i is in vertiport j's catchment)
           Optional extra rules, off by default (set a number to switch on):
                 drive_time(i, j) + flight_time(j, WSI) <= TIME_THRESHOLD
                 drive_time(i, j) + flight_time(j, WSI) <= drive_time(i, WSI) - MIN_TIME_SAVING
    Drive times come from OSRM (OpenStreetMap road network, Table service), cached in ROAD_CACHE.

Model
    min   sum_j ci_j * y_j                                  ci_j = cost index of site j
    s.t.  z_i <= sum_j a_ij * y_j                           area i covered only if a chosen vertiport serves it
          sum_i pop_i z_i        >= COVERAGE_THRESHOLD * coverable population
          sum_{i in r} pop_i z_i >= REGION_COVERAGE   * coverable population of region r  (each r)
          y_j, z_i binary

Inputs (same folder)
    suburbs_with_coord.xlsx (or sa2_names_with_coord.xlsx)
                                                 name, latitude, longitude, population [+ region, GCC_NAME21]
    vertiport_shuttle_existing_candidates.xlsx  sheet 'candidates' (build_shuttle_vertiports_existing_sites.py)
    vertiport_shuttle_new_build_candidates.xlsx (make_new_build_candidates.py)            [optional]
    Internet access to an OSRM server (the free public demo server by default; no key needed).
    If OSRM cannot be reached, drive times fall back to a straight-line estimate.

Outputs
    shuttle_selected_sites_V2.csv, shuttle_area_coverage_V2.csv, shuttle_region_coverage_V2.csv,
    shuttle_solution_map_V2.html, osrm_road_times_V2.csv (cache of every drive time fetched)
"""

import time
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

WSI = ('Western Sydney International (Nancy Bird Walton) Airport', -33.879444, 150.739722)
NOT_SITES = [WSI[0], 'Sydney Kingsford Smith International Airport']   # airports are never vertiport sites

UAV_SPEED_KMH       = 250     # cruise speed
TAKEOFF_LANDING_MIN = 4       # added to every flight (climb, descent, approach)
RANGE_KM            = 150     # longest direct flight (typical eVTOL range with reserve)

# Coverage is limited only by the eVTOL's range (RANGE_KM) and each vertiport's catchment (ROAD_MAX_MIN).
TIME_THRESHOLD      = None    # door-to-WSI time limit in minutes (None = off; e.g. 60 to switch on)
COVERAGE_THRESHOLD  = 1    # share of the study area's coverable population that must be covered
REGION_COVERAGE     = 0.80    # share of each region's coverable population (None = off)
REGION_COL          = None    # demand-file column holding the region; None = auto-detect SA4 / SA3
STUDY_AREA          = 'NSW'   # 'NSW' (all areas in the demand file) or 'Greater Sydney'
GREATER_SYDNEY_BOX  = (149.97, -34.33, 151.63, -32.99)   # used for 'Greater Sydney' if there is no GCC_NAME21 column

ROAD_MAX_MIN        = 30      # longest drive to a vertiport (catchment of a vertiport)
MIN_TIME_SAVING     = None    # minutes the eVTOL trip must beat driving straight to WSI (None = off; e.g. 10)
ONE_SITE_PER_CENTRE = True    # new-build sites: keep only the cheapest site per town centre
EXCLUDE_NAME_WORDS  = ['medical', 'hospital', 'health', 'careflight', 'ambulance',
                       'rescue', 'police', 'military', 'raaf', 'naval']   # never used as shuttle sites

# cost index (relative build cost): new-build sites carry their own (1.0 park, 1.5 car park, 2.0 rooftop)
COST_INDEX_EXISTING = {'helipad / HLS': 0.5, 'heliport': 0.5, 'park / oval HLS': 0.6,
                       'NSW POI helipad': 0.6, 'GA aerodrome': 0.5, 'passenger airport': 0.5}
COST_INDEX_DEFAULT  = 1.0

# --- OSRM drive times (OpenStreetMap road network) ---
# Default: the free public OSRM demo server (car profile, no traffic, no key). It asks for light
# use only, so requests are spaced OSRM_DELAY_S apart. For heavy use, run your own OSRM server
# (e.g. with Docker) and point OSRM_URL at it, then set OSRM_DELAY_S = 0.
OSRM_URL           = 'https://router.project-osrm.org'
OSRM_PROFILE       = 'driving'
OSRM_DELAY_S       = 0.3        # seconds between requests (same pacing as the original optim_model.py)
OSRM_BLOCK         = 50         # areas and sites per request (50 + 50 = 100 points, the public server's limit)
CELL_KM            = 10         # nearby areas are batched together in cells of this size
USER_AGENT         = 'vertiport-shuttle-model-V2 (university research project)'
ROAD_CACHE         = 'osrm_road_times_V2.csv'   # every drive time fetched is saved here and reused
PREFILTER_KM       = 35        # only route to vertiports within this straight-line distance of an area
                               # (keep it at least ROAD_MAX_MIN x ~1 km/min, or it caps the drive limit)
MAX_SITES_PER_AREA = 60        # ... and at most this many of the nearest ones
# fallback estimate when OSRM can't be reached or finds no route
EST_DETOUR, EST_SPEED_KMH = 1.3, 40

# --- solver ---
# Pyomo builds the model; a separate solver program solves it.
#   SOLVER = 'auto' tries CBC, then HiGHS, GLPK, Gurobi, CPLEX - whichever is installed.
#   If none is found, install one in the SAME Python environment you run this script from:
#     1. HiGHS (easiest):  pip install highspy
#     2. CBC via conda:    conda install -c conda-forge coincbc
#     3. Existing cbc.exe: set CBC_EXECUTABLE below to its full path
SOLVER         = 'auto'       # or 'cbc', 'appsi_highs', 'glpk', 'gurobi', 'cplex'
CBC_EXECUTABLE = None         # e.g. r'C:\path\to\cbc.exe'  (None = look on PATH)
TIME_LIMIT = 300              # seconds; after this the best solution found so far is used
MIP_GAP    = 0.005            # stop when within 0.5% of the best possible cost
SHOW_SOLVER_LOG = True        # print the solver's progress while it works


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def hav_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def flight_min(km):
    return km / UAV_SPEED_KMH * 60 + TAKEOFF_LANDING_MIN


def est_drive_min(km):
    return km * EST_DETOUR / EST_SPEED_KMH * 60


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
if STUDY_AREA == 'Greater Sydney':
    if 'GCC_NAME21' in dfj.columns:
        dfj = dfj[dfj['GCC_NAME21'] == 'Greater Sydney']
    else:
        x0, y0, x1, y1 = GREATER_SYDNEY_BOX
        dfj = dfj[dfj['longitude'].between(x0, x1) & dfj['latitude'].between(y0, y1)]
        print('No GCC_NAME21 column: Greater Sydney approximated by a lon/lat box')
dfj = dfj.dropna(subset=['latitude', 'longitude']).drop_duplicates('area').reset_index(drop=True)
dfj['population'] = dfj['population'].fillna(0)

AREA_LABEL = STUDY_AREA
demand_coords     = dict(zip(dfj['area'], zip(dfj['latitude'], dfj['longitude'])))
demand_population = dict(zip(dfj['area'], dfj['population']))
region = dict(zip(dfj['area'], dfj[REGION_COL])) if REGION_COL else {i: STUDY_AREA for i in demand_coords}

cand = pd.read_excel(EXISTING_FILE, sheet_name='candidates')
cand['cost_index'] = cand['site_type'].map(COST_INDEX_EXISTING).fillna(COST_INDEX_DEFAULT)
cand['new_build'] = False
if Path(NEW_BUILD_FILE).exists():
    nb = pd.read_excel(NEW_BUILD_FILE)
    cand = pd.concat([cand, nb[nb['new_build'] == True]], ignore_index=True)
cand['cost_index'] = cand['cost_index'].fillna(COST_INDEX_DEFAULT)

cand = cand[~cand['name'].isin(NOT_SITES)]
_bad = cand['name'].str.lower().str.contains('|'.join(EXCLUDE_NAME_WORDS), na=False)
if _bad.any():
    print(f'excluded {int(_bad.sum())} medical / emergency / defence sites: '
          + ', '.join(cand.loc[_bad, 'name'].head(5)) + (' ...' if _bad.sum() > 5 else ''))
cand = cand[~_bad]
cand = cand.dropna(subset=['latitude', 'longitude']).drop_duplicates('name').reset_index(drop=True)
cand['d_WSI'] = hav_km(cand['latitude'], cand['longitude'], WSI[1], WSI[2])
cand = cand[cand['d_WSI'] <= RANGE_KM].reset_index(drop=True)
cand['t_WSI'] = flight_min(cand['d_WSI'])

# one new-build site per town centre: the cheapest (ties -> closest to the station)
if ONE_SITE_PER_CENTRE and 'centre' in cand.columns:
    nbm = cand['new_build'] & cand['centre'].fillna('').astype(str).ne('')
    sort_cols = ['cost_index'] + (['dist_station_m'] if 'dist_station_m' in cand.columns else [])
    best = cand[nbm].sort_values(sort_cols).drop_duplicates('centre')['name']
    before = int(nbm.sum())
    cand = cand[~nbm | cand['name'].isin(best)].reset_index(drop=True)
    print(f'one site per town centre: {before} new-build sites -> {len(best)}')

cost = dict(zip(cand['name'], cand['cost_index']))
print(f'{len(demand_coords)} demand areas ({NAME_COL}), {len(cand)} candidate vertiports within {RANGE_KM} km of WSI')


# ─────────────────────────────────────────────────────────────
# DRIVE TIMES: OSRM Table service, cached
# ─────────────────────────────────────────────────────────────
def load_cache():
    if Path(ROAD_CACHE).exists():
        c = pd.read_csv(ROAD_CACHE)
        return {(a, d): (m, k) for a, d, m, k in zip(c['area'], c['destination'], c['drive_min'], c['road_km'])}
    return {}


def save_cache(cache):
    pd.DataFrame([{'area': a, 'destination': d, 'drive_min': m, 'road_km': k} for (a, d), (m, k) in cache.items()]
                 ).to_csv(ROAD_CACHE, index=False)


def osrm_table(origins, dests):
    """origins, dests: [(lat, lon), ...] -> matrix [origin][dest] of (minutes, km) or None (no route).
    One request computes every origin x destination pair, as in the original optim_model.py."""
    import requests
    pts = list(origins) + list(dests)
    coords = ';'.join(f'{lo:.6f},{la:.6f}' for la, lo in pts)          # OSRM wants lon,lat
    url = f'{OSRM_URL}/table/v1/{OSRM_PROFILE}/{coords}'
    n = len(origins)
    params = {'sources': ';'.join(map(str, range(n))),
              'destinations': ';'.join(map(str, range(n, len(pts)))),
              'annotations': 'duration,distance'}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers={'User-Agent': USER_AGENT}, timeout=120)
        except requests.RequestException as e:
            if attempt == 4:
                raise SystemExit(f'OSRM not reachable at {OSRM_URL}: {e}')
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):          # busy: wait and retry
            time.sleep(10 * (attempt + 1))
            continue
        js = r.json()
        if js.get('code') != 'Ok':
            raise SystemExit(f'OSRM error {r.status_code}: {js.get("code")} {js.get("message", "")}')
        break
    else:
        raise SystemExit(f'OSRM at {OSRM_URL} kept refusing requests; try again later or raise OSRM_DELAY_S')
    return [[(d / 60, k / 1000) if d is not None else None for d, k in zip(drow, krow)]
            for drow, krow in zip(js['durations'], js['distances'])]


def get_drive_times():
    """Returns {(area, destination): minutes} for each area -> nearby vertiports and area -> WSI."""
    lat_a, lon_a = dfj['latitude'].values, dfj['longitude'].values
    D = hav_km(lat_a[:, None], lon_a[:, None], cand['latitude'].values[None, :], cand['longitude'].values[None, :])
    names = cand['name'].values
    wanted = {}                                        # area -> [(dest name, lat, lon), ...]
    for ii, i in enumerate(dfj['area']):
        near = np.where(D[ii] <= PREFILTER_KM)[0]
        near = near[np.argsort(D[ii, near])][:MAX_SITES_PER_AREA]
        wanted[i] = [(names[jj], cand.at[jj, 'latitude'], cand.at[jj, 'longitude']) for jj in near]
        if wanted[i]:                                  # no vertiport nearby = can't be covered: don't route it
            wanted[i].append((WSI[0], WSI[1], WSI[2]))

    cache = load_cache()
    todo = {i: [d for d in ds if (i, d[0]) not in cache] for i, ds in wanted.items()}
    todo = {i: ds for i, ds in todo.items() if ds}
    n_new = sum(len(ds) for ds in todo.values())
    n_all = sum(len(ds) for ds in wanted.values())
    print(f'drive times needed: {n_all:,} ({n_all - n_new:,} already in {ROAD_CACHE}, {n_new:,} new)')

    if n_new:
        # batch nearby areas: group them into CELL_KM cells, then ask for every area x site pair
        # in a cell in blocks of OSRM_BLOCK x OSRM_BLOCK (one request returns up to 2,500 drive times)
        dpos = {}
        for i, ds in todo.items():
            for dname, la, lo in ds:
                dpos[dname] = (la, lo)
        cells = {}
        lat0 = np.mean([demand_coords[i][0] for i in todo])
        for i in todo:
            la, lo = demand_coords[i]
            key = (int(la * 111 // CELL_KM), int(lo * 111 * np.cos(np.radians(lat0)) // CELL_KM))
            cells.setdefault(key, []).append(i)
        jobs = []
        for areas in cells.values():
            dests = sorted({d[0] for i in areas for d in todo[i]})
            for a0 in range(0, len(areas), OSRM_BLOCK):
                for d0 in range(0, len(dests), OSRM_BLOCK):
                    jobs.append((areas[a0:a0 + OSRM_BLOCK], dests[d0:d0 + OSRM_BLOCK]))
        print(f'fetching {n_new:,} drive times from OSRM in {len(jobs):,} requests '
              f'(about {len(jobs) * (OSRM_DELAY_S + 0.7) / 60:.0f} min)...')
        try:
            for k, (areas, dests) in enumerate(jobs):
                mat = osrm_table([demand_coords[i] for i in areas], [dpos[d] for d in dests])
                for i, row in zip(areas, mat):
                    for dname, v in zip(dests, row):
                        cache[i, dname] = v if v else (np.nan, np.nan)      # NaN = OSRM found no route
                time.sleep(OSRM_DELAY_S)
                if (k + 1) % 25 == 0:
                    save_cache(cache)                                       # keep progress if interrupted
                    print(f'  {k + 1}/{len(jobs)} requests')
        except SystemExit as e:
            save_cache(cache)
            print(f'{e}\nUsing the straight-line estimate for the drive times still missing.')
        save_cache(cache)

    times, n_est = {}, 0
    for i, ds in wanted.items():
        for dname, la, lo in ds:
            m = cache.get((i, dname), (np.nan, np.nan))[0]
            if pd.isna(m):
                m = est_drive_min(hav_km(*demand_coords[i], la, lo))
                n_est += 1
            times[i, dname] = m
    if n_est:
        print(f'{n_est:,} drive times estimated (OSRM unreachable or no route found)')
    skipped = 0
    for i in demand_coords:                            # areas with no vertiport nearby: estimate the WSI drive
        if (i, WSI[0]) not in times:
            times[i, WSI[0]] = est_drive_min(hav_km(*demand_coords[i], WSI[1], WSI[2]))
            skipped += 1
    if skipped:
        print(f'{skipped:,} areas have no vertiport within {PREFILTER_KM} km (not routed, cannot be covered)')
    return times


# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX  a_ij
# ─────────────────────────────────────────────────────────────
drive = get_drive_times()
t_wsi = dict(zip(cand['name'], cand['t_WSI']))
sites_near = {}                                   # area -> [(vertiport, drive minutes), ...]
for (i, j), r in drive.items():
    if j != WSI[0]:
        sites_near.setdefault(i, []).append((j, r))


d_wsi = dict(zip(cand['name'], cand['d_WSI']))


def build_coverage_matrix(T=TIME_THRESHOLD, rng=RANGE_KM):
    """Returns {i: [(j, door-to-WSI minutes, drive minutes), ...]} for every vertiport that covers i."""
    a = {}
    for i in demand_coords:
        direct = drive[i, WSI[0]]
        opts = []
        for j, r in sites_near.get(i, []):
            total = r + t_wsi[j]
            if (d_wsi[j] <= rng and r <= ROAD_MAX_MIN
                    and (T is None or total <= T)
                    and (MIN_TIME_SAVING is None or total <= direct - MIN_TIME_SAVING)):
                opts.append((j, round(float(total), 1), round(float(r), 1)))
        a[i] = opts
    return a


# ─────────────────────────────────────────────────────────────
# PYOMO MODEL
# ─────────────────────────────────────────────────────────────
_SOLVER_CHOICE = None


def make_solver(name):
    if name == 'cbc' and CBC_EXECUTABLE:
        return SolverFactory('cbc', executable=CBC_EXECUTABLE)
    return SolverFactory(name)


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


def reduce_problem(a):
    """Shrinks the model without changing its optimal answer:
    1. drops a site if another site covers every area it covers at the same or lower cost (dominated);
    2. merges areas in the same region that are covered by exactly the same sites (populations added).
    Returns (groups, opts, J): groups = {group id: [areas]}, opts = {group id: [sites]}, J = sites kept."""
    I = [i for i in demand_coords if a[i]]
    S = {}                                                    # site -> set of areas it covers
    for i in I:
        for j, _, _ in a[i]:
            S.setdefault(j, set()).add(i)
    by_area = {}
    for j, cov in S.items():
        for i in cov:
            by_area.setdefault(i, []).append(j)
    order = sorted(S, key=lambda j: (cost[j], -len(S[j]), j))  # cheaper and bigger first
    rank = {j: n for n, j in enumerate(order)}
    keep = set()
    for j in order:
        rare = min(S[j], key=lambda i: len(by_area[i]))        # rivals must also cover this area
        dominated = any(k in keep and cost[k] <= cost[j] and S[j] <= S[k]
                        for k in by_area[rare] if k != j and rank[k] < rank[j])
        if not dominated:
            keep.add(j)
    groups, opts = {}, {}
    for i in I:
        sig = (region[i], frozenset(j for j in by_area[i] if j in keep))
        groups.setdefault(sig, []).append(i)
    gid = {sig: n for n, sig in enumerate(groups)}
    opts = {gid[sig]: sorted(sig[1]) for sig in groups}
    groups = {gid[sig]: members for sig, members in groups.items()}
    print(f'model size: {len(S):,} sites -> {len(keep):,} after dropping dominated sites; '
          f'{len(I):,} areas -> {len(groups):,} after merging areas with the same options')
    return groups, opts, sorted(keep)


def run_model(a, alpha, alpha_region):
    groups, opts, J = reduce_problem(a)
    G = list(groups)
    pop = demand_population
    gpop = {g: sum(pop[i] for i in groups[g]) for g in G}
    greg = {g: region[groups[g][0]] for g in G}
    cov_pop = sum(gpop.values())

    mod = ConcreteModel()
    mod.G = Set(initialize=G)
    mod.J = Set(initialize=J)
    mod.y = Var(mod.J, domain=Binary)
    # z does not need to be binary: z <= (whole number of chosen sites) and z <= 1 already force it to
    # 0 or 1 at the best answer, and a continuous z makes the solver much faster (same optimal sites)
    mod.z = Var(mod.G, bounds=(0, 1))

    mod.obj = Objective(expr=sum(cost[j] * mod.y[j] for j in J), sense=minimize)
    mod.cov = Constraint(mod.G, rule=lambda m, g: m.z[g] <= sum(m.y[j] for j in opts[g]))
    mod.demand = Constraint(expr=sum(gpop[g] * mod.z[g] for g in G) >= alpha * cov_pop)
    mod.reg = ConstraintList()
    if alpha_region:
        members = {}
        for g in G:
            members.setdefault(greg[g], []).append(g)
        for r_, gs in sorted(members.items()):
            mod.reg.add(sum(gpop[g] * mod.z[g] for g in gs) >= alpha_region * sum(gpop[g] for g in gs))

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
    print(f'solving with {name} (stops after {TIME_LIMIT}s or within {MIP_GAP:.1%} of optimal)...')
    res = opt.solve(mod, load_solutions=False, tee=SHOW_SOLVER_LOG)
    tc = res.solver.termination_condition
    if tc in (TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded):
        return 'infeasible', [], [], name
    try:
        mod.solutions.load_from(res)          # also loads a time-limited (non-optimal) solution
    except Exception:
        return 'infeasible', [], [], name
    status = 'optimal' if tc == TerminationCondition.optimal else f'feasible ({tc})'
    selected = [j for j in J if value(mod.y[j]) > 0.5]
    chosen = set(selected)
    covered = [i for g in G if any(j in chosen for j in opts[g]) for i in groups[g]]   # served by a chosen site
    return status, selected, covered, name


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
a = build_coverage_matrix()
pop = demand_population
coverable = [i for i in demand_coords if a[i]]
cov_pop, all_pop = sum(pop[i] for i in coverable), sum(pop.values())
rules = [f'a vertiport within {ROAD_MAX_MIN} min drive', f'that vertiport within {RANGE_KM} km of WSI']
if TIME_THRESHOLD:
    rules.append(f'door-to-WSI within {TIME_THRESHOLD} min')
if MIN_TIME_SAVING:
    rules.append(f'at least {MIN_TIME_SAVING} min faster than driving')
print(f'Coverable: {len(coverable)}/{len(demand_coords)} areas, {cov_pop:,.0f}/{all_pop:,.0f} people '
      f'({cov_pop / max(all_pop, 1) * 100:.1f}%) have ' + ', '.join(rules))

status, selected, covered, solver_used = run_model(a, COVERAGE_THRESHOLD, REGION_COVERAGE)
uncovered = [i for i in demand_coords if i not in covered]

print(f'\neVTOL range     = {RANGE_KM} km | catchment = {ROAD_MAX_MIN} min drive | '
      f'time limit = {TIME_THRESHOLD or "off"} | time saving rule = {MIN_TIME_SAVING or "off"}')
print(f'Coverage target = {COVERAGE_THRESHOLD:.0%} of coverable {AREA_LABEL} population'
      + (f', {REGION_COVERAGE:.0%} per region' if REGION_COVERAGE else ''))
print(f'Solver          = {solver_used}, status {status}')

if status == 'infeasible':
    raise SystemExit('INFEASIBLE: lower COVERAGE_THRESHOLD / REGION_COVERAGE, or relax the coverage rules.')

cj = cand.set_index('name')
sel_set = set(selected)
use = {}                                     # covered area -> (vertiport, door-to-WSI min, drive min)
for i in covered:
    use[i] = min((o for o in a[i] if o[0] in sel_set), key=lambda o: o[1])

sites = pd.DataFrame([{
    'name': j, 'site_type': cj.at[j, 'site_type'], 'new_build': bool(cj.at[j, 'new_build']),
    'latitude': cj.at[j, 'latitude'], 'longitude': cj.at[j, 'longitude'],
    'cost_index': cj.at[j, 'cost_index'], 'flight_WSI_min': round(cj.at[j, 't_WSI'], 1),
    'areas_served': sum(1 for i in use if use[i][0] == j),
    'population_served': sum(pop[i] for i in use if use[i][0] == j),
} for j in selected]).sort_values('population_served', ascending=False)

area_tab = pd.DataFrame([{
    'area': i, 'region': region[i], 'population': pop[i], 'coverable': bool(a[i]), 'covered': i in use,
    'vertiport': use[i][0] if i in use else '',
    'drive_to_vertiport_min': use[i][2] if i in use else None,
    'door_to_WSI_min': use[i][1] if i in use else None,
    'drive_to_WSI_min': round(drive[i, WSI[0]], 1),
} for i in demand_coords])
reg = (area_tab.assign(coverable_pop=area_tab['population'] * area_tab['coverable'],
                       covered_pop=area_tab['population'] * area_tab['covered'])
       .groupby('region').agg(areas=('area', 'size'), population=('population', 'sum'),
                              coverable_pop=('coverable_pop', 'sum'), covered_pop=('covered_pop', 'sum'))
       .reset_index())
reg['covered_of_coverable'] = (reg['covered_pop'] / reg['coverable_pop'].replace(0, np.nan)).round(3)
reg['covered_of_total'] = (reg['covered_pop'] / reg['population'].replace(0, np.nan)).round(3)

sites.to_csv('shuttle_selected_sites_V2.csv', index=False)
area_tab.to_csv('shuttle_area_coverage_V2.csv', index=False)
reg.to_csv('shuttle_region_coverage_V2.csv', index=False)

cov = sum(pop[i] for i in covered)
print(f'\nMinimum vertiports : {len(selected)}  '
      f'({sites["new_build"].sum()} new-build, {(~sites["new_build"]).sum()} existing)')
print(f'Covered            : {len(covered)}/{len(demand_coords)} areas, {cov:,.0f} people '
      f'({cov / max(cov_pop, 1) * 100:.1f}% of coverable, {cov / max(all_pop, 1) * 100:.1f}% of {AREA_LABEL})')
print('\nSelected sites:')
print(sites[['name', 'site_type', 'cost_index', 'flight_WSI_min', 'areas_served', 'population_served']]
      .to_string(index=False))
print(f'\nCoverage by region ({REGION_COL or "no region column: one region"}):')
print(reg.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS  (set RUN_SENSITIVITY = True; uses the cached drive times)
# ─────────────────────────────────────────────────────────────
RUN_SENSITIVITY = False
if RUN_SENSITIVITY:
    import matplotlib.pyplot as plt
    ranges           = [50, 75, 100, 125, 150]          # eVTOL range (km); must not exceed RANGE_KM
    coverage_targets = [0.70, 0.80, 0.85, 0.90, 0.95]
    results = {}
    for rk in ranges:
        a_R = build_coverage_matrix(rng=rk)
        for alpha in coverage_targets:
            s_, sel_, _, _ = run_model(a_R, alpha, REGION_COVERAGE)
            results[rk, alpha] = len(sel_) if s_ != 'infeasible' else np.nan
            print(f'range={rk:3d} km | coverage={alpha:.0%} | vertiports={results[rk, alpha]}')
    fig, ax = plt.subplots(figsize=(9, 6))
    for alpha in coverage_targets:
        ax.plot(ranges, [results[rk, alpha] for rk in ranges], marker='o', label=f'{alpha:.0%}')
    ax.set_xlabel('eVTOL range (km)')
    ax.set_ylabel('Vertiports required')
    ax.legend(title='Coverage target')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('shuttle_sensitivity_V2.png', dpi=150)

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
m_map = folium.Map(location=[WSI[1], WSI[2]], zoom_start=8,
                   tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
                   attr='Esri, HERE, Garmin, FAO, NOAA, USGS')
MeasureControl(position='bottomleft', primaryLengthUnit='kilometers', secondaryLengthUnit='meters',
               activeColor='orange', completedColor='green').add_to(m_map)
m_map.get_root().html.add_child(folium.Element('<style>.leaflet-interactive:focus { outline: none !important; }</style>'))

FLIGHT_COL = '#1f77b4'
folium.Circle([WSI[1], WSI[2]], radius=RANGE_KM * 1000, color='#1f77b4', weight=2, dash_array='8 6',
              fill=False, tooltip=f'eVTOL range: {RANGE_KM} km from WSI').add_to(m_map)
for _, s in sites.iterrows():                  # flight line from each vertiport to WSI
    folium.PolyLine([[s['latitude'], s['longitude']], [WSI[1], WSI[2]]],
                    color=FLIGHT_COL, weight=2, opacity=0.6).add_to(m_map)
for i in demand_coords:
    lat, lon = demand_coords[i]
    txt = (f'{i}<br>via {use[i][0]}: {use[i][2]} min drive, {use[i][1]} min to WSI '
           f'(driving {drive[i, WSI[0]]:.0f} min)') if i in use else f'{i}<br>driving to WSI {drive[i, WSI[0]]:.0f} min'
    folium.CircleMarker([lat, lon], radius=4, color='red' if i in use else 'orange', fill=True, fill_opacity=1,
                        popup=folium.Popup(txt, max_width=280), tooltip=i).add_to(m_map)
for _, s in sites.iterrows():
    folium.CircleMarker([s['latitude'], s['longitude']], radius=6, color='blue', fill=True, fill_opacity=1,
                        popup=folium.Popup(f"<b>{s['name']}</b><br>{s['site_type']}<br>"
                                           f"{s['areas_served']} areas, {s['population_served']:,.0f} people", max_width=250),
                        tooltip=s['name']).add_to(m_map)
cross = ('<div style="color:red; font-size:30px; font-weight:bold; line-height:30px; text-align:center; '
         'text-shadow:0 0 3px white, 0 0 3px white;">&#10006;</div>')
folium.Marker([WSI[1], WSI[2]], icon=folium.DivIcon(html=cross, icon_size=(30, 30), icon_anchor=(15, 15)),
              tooltip=WSI[0]).add_to(m_map)

dot = lambda col: f'<i style="background:{col}; width:12px; height:12px; display:inline-block; border-radius:50%;"></i>'
line = lambda col: f'<i style="background:{col}; width:16px; height:3px; display:inline-block;"></i>'
legend_html = f'''
<div style="position: fixed; bottom: 50px; right: 50px; width: 260px; background-color: white;
     border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Legend</p>
<p style="margin: 5px 0;"><span style="color:red; font-weight:bold;">&#10006;</span> Western Sydney International</p>
<p style="margin: 5px 0;">{dot('blue')} Selected Vertiports ({len(selected)})</p>
<p style="margin: 5px 0;">{dot('red')} Covered areas ({len(covered)})</p>
<p style="margin: 5px 0;">{dot('orange')} Uncovered areas ({len(uncovered)})</p>
<p style="margin: 5px 0;">{line(FLIGHT_COL)} Flight to WSI</p>
<p style="margin: 5px 0;"><i style="border-top:2px dashed #1f77b4; width:16px; display:inline-block;"></i> {RANGE_KM} km eVTOL range</p>
</div>'''
m_map.get_root().html.add_child(folium.Element(legend_html))
m_map.save('shuttle_solution_map_V2.html')
print('\nSaved shuttle_selected_sites_V2.csv, shuttle_area_coverage_V2.csv, shuttle_region_coverage_V2.csv, '
      'shuttle_solution_map_V2.html')