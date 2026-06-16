import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# STEP 1: LOAD DATA
# =========================

INPUT_FILE = "sa2_master_with_hospital_distance.xlsx"
HOSPITAL_FILE = "Emergency_hospitals_with_lat_lon.xlsx"
STATIONS_FILE = "Major_Power_Stations.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    print("\nLoaded file:", path)
    print("Rows:", len(df), "Cols:", len(df.columns))

    required = [
        "abs_code",
        "sa2_name",
        "lhd_name",
        "latitude",
        "longitude",
        "estimated_resident_population_count",
        "distance_to_nearest_emergency_hospital_km",
        "seifa_index_of_relative_socio_economic_advantage_and_disadvantage_irsad_rank_within_australia_decile",
        "total_dwellings_count",
        "total_persons_employed_aged_15_years_and_over_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))

    # Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "estimated_resident_population_count"])
    after = len(df)
    if after != before:
        print(f"Dropped {before - after} rows missing lat/lon/pop.")

    # Coerce numerics
    numeric_cols = [
        "latitude",
        "longitude",
        "estimated_resident_population_count",
        "distance_to_nearest_emergency_hospital_km",
        "total_dwellings_count",
        "total_persons_employed_aged_15_years_and_over_count",
        "owned_outright_pct",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop duplicate SA2 codes (if any)
    dup_count = int(df.duplicated(subset=["abs_code"]).sum())
    if dup_count > 0:
        print("Duplicate abs_code rows found:", dup_count)
        df = df.drop_duplicates(subset=["abs_code"], keep="first")
        print("After dropping duplicates, rows:", len(df))
    else:
        print("No duplicate abs_code rows found.")

    # Quick sanity checks
    print("\nSanity checks:")
    print(
        "Population min/median/max:",
        df["estimated_resident_population_count"].min(),
        df["estimated_resident_population_count"].median(),
        df["estimated_resident_population_count"].max(),
    )
    print(
        "Hospital distance (km) min/median/max:",
        df["distance_to_nearest_emergency_hospital_km"].min(),
        df["distance_to_nearest_emergency_hospital_km"].median(),
        df["distance_to_nearest_emergency_hospital_km"].max(),
    )
    print("Unique LHDs:", df["lhd_name"].nunique())

    return df.reset_index(drop=True)

def load_hospitals(path):
    h = pd.read_excel(path)
    print("\nLoaded file:", path)
    print("Rows:", len(h), "Cols:", len(h.columns))

    required = ["latitude", "longitude"]
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise KeyError("Hospital file missing columns:\n" + "\n".join(missing))

    #Drop row having missing latitute or longitude values
    h = h.dropna(subset=["latitude", "longitude"])
    h = h.reset_index(drop=True)

    return h

def load_stations(path):
    # St short for station
    st = pd.read_csv(path)
    
    # Only take operational stations
    st = st[st["operationalstatus"].str.strip().str.lower() == "operational"].copy()
    
    # Lat, lon coords for distance calc later
    st["X"] = pd.to_numeric(st["X"], errors="coerce")  
    
    # Remove missing 
    st = st.dropna(subset=["X", "Y"]).reset_index(drop=True)
    print(st)

    return st

# =========================
# STEP 2: FEATURE ENGINEERING
# =========================

def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.where((den > 0) & np.isfinite(den), num / den, np.nan)
    return pd.Series(out)


def add_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pop = pd.to_numeric(df["estimated_resident_population_count"], errors="coerce")
    emp = pd.to_numeric(df["total_persons_employed_aged_15_years_and_over_count"], errors="coerce")
    dwell = pd.to_numeric(df["total_dwellings_count"], errors="coerce")

    # Elderly share (emergency demand proxy)
    elderly_cols = [
        "persons_65_69_years_count",
        "persons_70_74_years_count",
        "persons_75_79_years_count",
        "persons_80_84_years_count",
        "persons_85_and_over_count",
    ]
    for c in elderly_cols:
        if c not in df.columns:
            df[c] = 0
    elderly_total = df[elderly_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    df["elderly_share"] = safe_divide(elderly_total, pop)

    # Paediatric share (optional)
    child_cols = [
        "persons_0_4_years_count",
        "persons_5_9_years_count",
        "persons_10_14_years_count",
    ]
    for c in child_cols:
        if c not in df.columns:
            df[c] = 0
    child_total = df[child_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    df["child_share"] = safe_divide(child_total, pop)

    # Transport shares (use employed as denominator since these are travel-to-work counts)
    if "used_at_least_one_form_of_public_transport_train_tram_bus_ferry_count" in df.columns:
        pt = df["used_at_least_one_form_of_public_transport_train_tram_bus_ferry_count"]
        df["public_transport_share"] = safe_divide(pt, emp)
    else:
        df["public_transport_share"] = np.nan

    if "travelled_to_work_by_car_as_a_driver_or_passenger_as_at_least_one_method_of_travel_count" in df.columns:
        car = df["travelled_to_work_by_car_as_a_driver_or_passenger_as_at_least_one_method_of_travel_count"]
        df["car_reliance_share"] = safe_divide(car, emp)
    else:
        df["car_reliance_share"] = np.nan

    if "no_motor_vehicles_count" in df.columns:
        no_veh = df["no_motor_vehicles_count"]
        df["no_vehicle_per_dwelling"] = safe_divide(no_veh, dwell)
    else:
        df["no_vehicle_per_dwelling"] = np.nan

    # Land feasibility
    if "land_area_ha" in df.columns:
        land = df["land_area_ha"]
        df["land_ha_per_dwelling"] = safe_divide(land, dwell)
    else:
        df["land_ha_per_dwelling"] = np.nan

    # Infrastructure readiness
    if "small_scale_solar_panel_system_installations_count" in df.columns:
        solar = df["small_scale_solar_panel_system_installations_count"]
        df["solar_per_1000_dwellings"] = safe_divide(solar, dwell) * 1000.0
    else:
        df["solar_per_1000_dwellings"] = np.nan

    if ("construction_count" in df.columns) and ("number_of_employee_jobs_total" in df.columns):
        constr = df["construction_count"]
        jobs_total = df["number_of_employee_jobs_total"]
        df["construction_share_jobs"] = safe_divide(constr, jobs_total)
    else:
        df["construction_share_jobs"] = np.nan
        
    # Fetch hospital distances to approximate the energy grid access. 
    # Assuming lower distance to station means better grid access.
    df["hospital_distance_km"] = pd.to_numeric(df["distance_to_nearest_emergency_hospital_km"], errors="coerce")


    sa2_coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    n = len(sa2_coords)
    # Initialise arr to store values
    dist_nearest = np.full(n, np.inf)
    
    # Do dist calc with our haversine function -> Take lowest distance out of those possible
    for _, row in stations.iterrows():
        tlat, tlon = float(row["Y"]), float(row["X"])
        dists = np.array([haversine_km(lat, lon, tlat, tlon) for lat, lon in sa2_coords])
        dist_nearest = np.minimum(dist_nearest, dists)
    
    # Add to df
    df["dist_nearest_station_km"] = dist_nearest

    # Rename for clarity in scoring
    df["hospital_distance_km"] = pd.to_numeric(df["distance_to_nearest_emergency_hospital_km"], errors="coerce")

    print("\nSTEP 2 diagnostics (engineered features):")
    diag_cols = [
        "elderly_share",
        "public_transport_share",
        "car_reliance_share",
        "land_ha_per_dwelling",
        "solar_per_1000_dwellings",
        "construction_share_jobs",
        "hospital_distance_km",
        "dist_nearest_station_km",
    ]
    for c in diag_cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy()
        print(f"{c:28s}  min/med/max = {np.nanmin(s):.4f} / {np.nanmedian(s):.4f} / {np.nanmax(s):.4f}")

    return df


# =========================
# STEP 3: USER-FRIENDLY, CONFIG-DRIVEN SCORING
# =========================

SCORING_CONFIG = {
    # Overall importance of each criterion in final score
    "criteria_weights": {
        "access_need": 0.35,          # Primary objective: improve access to hospitals
        "socio_need": 0.20,           # Equity objective: prioritise disadvantaged communities
        "land_feasibility": 0.15,     # Practicality: easier sites to build
        "transport_isolation": 0.15,  # Benefit: fewer alternatives means higher value from UAV access
        "infra_readiness": 0.15,      # Feasibility: energy and workforce signals for operations
    },

    # Indicators inside each criterion:
    # Format: ("column_name", direction, weight)
    # direction:
    #   "high" = higher values are better
    #   "low"  = lower values are better
    "criteria_indicators": {
        "access_need": [
            # Greater distance to emergency hospitals implies worse baseline access today,
            # so UAV vertiports can deliver larger accessibility gains.
            ("hospital_distance_km", "high", 0.50),

            # Higher elderly share indicates higher medical vulnerability and emergency demand,
            # so these communities benefit more from faster response.
            ("elderly_share", "high", 0.30),

            # Larger population means more people could benefit, improving efficiency of investment.
            ("estimated_resident_population_count", "high", 0.20),
        ],

        "socio_need": [
            # SEIFA IRSAD decile: lower values mean more disadvantaged,
            # so we treat lower as better to prioritise equity.
            ("seifa_index_of_relative_socio_economic_advantage_and_disadvantage_irsad_rank_within_australia_decile", "low", 1.00),
        ],

        "land_feasibility": [
            # More land per dwelling indicates lower density and fewer physical constraints,
            # making construction and operations easier.
            ("land_ha_per_dwelling", "high", 0.70),

            # Higher ownership rates can indicate more stable communities and potentially fewer barriers.
            ("owned_outright_pct", "high", 0.30),
        ],

        "transport_isolation": [
            # Higher car reliance suggests fewer alternatives, so added UAV access is more valuable.
            ("car_reliance_share", "high", 0.60),

            # Higher public transport share indicates existing mobility options,
            # so lower public transport share implies greater isolation.
            ("public_transport_share", "low", 0.40),
        ],

        "infra_readiness": [
            # Higher solar adoption can proxy stronger energy readiness and sustainability alignment.
            ("solar_per_1000_dwellings", "high", 0.40),

            # Higher construction share can proxy local workforce capacity for build and maintenance.
            ("construction_share_jobs", "high", 0.40),
            
            # Added here; We want closer values hence 'low'
            ("dist_nearest_station_km", "low", 0.20),
        ],
    },
}


def _minmax_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = np.nanmin(s.to_numpy())
    mx = np.nanmax(s.to_numpy())
    if np.isnan(mn) or np.isnan(mx) or np.isclose(mx - mn, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def _to_high_is_good(df: pd.DataFrame, col: str, direction: str) -> pd.Series:
    x01 = _minmax_01(df[col]).fillna(0.0)
    if direction == "high":
        return x01
    if direction == "low":
        return 1.0 - x01
    raise ValueError("direction must be 'high' or 'low'")


def _normalise_weights(w: dict) -> dict:
    total = float(sum(w.values()))
    if total <= 0:
        raise ValueError("Weights must sum to > 0")
    return {k: float(v) / total for k, v in w.items()}


def _normalise_indicator_weights(indicators: list) -> list:
    total = float(sum(w for _, _, w in indicators))
    if total <= 0:
        raise ValueError("Indicator weights must sum to > 0")
    return [(col, direction, float(w) / total) for col, direction, w in indicators]


def compute_scores(df: pd.DataFrame, config: dict = SCORING_CONFIG) -> pd.DataFrame:
    df = df.copy()

    crit_w = _normalise_weights(config["criteria_weights"])
    crit_inds = config["criteria_indicators"]

    # Criterion scores
    for crit, indicators in crit_inds.items():
        if crit not in crit_w:
            raise KeyError(f"Criterion '{crit}' is missing from criteria_weights.")

        available = []
        missing_cols = []
        for col, direction, w in indicators:
            if col in df.columns:
                available.append((col, direction, w))
            else:
                missing_cols.append(col)

        if missing_cols:
            print(f"Warning: {crit} missing columns, skipped: {missing_cols}")

        if not available:
            df[f"score_{crit}"] = 0.0
            continue

        available = _normalise_indicator_weights(available)

        score = np.zeros(len(df), dtype=float)
        for col, direction, w in available:
            score += w * _to_high_is_good(df, col, direction).to_numpy()

        df[f"score_{crit}"] = score

    # Final total score
    total = np.zeros(len(df), dtype=float)
    for crit, w in crit_w.items():
        total += w * df[f"score_{crit}"].fillna(0.0).to_numpy()

    df["score_total"] = total

    return df

from math import radians, sin, cos, sqrt, atan2

# =========================
# STEP 4: NETWORK-BASED VERTIPORT SELECTION (COVERAGE + SCORE)
# =========================

# ---- Network parameters (edit these) ----
K_PORTS = 10
MIN_PORT_SEPARATION_KM = 40.0       # set 0 to disable spacing
UAV_SPEED_KMH = 160.0               # used for SA2->port time proxy
UAV_RANGE_KM = 120.0                # set None to disable range constraint
COVERAGE_THRESHOLD_MIN = 30.0       # SA2 is "covered" if SA2->port time <= threshold
ALPHA_COVERAGE = 0.75               # 0..1: higher = prioritise coverage, lower = prioritise suitability score


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def estimate_minutes(distance_km: np.ndarray, speed_kmh: float) -> np.ndarray:
    return (distance_km / speed_kmh) * 60.0


def compute_sa2_to_sa2_distance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Precompute distances between all SA2 centroids.
    D[i, j] = distance km from SA2 i to SA2 j.
    """
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        lat1, lon1 = coords[i]
        for j in range(n):
            lat2, lon2 = coords[j]
            D[i, j] = haversine_km(lat1, lon1, lat2, lon2)
    return D


def select_vertiports_network(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy facility location selection:
    Choose K ports to maximise a blended objective:
      gain = ALPHA_COVERAGE * (newly covered population / total population)
           + (1-ALPHA_COVERAGE) * suitability_score(port)

    Coverage is based on SA2 -> port travel time under UAV_SPEED_KMH and threshold.
    """
    df = df.copy().reset_index(drop=True)

    # Normalise suitability score for blending (0-1)
    df["score_total_01"] = _minmax_01(df["score_total"]).fillna(0.0)

    pop = pd.to_numeric(df["estimated_resident_population_count"], errors="coerce").fillna(0.0).to_numpy()
    total_pop = float(pop.sum()) if pop.sum() > 0 else 1.0

    # Distance & time matrices (SA2 -> candidate port)
    D = compute_sa2_to_sa2_distance_matrix(df)  # n x n
    T = estimate_minutes(D, UAV_SPEED_KMH)

    # Coverage boolean: cover[i, p] means SA2 i can reach port p within threshold
    cover = T <= float(COVERAGE_THRESHOLD_MIN)
    if UAV_RANGE_KM is not None:
        cover = cover & (D <= float(UAV_RANGE_KM))

    # Greedy selection
    selected_idx = []
    covered_mask = np.zeros(len(df), dtype=bool)

    def respects_spacing(candidate_idx: int) -> bool:
        if MIN_PORT_SEPARATION_KM <= 0 or not selected_idx:
            return True
        lat_c, lon_c = df.loc[candidate_idx, ["latitude", "longitude"]]
        for s in selected_idx:
            lat_s, lon_s = df.loc[s, ["latitude", "longitude"]]
            if haversine_km(lat_c, lon_c, lat_s, lon_s) < float(MIN_PORT_SEPARATION_KM):
                return False
        return True

    remaining = set(range(len(df)))

    for _ in range(int(K_PORTS)):
        best = None
        best_gain = -1.0

        for p in list(remaining):
            if not respects_spacing(p):
                continue

            newly = cover[:, p] & (~covered_mask)
            newly_pop = float(pop[newly].sum())
            coverage_gain = newly_pop / total_pop

            suitability_gain = float(df.loc[p, "score_total_01"])
            gain = float(ALPHA_COVERAGE) * coverage_gain + (1.0 - float(ALPHA_COVERAGE)) * suitability_gain

            if gain > best_gain:
                best_gain = gain
                best = p

        if best is None:
            break

        selected_idx.append(best)
        covered_mask = covered_mask | cover[:, best]
        remaining.remove(best)

    selected = df.loc[selected_idx].copy()
    selected = selected.sort_values("score_total", ascending=False).reset_index(drop=True)
    selected.insert(0, "selected_rank", range(1, len(selected) + 1))

    # Print quick coverage stat
    covered_pop = float(pop[covered_mask].sum())
    covered_pct = covered_pop / total_pop * 100.0
    print(f"\nSTEP 4: Selected {len(selected)} vertiports.")
    print(f"STEP 4: Population covered within {COVERAGE_THRESHOLD_MIN:.0f} min to a port: {covered_pct:.1f}%")
    
    codes = selected["abs_code"]
    for c in codes:
        name = df.loc[df["abs_code"] == c, "sa2_name"].values[0]
        print(name)
    
    # Covered_pct has the population % as a decimal
    return selected, covered_pct

def attach_nearest_hospital_to_ports(ports, hospitals):
    names, lats, lons, dists = [], [], [], []

    for _, row in ports.iterrows():

        best_d = float("inf")
        best_h = None

        #Find the nearest hospital to the selected vertiport and its distance
        for _, h in hospitals.iterrows():
            d = haversine_km(
                row["latitude"], row["longitude"],
                h["latitude"], h["longitude"]
            )

            if d < best_d:
                best_d = d
                best_h = h
 
        #Add the information of the hospitals to the respected lists
        names.append(best_h.get("Name", "Hospital"))
        lats.append(best_h["latitude"])
        lons.append(best_h["longitude"])
        dists.append(best_d)

    #Add columns to selected ports data
    ports = ports.copy()
    ports["nearest_hospital_name"] = names
    ports["nearest_hospital_lat"] = lats
    ports["nearest_hospital_lon"] = lons
    ports["nearest_hospital_dist"] = dists

    return ports

# =========================
# STEP 5: NETWORK ACCESSIBILITY SIMULATION
# =========================

ROAD_SPEED_KMH = 80.0
UAV_SPEED_KMH_NETWORK = 160.0


def compute_accessibility(df, selected_ports):

    df = df.copy()

    # Baseline travel time to hospital (proxy using distance column)
    df["baseline_time_min"] = (
        df["hospital_distance_km"] / ROAD_SPEED_KMH
    ) * 60.0

    # Compute distance from each SA2 to nearest vertiport
    port_coords = selected_ports[["latitude", "longitude"]].to_numpy()

    sa2_coords = df[["latitude", "longitude"]].to_numpy()

    nearest_port_dist = []

    for lat1, lon1 in sa2_coords:

        dists = [
            haversine_km(lat1, lon1, lat2, lon2)
            for lat2, lon2 in port_coords
        ]

        nearest_port_dist.append(min(dists))

    df["dist_to_port_km"] = nearest_port_dist

    # UAV travel time SA2 → vertiport
    df["time_sa2_to_port"] = (
        df["dist_to_port_km"] / UAV_SPEED_KMH_NETWORK
    ) * 60.0

    # Assume vertiport located near town → hospital distance proxy
    # Using same hospital distance variable as approximation
    df["time_port_to_hospital"] = (
        df["hospital_distance_km"] / UAV_SPEED_KMH_NETWORK
    ) * 60.0

    # Total UAV response time
    df["uav_time_min"] = (
        df["time_sa2_to_port"] + df["time_port_to_hospital"]
    )

    # Improvement
    df["time_saved_min"] = (
        df["baseline_time_min"] - df["uav_time_min"]
    )

    return df

def evaluate_network(df):

    avg_baseline = df["baseline_time_min"].mean()
    avg_uav = df["uav_time_min"].mean()

    improvement = avg_baseline - avg_uav

    print("\nNETWORK PERFORMANCE")
    print("-------------------")

    print(f"Average baseline response time: {avg_baseline:.1f} minutes")
    print(f"Average UAV response time: {avg_uav:.1f} minutes")
    print(f"Average improvement: {improvement:.1f} minutes")

    # Coverage metrics
    for threshold in [30, 45, 60]:

        baseline_cov = (
            (df["baseline_time_min"] <= threshold).mean() * 100
        )

        uav_cov = (
            (df["uav_time_min"] <= threshold).mean() * 100
        )

        print(
            f"\nPopulation within {threshold} min hospital access:"
        )
        print(f"Baseline: {baseline_cov:.1f}%")
        print(f"With UAV: {uav_cov:.1f}%")


# =========================
# STEP 6: SENSITIVITY ANALYSIS (parameter sweeps)
# =========================
# This runs multiple model configurations and records network performance.
# You can sweep: number of ports, spacing, UAV speed, range, coverage threshold, and scoring weights.

def run_one_scenario(df_base: pd.DataFrame, scenario_name: str, i) -> dict:
    """
    Runs Steps 3–5 for the current global parameters + SCORING_CONFIG.
    Returns a dictionary of summary metrics for one scenario.
    """
    # Step 3: score (config-driven)
    df_scored = compute_scores(df_base, SCORING_CONFIG)

    # Step 4: select ports (uses global K_PORTS, MIN_PORT_SEPARATION_KM, UAV_SPEED_KMH, UAV_RANGE_KM, COVERAGE_THRESHOLD_MIN, ALPHA_COVERAGE)
    ports, pct_currentscenario = select_vertiports_network(df_scored)

    #Find hospitals closest to selected locations
    ports = attach_nearest_hospital_to_ports(ports, hospitals)

    #Plot the acquired port locations on map
    plot_vertiports_map(df_scored, ports, i)
    
    # Step 5: accessibility (uses ROAD_SPEED_KMH, UAV_SPEED_KMH_NETWORK)
    results = compute_accessibility(df_scored, ports)

    # Metrics
    avg_baseline = float(results["baseline_time_min"].mean())
    avg_uav = float(results["uav_time_min"].mean())
    avg_improvement = avg_baseline - avg_uav

    # Population coverage within thresholds (weighted by population, not just % of SA2s)
    pop = pd.to_numeric(results["estimated_resident_population_count"], errors="coerce").fillna(0.0).to_numpy()
    total_pop = float(pop.sum()) if pop.sum() > 0 else 1.0

    def pop_coverage(col: str, thr: float) -> float:
        mask = pd.to_numeric(results[col], errors="coerce").to_numpy() <= thr
        return float(pop[mask].sum()) / total_pop * 100.0

    cov30_base = pop_coverage("baseline_time_min", 30)
    cov45_base = pop_coverage("baseline_time_min", 45)
    cov60_base = pop_coverage("baseline_time_min", 60)

    cov30_uav = pop_coverage("uav_time_min", 30)
    cov45_uav = pop_coverage("uav_time_min", 45)
    cov60_uav = pop_coverage("uav_time_min", 60)

    # Equity-style cut: improvement by SEIFA decile (1 = most disadvantaged)
    seifa = pd.to_numeric(
        results["seifa_index_of_relative_socio_economic_advantage_and_disadvantage_irsad_rank_within_australia_decile"],
        errors="coerce",
    )
    results["_seifa_decile"] = seifa
    results["_pop"] = pop

    # population-weighted mean time saved by decile
    decile_rows = []
    for d in range(1, 11):
        sub = results[results["_seifa_decile"] == d].copy()
        if len(sub) == 0:
            decile_rows.append(np.nan)
            continue
        w = pd.to_numeric(sub["_pop"], errors="coerce").fillna(0.0).to_numpy()
        ts = pd.to_numeric(sub["time_saved_min"], errors="coerce").fillna(np.nan).to_numpy()
        if w.sum() <= 0:
            decile_rows.append(float(np.nanmean(ts)))
        else:
            decile_rows.append(float(np.nansum(ts * w) / np.nansum(w)))
    # difference between most disadvantaged (1–3) and least disadvantaged (8–10)
    low_mean = float(np.nanmean(decile_rows[0:3]))
    high_mean = float(np.nanmean(decile_rows[7:10]))
    equity_gap = low_mean - high_mean  # positive means disadvantaged benefit more

    return {
        "scenario": scenario_name,
        "k_ports": int(K_PORTS),
        "min_sep_km": float(MIN_PORT_SEPARATION_KM),
        "uav_speed_kmh": float(UAV_SPEED_KMH),
        "uav_range_km": (None if UAV_RANGE_KM is None else float(UAV_RANGE_KM)),
        "coverage_threshold_min": float(COVERAGE_THRESHOLD_MIN),
        "alpha_coverage": float(ALPHA_COVERAGE),
        "road_speed_kmh": float(ROAD_SPEED_KMH),
        "avg_baseline_min": avg_baseline,
        "avg_uav_min": avg_uav,
        "avg_improvement_min": float(avg_improvement),
        "cov30_base_pop_pct": float(cov30_base),
        "cov45_base_pop_pct": float(cov45_base),
        "cov60_base_pop_pct": float(cov60_base),
        "cov30_uav_pop_pct": float(cov30_uav),
        "cov45_uav_pop_pct": float(cov45_uav),
        "cov60_uav_pop_pct": float(cov60_uav),
        "equity_gap_time_saved_min": float(equity_gap),
        
        "pct_coverage" : pct_currentscenario,
    }


def sensitivity_analysis(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Define sweeps here. Each sweep changes globals (parameters), runs a scenario, and stores results.
    """
    global K_PORTS, MIN_PORT_SEPARATION_KM, UAV_SPEED_KMH, UAV_RANGE_KM, COVERAGE_THRESHOLD_MIN, ALPHA_COVERAGE
    global ROAD_SPEED_KMH, UAV_SPEED_KMH_NETWORK

    rows = []

    # ---- Baseline parameters (set your preferred defaults) ----
    K_PORTS = 10
    MIN_PORT_SEPARATION_KM = 40.0
    UAV_SPEED_KMH = 160.0
    UAV_SPEED_KMH_NETWORK = 160.0
    UAV_RANGE_KM = 120.0
    COVERAGE_THRESHOLD_MIN = 30.0
    ALPHA_COVERAGE = 0.75
    ROAD_SPEED_KMH = 80.0

    
    while True:
        i = 1
        rows.append(run_one_scenario(df_base, "base",i))

        # ---- Sweep 1: number of ports ----
        for k in [5, 8, 10, 12, 15]:
            i += 1
            K_PORTS = k
            rows.append(run_one_scenario(df_base, f"k_ports_{k}",i))

        # reset
        K_PORTS = 10

        # ---- Sweep 2: UAV speed ----
        for sp in [120, 160, 200]:
            i += 1
            UAV_SPEED_KMH = float(sp)
            UAV_SPEED_KMH_NETWORK = float(sp)
            rows.append(run_one_scenario(df_base, f"uav_speed_{sp}",i))

        # reset
        UAV_SPEED_KMH = 160.0
        UAV_SPEED_KMH_NETWORK = 160.0

        # ---- Sweep 3: coverage threshold (what counts as "served by a port") ----
        for t in [20, 30, 45]:
            i += 1
            COVERAGE_THRESHOLD_MIN = float(t)
            rows.append(run_one_scenario(df_base, f"threshold_{t}min",i))

        COVERAGE_THRESHOLD_MIN = 30.0

        # ---- Sweep 4: spacing constraint ----
        for s in [0, 20, 40, 80]:
            i += 1
            MIN_PORT_SEPARATION_KM = float(s)
            rows.append(run_one_scenario(df_base, f"spacing_{s}km",i))

        MIN_PORT_SEPARATION_KM = 40.0

        # ---- Sweep 5: equity vs efficiency weighting (alpha) ----
        for a in [0.3, 0.5, 0.75, 0.9]:
            i += 1
            ALPHA_COVERAGE = float(a)
            rows.append(run_one_scenario(df_base, f"alpha_{a}",i))

        ALPHA_COVERAGE = 0.75

        # ---- Optional: scoring weight scenarios (edit freely) ----
        # Example: equity-focused (increase socio_need)
        global SCORING_CONFIG
        base_weights = dict(SCORING_CONFIG["criteria_weights"])

        SCORING_CONFIG["criteria_weights"] = {
            "access_need": 0.30,
            "socio_need": 0.35,          # more equity focus
            "land_feasibility": 0.10,
            "transport_isolation": 0.10,
            "infra_readiness": 0.15,
        }
        i += 1
        rows.append(run_one_scenario(df_base, "weights_equity_focus",i))

        # Example: access-focused (more hospital distance + population)
        SCORING_CONFIG["criteria_weights"] = {
            "access_need": 0.50,
            "socio_need": 0.15,
            "land_feasibility": 0.10,
            "transport_isolation": 0.10,
            "infra_readiness": 0.15,
        }
        i += 1
        rows.append(run_one_scenario(df_base, "weights_access_focus",i))
        break

    # restore
    SCORING_CONFIG["criteria_weights"] = base_weights

    return pd.DataFrame(rows)


def plot_sensitivity(df_sens: pd.DataFrame):
    """
    Simple plots for report figures.
    No custom colors (defaults only).
    """
    # Plot: ports vs average improvement
    subset = df_sens[df_sens["scenario"].str.startswith("k_ports_")].copy()
    if len(subset) > 0:
        subset["k"] = subset["k_ports"].astype(int)
        subset = subset.sort_values("k")
        plt.figure()
        plt.plot(subset["k"], subset["avg_improvement_min"], marker="o")
        plt.xlabel("Number of vertiports (K)")
        plt.ylabel("Average time saved (minutes)")
        plt.title("Sensitivity: time saved vs number of vertiports")
        plt.tight_layout()
        plt.savefig("sensitivity_k_ports.png", dpi=200)

    # Plot: UAV speed vs average improvement
    subset = df_sens[df_sens["scenario"].str.startswith("uav_speed_")].copy()
    if len(subset) > 0:
        subset["speed"] = subset["uav_speed_kmh"].astype(float)
        subset = subset.sort_values("speed")
        plt.figure()
        plt.plot(subset["speed"], subset["avg_improvement_min"], marker="o")
        plt.xlabel("UAV cruise speed (km/h)")
        plt.ylabel("Average time saved (minutes)")
        plt.title("Sensitivity: time saved vs UAV speed")
        plt.tight_layout()
        plt.savefig("sensitivity_uav_speed.png", dpi=200)
        
    # Pie plot of criteria weightings
    pieplot_labels = list(SCORING_CONFIG["criteria_weights"].keys())
    pieplot_weights = list(SCORING_CONFIG["criteria_weights"].values())
    plt.figure()
    plt.pie(pieplot_weights, labels = pieplot_labels, autopct="%1.0f%%", wedgeprops={"edgecolor": "black"})
    plt.title("Criteria weight contribution")
    plt.tight_layout()
    plt.savefig("criteria_weight_pie.png", dpi=200)
    
    # Pie plot of indicator weightings contribution to whole
    cum_labels = []
    cum_weights = []
    for i in SCORING_CONFIG["criteria_weights"]:
        # Goes through all the diff criterias
        current_criteria_weight = SCORING_CONFIG["criteria_weights"][i]
        # Goes through the indicators in each criteria
        for j in SCORING_CONFIG["criteria_indicators"][i]:
            # j[0] is col name, j[2] is weighting which multiplied to criteria weight
            # Name way too long so it needs to be shortened
            if j[0] == "seifa_index_of_relative_socio_economic_advantage_and_disadvantage_irsad_rank_within_australia_decile":
                cum_labels.append("seifa_decile")
            else:
                cum_labels.append(j[0])
            cum_weights.append(j[2] * current_criteria_weight)
            
    plt.figure(figsize = (9,7))
    plt.pie(cum_weights, labels = cum_labels, autopct="%1.0f%%", wedgeprops={"edgecolor": "black"})
    plt.title("Criteria weight contribution")
    plt.savefig("indicator_pie.png", dpi=200)
    
    # Coverage boxplot
    coverage_data = df_sens[df_sens["scenario"] == "base"].iloc[0]
    coverage_time_cases = [30, 45, 60]
    base_values = [coverage_data[f"cov{i}_base_pop_pct"] for i in coverage_time_cases]
    uav_values = [coverage_data[f"cov{i}_uav_pop_pct"] for i in coverage_time_cases]
    
    # Setup for ticks
    cov_x = np.arange(3)
    
    plt.figure()
    plt.bar(cov_x - 0.15, base_values, width = 0.3, label = 'Base')
    plt.bar(cov_x + 0.15, uav_values, width = 0.3, label = 'UAV')
    plt.xticks(cov_x, [f"{i} mins" for i in coverage_time_cases])
    plt.ylabel("Population covarage (%)")
    plt.title("Base v UAV coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Base_v_UAV.png", dpi = 200)
    
    # Pct coverage (%) for scenarios
    plt.figure()
    plt.bar(df_sens["scenario"], df_sens["pct_coverage"])
    plt.xticks(rotation = 90)
    plt.ylabel("Population covarage (%)")
    plt.title("Coverage on diff scenarios")
    plt.tight_layout()
    plt.savefig("pct_coverage.png", dpi = 200)
    
    
            
    


# =========================
# STEP 7: MAP VISUALISATION
# =========================

import folium

def plot_vertiports_map(df, selected_ports, i):
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Plot selected vertiports
    for _, row in selected_ports.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['sa2_name']} (Rank {row['selected_rank']})",
            tooltip=row["sa2_name"],
            tiles="CartoDB positron"
        ).add_to(m)

        #Plot nearest hospital to each vertiport
        folium.Marker(
            location=[row["nearest_hospital_lat"], row["nearest_hospital_lon"]],
            icon=folium.Icon(color="red", icon="h-square", prefix="fa"),
            popup=row["nearest_hospital_name"]
        ).add_to(m)

        #Label distance to nearest hospital for each location
        folium.PolyLine(
            locations=[
                [row["latitude"], row["longitude"]],
                [row["nearest_hospital_lat"], row["nearest_hospital_lon"]]
            ],
            color="green",
            weight=2,
            opacity=0.7,
            tooltip=f"{row['sa2_name']} → {row['nearest_hospital_name']}: {row['nearest_hospital_dist']:.1f} km"
        ).add_to(m)

        
        mid_lat = (row["latitude"] + row["nearest_hospital_lat"]) / 2
        mid_lon = (row["longitude"] + row["nearest_hospital_lon"]) / 2
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px; color:green; background:white; padding:1px 3px; border-radius:3px; opacity:0.8">{row["nearest_hospital_dist"]:.0f} km</div>',
                icon_size=(55, 18),
                icon_anchor=(27, 9),
            )
        ).add_to(m)

    # Label the distance between every pair of vertiports
    ports_list = selected_ports.reset_index(drop=True)
    for p1 in range(len(ports_list)):
        for p2 in range(p1 + 1, len(ports_list)):
            lat1 = ports_list.loc[p1, "latitude"]
            lon1 = ports_list.loc[p1, "longitude"]
            lat2 = ports_list.loc[p2, "latitude"]
            lon2 = ports_list.loc[p2, "longitude"]

            dist = haversine_km(lat1, lon1, lat2, lon2)

            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="red",
                weight=3,
                opacity=0.5,
                tooltip=f"{ports_list.loc[p1, 'sa2_name']} ↔ {ports_list.loc[p2, 'sa2_name']}: {dist:.1f} km"
            ).add_to(m)

            # Put distance label at midpoint of the line
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f'''<div style="font-size:9px; color:blue; background:white;
                    padding:1px 3px; border-radius:3px; opacity:0.8">{dist:.0f} km</div>''',
                    icon_size=(55, 18),
                    icon_anchor=(27, 9),
                )
            ).add_to(m)


    m.save(f"vertiports{i}_map.html")
    print(f"Saved: vertiports{i}_map.html")
    
# =========================
# STEP 8: MAXING VERTIPORT % WIP
# =========================
def output_average_vp_coverage(sens):
    # Convert to numpy to perform numeric stuff
    pct_coverage_list = sens["pct_coverage"].to_numpy();
    
    average_pct_coverage = np.mean(pct_coverage_list);
    
    print(f"Current average coverage across scenario is {average_pct_coverage:.1f}%")
          
    return average_pct_coverage;



# =========================
# Add to MAIN (at the end)
# =========================
# After you compute df = load_data(...) and df = add_features(...),
# run sensitivity:

if __name__ == "__main__":
    hospitals = load_hospitals(HOSPITAL_FILE)
    stations = load_stations(STATIONS_FILE)
    df = load_data(INPUT_FILE)
    df = add_features(df, stations)

    # Run sensitivity experiments
    sens = sensitivity_analysis(df)
    
    output_average_vp_coverage(sens);

    # Save results table
    sens.to_excel("sensitivity_results.xlsx", index=False)
    print("\nSaved: sensitivity_results.xlsx")

    # Make 1–2 simple figures for the report
    plot_sensitivity(sens)
    print("Saved plots")
