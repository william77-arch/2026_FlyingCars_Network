"""
convert_to_cleaned_abs_ldh.py

Converts NSW_SA2_DataExplorer_Combined.xlsx (the live-pulled ABS API data)
into the same 9-sheet, per-LHD, 23-indicator format as Cleaned_ABS_LDH_data.xlsx,
so it's a drop-in replacement using current data instead of the old
FULL_ABS_Data.xlsx snapshot.

NSW_SA2_DataExplorer_Combined.xlsx has one flat sheet across all of NSW, no
per-LHD split, so we rebuild that split using FULL_ABS_Data_OLD_sheets.xlsx's
region sheets purely as an ABS_Code -> LHD lookup (no indicator data read
from it). Column names are also completely different between the two
sources - INDICATOR_MAPPING below matches each of the 23 original
indicators to its (heading, label) pair in the new data, verified against
known SA2 values (owned_outright_pct matched exactly; population/dwelling/
business figures showed small realistic revisions consistent with newer
data). Lat/lon/area come from NSW_SA2_with_LatLon.csv, not the API file.

Row filtering (reverse-engineered from Cleaned_ABS_LDH_data.xlsx, verified
to reproduce it exactly for all 9 sheets): drop rows with no abs_code, and
drop rows where every one of the 23 indicators is blank.

Usage:
    pip install pandas openpyxl --break-system-packages
    python convert_to_cleaned_abs_ldh.py
"""

import pandas as pd

INDICATORS_FILE = "NSW_SA2_DataExplorer_Combined.xlsx"
INDICATORS_SHEET = "Indicators"
LATLON_FILE = "NSW_SA2_with_Calculated_Coords.csv"

# region lookup only - must be the version with the per-region sheets
# (Far West, Central Coast, etc.), not the sa2_centroids_2021 one
REGION_LOOKUP_FILE = "FULL_ABS_Data_OLD_sheets.xlsx"

OUTPUT_FILE = "Cleaned_ABS_LDH_data.xlsx"

REGION_SHEETS = [
    "Far West", "Central Coast", "New Hunter England", "Illawarra",
    "Mid North Coast", "Murrumbidgee", "Northern NSW", "Southern NSW",
    "Western NSW",
]

# old snake_case name -> (heading, label) in the new API-sourced data
INDICATOR_MAPPING = {
    "owned_outright_pct": (
        "Tenure type - Occupied private dwellings - Census",
        "Occupied private dwellings:Owned outright (%)"),
    "seifa_index_of_relative_socio_economic_advantage_and_disadvantage_irsad_rank_within_australia_decile": (
        "Socio-economic indexes for areas (SEIFA) - rank within Australia - Census",
        "SEIFA Index of relative socio-economic advantage and disadvantage (IRSAD):Rank within Australia (decile)"),
    "used_at_least_one_form_of_public_transport_train_tram_bus_ferry_count": (
        "Method of travel to work - Employed persons aged 15 years and over - Census",
        "People who travelled to work using at least one form of public transport (train, tram, bus, ferry) (no.)"),
    "travelled_to_work_by_car_as_a_driver_or_passenger_as_at_least_one_method_of_travel_count": (
        "Method of travel to work - Employed persons aged 15 years and over - Census",
        "People who travelled to work by car (as a driver or passenger) as at least one of their methods of travel (no.)"),
    "total_persons_employed_aged_15_years_and_over_count": (
        "Industry of employment - Persons aged 15 years and over - Census",
        "Industry of employment:Total persons employed aged 15 years and over (no.)"),
    "no_motor_vehicles_count": (
        "Number of motor vehicles per household - Occupied private dwellings - Census",
        "Households with no motor vehicles (no.)"),
    "number_of_employee_jobs_total": (
        "Jobs in Australia - year ended 30 June",
        "Total number of employee jobs"),
    "estimated_resident_population_count": (
        "Estimated resident population - 30 June",
        "Estimated resident population: Persons (no.)"),
    "population_density_persons_km2": (
        "Estimated resident population - 30 June",
        "Population density (persons/km2)"),
    "median_age_persons_years": (
        "Estimated resident population - 30 June",
        "Median age:Persons (years)"),
    "persons_0_4_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 0-4 years (no.)"),
    "persons_5_9_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 5-9 years (no.)"),
    "persons_10_14_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 10-14 years (no.)"),
    "persons_65_69_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 65-69 years (no.)"),
    "persons_70_74_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 70-74 years (no.)"),
    "persons_75_79_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 75-79 years (no.)"),
    "persons_80_84_years_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 80-84 years (no.)"),
    "persons_85_and_over_count": (
        "Estimated resident population - Persons - 30 June",
        "Estimated resident population:Persons aged 85 and over (no.)"),
    "land_area_ha": (
        "Land area",
        "Land area (ha)"),
    "small_scale_solar_panel_system_installations_count": (
        "Solar installations - year ended 31 December",
        "Small-scale solar panel system installations (no.)"),
    "persons_who_have_need_for_assistance_with_core_activities_pct": (
        "Core activity need for assistance - Census",
        "Persons who have need for assistance with core activities (%)"),
    "construction_count": (
        "Number of businesses by industry - 30 June",
        "Number of businesses:Construction industry (no.)"),
    "total_dwellings_count": (
        "Estimated dwelling stock - June quarter",
        "Estimated dwelling stock:Total estimated dwelling stock (no.)"),
}


def load_indicators():
    print(f"Reading '{INDICATORS_FILE}' ({INDICATORS_SHEET})...")
    raw = pd.read_excel(INDICATORS_FILE, sheet_name=INDICATORS_SHEET, header=None)

    heading_row = raw.iloc[0]
    label_row = raw.iloc[1]
    abs_codes = raw.iloc[2:, 0].reset_index(drop=True)

    # heading row is merged across columns that share a topic - only the
    # leftmost cell in each merged run actually holds text, so forward-fill
    filled_headings = {}
    last_heading = ""
    for col_idx in range(1, raw.shape[1]):
        val = heading_row[col_idx]
        if pd.notna(val) and str(val).strip():
            last_heading = str(val).strip()
        filled_headings[col_idx] = last_heading

    position_lookup = {}
    for col_idx in range(1, raw.shape[1]):
        h = filled_headings[col_idx]
        l = "" if pd.isna(label_row[col_idx]) else str(label_row[col_idx]).strip()
        position_lookup[(h, l)] = col_idx

    result = pd.DataFrame({"abs_code": abs_codes})
    missing = []
    for old_name, (heading, label) in INDICATOR_MAPPING.items():
        pos = position_lookup.get((heading, label))
        if pos is None:
            missing.append(old_name)
            continue
        result[old_name] = raw.iloc[2:, pos].reset_index(drop=True)

    if missing:
        print(f"  ! {len(missing)} indicator(s) not found in the new data "
              f"(ABS may have renamed/discontinued them) -- skipped: {missing}")
    else:
        print(f"  All {len(INDICATOR_MAPPING)} mapped indicators found.")

    return result


def load_latlon():
    print(f"Reading '{LATLON_FILE}'...")
    df = pd.read_csv(LATLON_FILE)
    df = df.rename(columns={
        "ABS_Code": "abs_code", "SA2_Name": "sa2_name",
        "longitude": "longitude", "latitude": "latitude",
        "Area_KM2": "area_km2", "Equiv_Radius_KM": "equiv_radius_km",
    })
    return df[["abs_code", "sa2_name", "longitude", "latitude", "area_km2", "equiv_radius_km"]]


def load_region_lookup():
    print(f"Reading region assignments from '{REGION_LOOKUP_FILE}'...")

    xl = pd.ExcelFile(REGION_LOOKUP_FILE)
    actual_sheets = xl.sheet_names
    print(f"  Sheets actually present in '{REGION_LOOKUP_FILE}': {actual_sheets}")

    normalized_actual = {s.strip().lower(): s for s in actual_sheets}

    lookup = {}
    for expected in REGION_SHEETS:
        actual = None
        if expected in actual_sheets:
            actual = expected
        elif expected.strip().lower() in normalized_actual:
            actual = normalized_actual[expected.strip().lower()]
            print(f"  ! '{expected}' matched to '{actual}' (whitespace/case difference only).")
        else:
            print(f"  ! Sheet '{expected}' not found in '{REGION_LOOKUP_FILE}' -- skipping this region.")
            continue

        df = pd.read_excel(REGION_LOOKUP_FILE, sheet_name=actual, header=0, usecols=["abs_code"])
        for code in df["abs_code"].dropna():
            lookup[str(int(code))] = expected

    print(f"  {len(lookup)} SA2 codes assigned to a region.")
    return lookup


def main():
    indicators = load_indicators()
    latlon = load_latlon()
    region_lookup = load_region_lookup()

    indicators["abs_code"] = indicators["abs_code"].astype(str).str.strip()
    latlon["abs_code"] = latlon["abs_code"].astype(str).str.strip()

    merged = pd.merge(latlon, indicators, on="abs_code", how="left")
    merged["region"] = merged["abs_code"].map(region_lookup)

    indicator_cols = [c for c in INDICATOR_MAPPING if c in merged.columns]

    unassigned = merged["region"].isna().sum()
    if unassigned:
        print(f"  ! {unassigned} SA2(s) in '{LATLON_FILE}' have no region assignment -- won't appear in output.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for sheet in REGION_SHEETS:
            region_df = merged[merged["region"] == sheet].copy()
            region_df = region_df[
                region_df["abs_code"].notna()
                & region_df[indicator_cols].notna().any(axis=1)
            ]
            out_cols = ["abs_code", "sa2_name", "longitude", "latitude",
                        "area_km2", "equiv_radius_km"] + indicator_cols
            region_df[out_cols].to_excel(writer, sheet_name=sheet, index=False)
            print(f"  '{sheet}': {len(region_df)} rows x {len(out_cols)} columns")

    print(f"\nSuccess! Saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()