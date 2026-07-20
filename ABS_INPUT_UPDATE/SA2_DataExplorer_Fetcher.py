"""
SA2_DataExplorer_Fetcher.py

Pulls SA2-level "Data by Region" indicators from the ABS Data API, groups
them under ABS-style topic headings, and merges them with
NSW_SA2_with_Calculated_Coords.csv into one finished CSV.

The static per-theme spreadsheets on the ABS site only go down to LGA
level, so SA2 data has to come from the live API (ABS_REGIONAL_ASGS2021
dataflow). Dimension order there is MEASURE, REGIONTYPE, ASGS_2021,
FREQUENCY, 0-indexed. MEASURE is a hierarchical codelist - each indicator
code has a parent code whose name is the topic heading ABS shows above it
in their own spreadsheets, so we rebuild that from the parent link. We
pivot on the MEASURE code rather than the label since labels aren't
guaranteed unique.

Output:
  NSW_SA2_DataExplorer_Combined.xlsx  - indicators, grouped header, Reference sheet
  NSW_SA2_Full.csv                    - lat/lon + indicators merged, flat header

Usage:
    pip install pandas openpyxl requests --break-system-packages
    python SA2_DataExplorer_Fetcher.py
"""

import re
import time
from io import StringIO

import requests
import pandas as pd
import openpyxl

# Need to replace Input files on update

NSW_CODES_FILE = "NSW_SA2_Area_2021.csv"
LATLON_FILE = "NSW_SA2_with_Calculated_Coords.csv"

# Output
INDICATORS_OUTPUT = "NSW_SA2_DataExplorer_Combined.xlsx"
FINAL_OUTPUT = "NSW_SA2_Full.csv"
STRUCTURE_CACHE_FILE = "dsd_structure.json"

# Data source
API_BASE = "https://data.api.abs.gov.au/rest"
AGENCY = "ABS"

# Replace 2021 with 2026 in upcoming
DATAFLOW_ID = "ABS_REGIONAL_ASGS2021"

STRUCTURE_HEADERS = {"Accept": "application/vnd.sdmx.structure+json"}
DATA_HEADERS = {"Accept": "application/vnd.sdmx.data+csv"}

CODES_PER_BATCH = 40  # API's dataKey limit is 5,000 chars
REQUEST_TIMEOUT = 120
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 5

PARENT_KEY_CANDIDATES = ["parent", "parentCode", "parentId", "Parent"]

OLD_ABS_FALLBACK_FILE = "FULL_ABS_Data.xlsx"
OLD_ABS_FALLBACK_SHEET = "sa2_centroids_2021"


def normalize_label(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def get_with_retry(url, headers=None, params=None):
    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            print(f"    ! request failed (attempt {attempt}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SEC)
    raise last_err


def load_nsw_codes():
    print(f"Reading NSW SA2 codes from '{NSW_CODES_FILE}'...")
    df = pd.read_csv(NSW_CODES_FILE)
    codes = sorted(set(df["ABS_Code"].astype(str).str.strip()))
    print(f"  -> {len(codes)} NSW SA2 codes loaded.")
    return codes


def discover_structure():
    print(f"\n=== Stage 1: discovering structure of {DATAFLOW_ID} ===")
    url = f"{API_BASE}/dataflow/{AGENCY}/{DATAFLOW_ID}"
    resp = get_with_retry(url, headers=STRUCTURE_HEADERS,
                           params={"references": "all", "detail": "full"})

    with open(STRUCTURE_CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"  Raw structure cached to '{STRUCTURE_CACHE_FILE}'.")

    return resp.json()


def fetch_codelist_raw(codelist_id):
    url = f"{API_BASE}/codelist/{AGENCY}/{codelist_id}"
    resp = get_with_retry(url, headers=STRUCTURE_HEADERS, params={"detail": "full"})
    data = resp.json()
    for cl in data.get("data", {}).get("codelists", []):
        if cl.get("id") == codelist_id:
            return cl.get("codes", [])
    return []


def fetch_codelist_direct(codelist_id):
    try:
        raw = fetch_codelist_raw(codelist_id)
    except Exception as e:
        print(f"    ! Direct fetch of codelist '{codelist_id}' failed: {e}")
        return {}
    return {c["id"]: c.get("name", c["id"]) for c in raw}


def find_dimensions(structure_json):
    try:
        dsd = structure_json["data"]["dataStructures"][0]
        dim_list = dsd["dataStructureComponents"]["dimensionList"]["dimensions"]
    except (KeyError, IndexError, TypeError):
        print("  ! Could not locate dimensionList in the expected JSON path.")
        print("    Top-level keys found:", list(structure_json.get("data", {}).keys()))
        raise

    dims = []
    for d in dim_list:
        codelist_id = None
        enum_ref = d.get("localRepresentation", {}).get("enumeration")
        if enum_ref:
            codelist_id = enum_ref.split("=")[-1].split("(")[0].split(":")[-1]
        dims.append({"id": d.get("id"), "position": d.get("position"), "codelist_id": codelist_id})

    dims.sort(key=lambda x: x["position"])
    print("  Dimensions found (in key order):")
    for d in dims:
        print(f"    position {d['position']}: {d['id']}  (codelist: {d['codelist_id']})")

    needed_codelist_ids = {d["codelist_id"] for d in dims if d["codelist_id"]}

    codelists = {}
    for cl in structure_json.get("data", {}).get("codelists", []):
        cl_id = cl.get("id")
        if cl_id not in needed_codelist_ids:
            continue
        items = {c["id"]: c.get("name", c["id"]) for c in cl.get("codes", [])}
        if not items:
            print(f"  Codelist '{cl_id}' came back empty, fetching it directly...")
            items = fetch_codelist_direct(cl_id)
        codelists[cl_id] = items
        print(f"  Codelist '{cl_id}': {len(items)} codes (sample: {list(items.items())[:3]})")

    return dims, codelists


def find_region_dimension(dims, codelists, nsw_codes):
    region_dim = None
    for d in dims:
        cl = codelists.get(d["codelist_id"], {})
        overlap = sum(1 for code in nsw_codes if code in cl)
        if overlap > 0:
            print(f"  -> Dimension '{d['id']}' matches {overlap} of our NSW SA2 codes.")
            region_dim = d
            break

    if region_dim is None:
        print(f"  ! No dimension's codelist contained our SA2 codes. "
              f"Inspect '{STRUCTURE_CACHE_FILE}' by hand.")

    region_type_dim = next(
        (d for d in dims if "REGIONTYPE" in (d["id"] or "").upper()
         or "REGION_TYPE" in (d["id"] or "").upper()), None
    )
    sa2_key = None
    if region_type_dim:
        cl = codelists.get(region_type_dim["codelist_id"], {})
        sa2_key = next((k for k in cl if k.upper() == "SA2"), None)
        print(f"  -> Region-type dimension '{region_type_dim['id']}', SA2 code: {sa2_key}")

    return region_dim, region_type_dim, sa2_key


def build_measure_hierarchy(dims):
    measure_dim = next((d for d in dims if d["id"] == "MEASURE"), None)
    if not measure_dim or not measure_dim["codelist_id"]:
        print("  ! Could not find MEASURE dimension for heading lookup.")
        return {"heading": {}, "order": {}}

    print(f"\n=== Stage 2: building indicator headings from '{measure_dim['codelist_id']}' ===")
    raw_codes = fetch_codelist_raw(measure_dim["codelist_id"])
    if not raw_codes:
        print("  ! No codes returned -- headings will be blank.")
        return {"heading": {}, "order": {}}

    print(f"  {len(raw_codes)} measure codes fetched.")
    print(f"  First code entry keys: {list(raw_codes[0].keys())}")

    # top-level codes have no parent, so check across all codes not just the first
    parent_key = next(
        (cand for cand in PARENT_KEY_CANDIDATES if any(cand in c for c in raw_codes)),
        None
    )
    print(f"  Parent field detected: {parent_key!r}")

    def order_annotation(code_dict):
        for ann in code_dict.get("annotations", []):
            if ann.get("type") == "ORDER":
                try:
                    return int(ann.get("text"))
                except (TypeError, ValueError):
                    return None
        return None

    id_to_name = {}
    id_to_parent = {}
    id_to_order = {}
    for idx, c in enumerate(raw_codes):
        cid = c.get("id")
        id_to_name[cid] = c.get("name", cid)
        order_val = order_annotation(c)
        id_to_order[cid] = order_val if order_val is not None else idx
        parent_val = c.get(parent_key) if parent_key else None
        if isinstance(parent_val, dict):
            parent_val = parent_val.get("id")
        elif isinstance(parent_val, str) and ":" in parent_val:
            parent_val = parent_val.split(":")[-1].split("(")[0]
        id_to_parent[cid] = parent_val

    heading_map = {cid: id_to_name.get(id_to_parent.get(cid)) if id_to_parent.get(cid) else None
                   for cid in id_to_name}

    with_parent = sum(1 for v in id_to_parent.values() if v)
    print(f"  {with_parent}/{len(id_to_name)} codes have a parent heading "
          f"(the rest fall back to the old ABS file, or blank).")

    return {"heading": heading_map, "order": id_to_order}


def build_data_key(dims, region_dim, region_type_dim, sa2_key, region_codes_batch):
    key_parts = [""] * len(dims)
    for d in dims:
        idx = d["position"]
        if region_dim and d["id"] == region_dim["id"]:
            key_parts[idx] = "+".join(region_codes_batch)
        elif region_type_dim and d["id"] == region_type_dim["id"] and sa2_key:
            key_parts[idx] = sa2_key
    return ".".join(key_parts)


def fetch_sa2_data(dims, region_dim, region_type_dim, sa2_key, nsw_codes):
    print(f"\n=== Stage 3: fetching data for {len(nsw_codes)} NSW SA2 codes ===")
    if region_dim is None:
        print("  ! No region dimension confirmed, aborting fetch.")
        return None

    all_frames = []
    batches = [nsw_codes[i:i + CODES_PER_BATCH] for i in range(0, len(nsw_codes), CODES_PER_BATCH)]
    for b_idx, batch in enumerate(batches, start=1):
        data_key = build_data_key(dims, region_dim, region_type_dim, sa2_key, batch)
        url = f"{API_BASE}/data/{AGENCY},{DATAFLOW_ID}/{data_key}"
        print(f"  Batch {b_idx}/{len(batches)} ({len(batch)} codes)...")
        try:
            resp = get_with_retry(url, headers=DATA_HEADERS, params={"format": "csvfilewithlabels"})
        except Exception as e:
            print(f"    ! Batch {b_idx} failed after retries: {e}")
            continue

        if not resp.text.strip():
            print(f"    ! Batch {b_idx} returned an empty response.")
            continue

        try:
            df = pd.read_csv(StringIO(resp.text))
        except Exception as e:
            print(f"    ! Could not parse batch {b_idx} as CSV: {e}")
            debug_path = f"batch_{b_idx}_raw_response.txt"
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(resp.text[:5000])
            print(f"      First 5000 chars saved to '{debug_path}'.")
            continue

        all_frames.append(df)
        print(f"    + {len(df)} rows received.")

    if not all_frames:
        print("  ! No data retrieved in any batch.")
        return None

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"  Total rows: {len(combined)}")
    print(f"  Columns: {list(combined.columns)}")
    return combined


def load_old_heading_fallback():
    try:
        raw = pd.read_excel(OLD_ABS_FALLBACK_FILE, sheet_name=OLD_ABS_FALLBACK_SHEET, header=None)
    except FileNotFoundError:
        print(f"  (No '{OLD_ABS_FALLBACK_FILE}' found -- indicators with no "
              f"heading from the API will be left blank.)")
        return {}
    except Exception as e:
        print(f"  ! Could not read '{OLD_ABS_FALLBACK_FILE}': {e}")
        return {}

    heading_row = raw.iloc[0]
    label_row = raw.iloc[1]

    fallback = {}
    last_heading = None
    for heading_val, label_val in zip(heading_row, label_row):
        if pd.notna(heading_val):
            last_heading = str(heading_val).strip()
        if pd.isna(label_val):
            continue
        key = normalize_label(label_val)
        if key and key not in fallback:
            fallback[key] = last_heading

    print(f"  Loaded {len(fallback)} label->heading fallback entries from '{OLD_ABS_FALLBACK_FILE}'.")
    return fallback


def build_wide_table(long_df, region_dim, measure_hierarchy):
    print("\n=== Stage 4: pivoting on MEASURE code ===")
    if long_df is None or long_df.empty:
        print("  ! Nothing to pivot.")
        return None, None

    cols_lower = {c.lower(): c for c in long_df.columns}
    region_col = cols_lower.get(region_dim["id"].lower()) if region_dim else None
    if region_col is None:
        region_col = next((c for c in long_df.columns if "region" in c.lower()), None)
    value_col = cols_lower.get("obs_value") or next((c for c in long_df.columns if "value" in c.lower()), None)
    time_col = cols_lower.get("time_period") or next((c for c in long_df.columns if "time" in c.lower()), None)
    measure_code_col = cols_lower.get("measure")
    measure_label_col = cols_lower.get("data item") or cols_lower.get("data_item")

    print(f"  region_col='{region_col}', measure_code_col='{measure_code_col}', "
          f"measure_label_col='{measure_label_col}', time_col='{time_col}', value_col='{value_col}'")

    if not all([region_col, value_col, measure_code_col]):
        print("  ! Could not identify required columns.")
        return None, None

    df = long_df.copy()
    df[region_col] = df[region_col].astype(str).str.strip()

    if time_col:
        df = df.sort_values(time_col).drop_duplicates(subset=[region_col, measure_code_col], keep="last")

    wide = df.pivot_table(index=region_col, columns=measure_code_col, values=value_col, aggfunc="first")
    wide = wide.reset_index().rename(columns={region_col: "ABS_Code"})

    code_to_label = {}
    if measure_label_col:
        pairs = df[[measure_code_col, measure_label_col]].drop_duplicates()
        code_to_label = dict(zip(pairs[measure_code_col], pairs[measure_label_col]))

    heading_lookup = measure_hierarchy.get("heading", {})
    order_lookup = measure_hierarchy.get("order", {})
    heading_fallback = load_old_heading_fallback()

    data_codes = [c for c in wide.columns if c != "ABS_Code"]
    column_meta = []
    no_heading = 0
    via_fallback = 0
    for code in data_codes:
        label = code_to_label.get(code, str(code))
        heading = heading_lookup.get(code)
        if not heading:
            fallback_heading = heading_fallback.get(normalize_label(label))
            if fallback_heading:
                heading = fallback_heading
                via_fallback += 1
            else:
                heading = ""
                no_heading += 1
        column_meta.append({"code": code, "heading": heading, "label": label})

    print(f"  Headings: {len(column_meta) - no_heading} resolved ({via_fallback} via fallback), "
          f"{no_heading} left blank.")

    column_meta.sort(key=lambda m: (order_lookup.get(m["code"], float("inf")), m["heading"], m["label"]))
    ordered_codes = [m["code"] for m in column_meta]
    wide = wide[["ABS_Code"] + ordered_codes]

    print(f"  Wide table: {wide.shape[0]} rows x {len(column_meta)} indicator columns")
    return wide, column_meta


def write_indicators_excel(wide, column_meta, path):
    print(f"\n=== Stage 5: writing '{path}' ===")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Indicators"

    ws.cell(row=1, column=1, value="ABS_Code")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)

    start_col = 2
    run_start = start_col
    prev_heading = None
    for i, meta in enumerate(column_meta):
        c = start_col + i
        ws.cell(row=1, column=c, value=meta["heading"])
        ws.cell(row=2, column=c, value=meta["label"])
        if meta["heading"] != prev_heading:
            if prev_heading is not None and run_start < c:
                ws.merge_cells(start_row=1, start_column=run_start, end_row=1, end_column=c - 1)
            run_start = c
            prev_heading = meta["heading"]
    last_col = start_col + len(column_meta) - 1
    if column_meta and run_start <= last_col:
        ws.merge_cells(start_row=1, start_column=run_start, end_row=1, end_column=last_col)

    for r, row in enumerate(wide.itertuples(index=False), start=3):
        for c_idx, val in enumerate(row, start=1):
            ws.cell(row=r, column=c_idx, value=val)

    ws.freeze_panes = "B3"

    ref_ws = wb.create_sheet("Reference")
    ref_ws.append(["MEASURE_code", "Heading", "Label"])
    for meta in column_meta:
        ref_ws.append([meta["code"], meta["heading"], meta["label"]])

    wb.save(path)
    print(f"  Saved: {wide.shape[0]} rows, {len(column_meta)} columns, + Reference sheet.")


def flatten_columns(wide, column_meta):
    seen = {}
    flat_names = []
    for meta in column_meta:
        name = f"{meta['heading']} - {meta['label']}" if meta["heading"] and meta["heading"] != meta["label"] else meta["label"]
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        flat_names.append(name)

    ordered_codes = [m["code"] for m in column_meta]
    flat = wide[["ABS_Code"] + ordered_codes].copy()
    flat.columns = ["ABS_Code"] + flat_names
    return flat


def merge_with_latlon(flat_wide):
    print(f"\n=== Stage 6: merging with '{LATLON_FILE}' ===")
    try:
        latlon = pd.read_csv(LATLON_FILE)
    except FileNotFoundError:
        print(f"  ! '{LATLON_FILE}' not found -- skipping merge, indicators-only file is still saved.")
        return None

    latlon["ABS_Code"] = latlon["ABS_Code"].astype(str).str.strip()
    flat_wide = flat_wide.copy()
    flat_wide["ABS_Code"] = flat_wide["ABS_Code"].astype(str).str.strip()

    final = pd.merge(latlon, flat_wide, on="ABS_Code", how="left")
    print(f"  Merged shape: {final.shape[0]} rows x {final.shape[1]} columns")
    return final


def main():
    nsw_codes = load_nsw_codes()

    structure_json = discover_structure()
    dims, codelists = find_dimensions(structure_json)
    region_dim, region_type_dim, sa2_key = find_region_dimension(dims, codelists, nsw_codes)
    measure_hierarchy = build_measure_hierarchy(dims)

    long_df = fetch_sa2_data(dims, region_dim, region_type_dim, sa2_key, nsw_codes)
    wide, column_meta = build_wide_table(long_df, region_dim, measure_hierarchy)

    if wide is None:
        print(f"\nCould not build a combined table -- see diagnostics above "
              f"and the cached structure in '{STRUCTURE_CACHE_FILE}'.")
        return

    write_indicators_excel(wide, column_meta, INDICATORS_OUTPUT)

    flat_wide = flatten_columns(wide, column_meta)
    final = merge_with_latlon(flat_wide)
    if final is not None:
        final.to_csv(FINAL_OUTPUT, index=False)
        print(f"\nSuccess! Final merged file saved to '{FINAL_OUTPUT}'")
        print(f"Final shape: {final.shape[0]} rows x {final.shape[1]} columns")


if __name__ == "__main__":
    main()