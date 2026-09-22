"""
bushfire_risk.py

Queries the NSW Bush Fire Prone Land (BFPL) layer, hosted live by NSW
Spatial Services, for each candidate vertiport location, and saves the
result to vertiport_bushfire_risk.csv.

This is a point-in-polygon lookup done via the ArcGIS REST API rather
than a local shapefile join, because the full statewide BFPL shapefile
is too large to work with directly. Each vertiport's lat/lon is sent to
the service, which returns whichever BFPL polygon (if any) it falls
inside.

RUN THIS LOCALLY (not in a sandboxed/offline environment) -- it needs
outbound internet access to spatial.industry.nsw.gov.au.

Requires: pandas, requests
    pip install pandas requests openpyxl
"""

import pandas as pd
import requests
import time

# =========================================================
# CONFIG
# =========================================================

VERTIPORT_FILE = 'vertiport_name_with_coord.xlsx'
OUTPUT_FILE    = 'vertiport_bushfire_risk.csv'

BFPL_QUERY_URL = ('https://portal.data.nsw.gov.au/arcgis/rest/services/'
                   'Bush_Fire_Prone_Land_MIL1/MapServer/0/query')

REQUEST_DELAY_SEC = 0.5   # be polite to the public service between requests
REQUEST_TIMEOUT_SEC = 20  # fail reasonably fast rather than hang for minutes
MAX_RETRIES = 2

# =========================================================
# HTTP SESSION
# =========================================================
# Government ArcGIS servers often stall/timeout requests that look like
# they're coming from a bare script (no browser-like headers), rather
# than returning a clean error. A realistic User-Agent plus a retry
# policy fixes most of these silent timeouts.

session = requests.Session()
session.headers.update({
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')
})

retry_strategy = requests.adapters.Retry(
    total=MAX_RETRIES,
    backoff_factor=1,          # 1s, 2s between retries
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=['GET'],
)
adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
session.mount('https://', adapter)
session.mount('http://', adapter)

# =========================================================
# LOAD VERTIPORT LIST
# =========================================================

dfi = pd.read_excel(VERTIPORT_FILE)
dfi = dfi.dropna(subset=['name', 'latitude', 'longitude']).reset_index(drop=True)

# =========================================================
# QUERY THE BFPL LAYER FOR EACH VERTIPORT
# =========================================================

def query_bfpl(lat, lon):
    """
    Returns a list of matching BFPL polygon attribute dicts for a given
    point. Usually 0 or 1 matches; occasionally more if vegetation
    category and buffer polygons overlap at that point.
    """
    params = {
        'geometry': f'{lon},{lat}',
        'geometryType': 'esriGeometryPoint',
        'inSR': 4326,
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'Category,d_Category,Guideline,d_Guidelin,Area,StartDate,EndDate',
        'returnGeometry': 'false',
        'f': 'json',
    }
    resp = session.get(BFPL_QUERY_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    if 'error' in data:
        raise requests.RequestException(data['error'].get('message', 'Unknown ArcGIS error'))
    return data.get('features', [])

rows = []

for idx, row in dfi.iterrows():
    name = row['name']
    lat  = row['latitude']
    lon  = row['longitude']

    try:
        features = query_bfpl(lat, lon)
    except requests.RequestException as e:
        print(f"[{idx+1}/{len(dfi)}] {name}: request failed ({e}) -- recorded as unknown")
        rows.append({
            'name': name, 'latitude': lat, 'longitude': lon,
            'bfpl_category': None, 'bfpl_category_label': 'query_failed',
            'bfpl_area': None, 'n_matches': None,
        })
        time.sleep(REQUEST_DELAY_SEC)
        continue

    if not features:
        # No BFPL polygon at this point -- not mapped as bush fire prone
        rows.append({
            'name': name, 'latitude': lat, 'longitude': lon,
            'bfpl_category': 0, 'bfpl_category_label': 'Not bush fire prone',
            'bfpl_area': None, 'n_matches': 0,
        })
    else:
        # If multiple overlapping polygons match, keep the highest-risk
        # (lowest Category number, since 1 = highest risk vegetation)
        attrs = [f['attributes'] for f in features]
        best = min(attrs, key=lambda a: a.get('Category', 99) if a.get('Category') is not None else 99)
        rows.append({
            'name': name, 'latitude': lat, 'longitude': lon,
            'bfpl_category': best.get('Category'),
            'bfpl_category_label': best.get('d_Category'),
            'bfpl_area': best.get('Area'),
            'n_matches': len(features),
        })

    if (idx + 1) % 25 == 0 or (idx + 1) == len(dfi):
        print(f"[{idx+1}/{len(dfi)}] processed")

    time.sleep(REQUEST_DELAY_SEC)

# =========================================================
# SAVE RESULTS
# =========================================================

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved {len(result_df)} rows to {OUTPUT_FILE}")
print("\nCategory breakdown:")
print(result_df['bfpl_category_label'].value_counts(dropna=False))
