"""
Adds public parks, ovals and sports grounds from OpenStreetMap to the airport-shuttle
candidate file, as new-build vertiport sites.

Run on a computer with internet access, in the same folder as
vertiport_shuttle_candidates.xlsx (made by build_shuttle_vertiports.py):

    pip install requests pandas openpyxl
    python add_osm_parks_ovals.py

Output: vertiport_shuttle_candidates_osm.xlsx
  'candidates'     : original candidates + OSM sites (site_type = 'OSM park / oval')
  'osm_raw'        : every OSM site found, before de-duplication
  'excluded'       : copied from the input file
"""

import time
import numpy as np
import pandas as pd
import requests

INPUT  = 'vertiport_shuttle_candidates.xlsx'
OUTPUT = 'vertiport_shuttle_candidates_osm.xlsx'
OVERPASS_URLS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
]
# Overpass rejects the default python-requests User-Agent with 406 Not Acceptable
HEADERS = {
    'User-Agent': 'vertiport-shuttle-study/1.0 (university research; contact: your-email@example.com)',
    'Accept': 'application/json',
}

# search radius (km) around each passenger airport
RADIUS_KM = {'major': 30, 'regional': 15}
MAJOR_AIRPORTS = {'Sydney Kingsford Smith International Airport',
                  'Western Sydney International (Nancy Bird Walton) Airport'}

MIN_AREA_M2   = 10_000   # ~1 ha: big enough for a FATO plus safety area (a cricket oval is ~1.5-2 ha)
DEDUP_M       = 300      # drop OSM sites within this distance of an existing candidate
LEISURE_TYPES = 'park|pitch|recreation_ground|sports_centre'


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def polygon_area_m2(coords):
    """Shoelace area on a local equirectangular projection (fine for park-sized shapes)."""
    lat = np.radians([c['lat'] for c in coords])
    lon = np.radians([c['lon'] for c in coords])
    x = 6371000 * lon * np.cos(lat.mean())
    y = 6371000 * lat
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def query_overpass(lat, lon, radius_km):
    r = int(radius_km * 1000)
    q = f"""
    [out:json][timeout:120];
    (
      way["leisure"~"^({LEISURE_TYPES})$"](around:{r},{lat},{lon});
      way["landuse"="recreation_ground"](around:{r},{lat},{lon});
    );
    out geom tags;
    """
    errors = []
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                resp = requests.post(url, data={'data': q}, headers=HEADERS, timeout=180)
            except requests.RequestException as e:
                errors.append(f'{url}: {e}')
                break
            if resp.status_code == 200:
                return resp.json()['elements']
            errors.append(f'{url}: HTTP {resp.status_code}')
            if resp.status_code in (429, 504):      # busy: back off and retry same server
                time.sleep(15 * (attempt + 1))
            else:                                   # 400/403/406 etc: try the next server
                break
    raise RuntimeError('All Overpass servers failed:\n  ' + '\n  '.join(errors))


def parse_elements(elements, airport_name):
    rows = []
    for el in elements:
        geom = el.get('geometry')
        if not geom or len(geom) < 4:
            continue
        tags = el.get('tags', {})
        # skip private / restricted land
        if tags.get('access') in ('private', 'no') or 'school' in tags.get('name', '').lower():
            continue
        area = polygon_area_m2(geom)
        if area < MIN_AREA_M2:
            continue
        rows.append({
            'osm_id':    f"way/{el['id']}",
            'name':      tags.get('name') or f"Unnamed {tags.get('leisure', 'recreation ground')} (OSM {el['id']})",
            'latitude':  float(np.mean([g['lat'] for g in geom])),
            'longitude': float(np.mean([g['lon'] for g in geom])),
            'area_m2':   round(area),
            'osm_type':  tags.get('leisure') or tags.get('landuse'),
            'sport':     tags.get('sport', ''),
            'search_airport': airport_name,
        })
    return rows


cand = pd.read_excel(INPUT, sheet_name='candidates')
excl = pd.read_excel(INPUT, sheet_name='excluded')
airports = cand[cand['role'] == 'airport']

osm_rows = []
for _, apt in airports.iterrows():
    radius = RADIUS_KM['major' if apt['name'] in MAJOR_AIRPORTS else 'regional']
    print(f"Querying {apt['name']} ({radius} km)...", end=' ', flush=True)
    found = parse_elements(query_overpass(apt['latitude'], apt['longitude'], radius), apt['name'])
    print(len(found))
    osm_rows += found
    time.sleep(2)

osm = pd.DataFrame(osm_rows).drop_duplicates('osm_id')

# drop OSM sites that duplicate an existing candidate (e.g. an oval that already has an HLS)
D = haversine_km(osm['latitude'].values[:, None], osm['longitude'].values[:, None],
                 cand['latitude'].values[None, :], cand['longitude'].values[None, :]) * 1000
osm_new = osm[D.min(axis=1) > DEDUP_M].copy()

# names must be unique (the model keys on name)
dup = osm_new['name'].duplicated(keep=False) | osm_new['name'].isin(cand['name'])
osm_new.loc[dup, 'name'] = osm_new.loc[dup, 'name'] + ' (OSM ' + osm_new.loc[dup, 'osm_id'].str.split('/').str[1] + ')'

# nearest passenger airport
Da = haversine_km(osm_new['latitude'].values[:, None], osm_new['longitude'].values[:, None],
                  airports['latitude'].values[None, :], airports['longitude'].values[None, :])
osm_new['nearest_airport'] = airports['name'].values[Da.argmin(axis=1)]
osm_new['dist_airport_km'] = Da.min(axis=1).round(1)
osm_new['role']      = 'feeder'
osm_new['site_type'] = 'OSM park / oval'
osm_new['access']    = 'public land (OSM), new build'
osm_new['notes']     = ('area ' + osm_new['area_m2'].astype(str) + ' m2; ' + osm_new['osm_type'].astype(str)
                        + np.where(osm_new['sport'] != '', '; sport=' + osm_new['sport'], '') + '; ' + osm_new['osm_id'])

merged = pd.concat([cand, osm_new[cand.columns]], ignore_index=True)
assert merged['name'].is_unique

with pd.ExcelWriter(OUTPUT) as xw:
    merged.to_excel(xw, sheet_name='candidates', index=False)
    osm.to_excel(xw, sheet_name='osm_raw', index=False)
    excl.to_excel(xw, sheet_name='excluded', index=False)

print(f"\nAdded {len(osm_new)} OSM sites ({len(osm) - len(osm_new)} dropped as duplicates). "
      f"Total candidates: {len(merged)}. Saved to {OUTPUT}")