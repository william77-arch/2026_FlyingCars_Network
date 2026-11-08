"""
NEW-BUILD airport-shuttle vertiport candidates for Greater Sydney, found by searching
place NAMES for keywords (parks, ovals, shopping centres, car parks).

Sources (set SOURCE below):
  'nsw'    : NSW Spatial Services, NSW Points of Interest (default; free, no key)
             https://maps1.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer/0
  'google' : Google Places API (New) Text Search. Needs your own API key in the
             GOOGLE_API_KEY environment variable; billed per request by Google, and
             Google's terms limit how the results may be stored and shown.
  'cache'  : the CSVs in sydney_poi/ (offline; only sports fields, shopping centres
             and parking areas, no plain parks)

Steps: search names -> classify by keyword -> drop excluded names -> thin out so each
category keeps at most one site per THIN_M -> add airport distances and Unity coords.

Each point is the place's label / centre point, not a surveyed pad position.
cost_index is a rough relative build cost (assumption, not a quote):
  1.0 park / oval (ground-level pad), 1.5 car park, 2.0 shopping centre rooftop.

Outputs:
  vertiport_shuttle_new_build_candidates.xlsx
  vertiport_shuttle_new_build_candidates.csv   (for Unity)
  new_build_raw_matches.csv                    (every name match before thinning)
Unity columns: unity_x = metres east, unity_z = metres north of UNITY_ORIGIN.

    pip install requests pandas numpy openpyxl
    python make_new_build_candidates.py
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = 'nsw'                       # 'nsw', 'google' or 'cache'
# search area: a box around Western Sydney International big enough for the eVTOL's range
WSI_LAT, WSI_LON = -33.879444, 150.739722
RANGE_KM = 150                       # same as RANGE_KM in shuttle_optim_model_V2.py
_dlat = RANGE_KM / 111.0
_dlon = RANGE_KM / (111.0 * np.cos(np.radians(WSI_LAT)))
BBOX   = (WSI_LON - _dlon, WSI_LAT - _dlat, WSI_LON + _dlon, WSI_LAT + _dlat)   # lon_min, lat_min, lon_max, lat_max
THIN_M = 500                         # per category, keep at most one site within this distance
SHORT_TRIP_KM = 8                    # sites closer than this to an airport are flagged

# ── keywords (matched on upper-case names, whole words) ──────────────────────
CATEGORIES = {
    # category: (site_type, keywords, priority order for thinning: earlier keyword wins)
    'park / oval': ['OVAL', 'SPORTSGROUND', 'SPORTS GROUND', 'SHOWGROUND', 'PLAYING FIELDS',
                    'PLAYING FIELD', 'SPORTS COMPLEX', 'RESERVE', 'PARK'],
    'shopping centre rooftop': ['WESTFIELD', 'SHOPPING CENTRE', 'SHOPPING TOWN', 'SHOPPING VILLAGE',
                                'TOWN CENTRE', 'MARKETPLACE', 'MARKET PLACE', 'PLAZA', 'MALL'],
    'car park': ['COMMUTER CAR PARK', 'MULTISTOREY CARPARK', 'MULTISTOREY CAR PARK', 'CAR PARK',
                 'CARPARK', 'PARKING STATION', 'PARKING'],
}
# names containing any of these are dropped (not public open space / not buildable)
EXCLUDE = ['SCHOOL', 'COLLEGE', 'UNIVERSITY', 'CHURCH', 'CEMETERY', 'CREMATORIUM', 'HOSPITAL',
           'NATIONAL PARK', 'CARAVAN PARK', 'TOURIST PARK', 'HOLIDAY PARK', 'BUSINESS PARK',
           'INDUSTRIAL PARK', 'TECHNOLOGY PARK', 'OFFICE PARK', 'RETIREMENT', 'NURSING', 'AGED CARE',
           'VILLAGE GREEN BOWLING', 'BOWLING', 'POST OFFICE', 'POLICE', 'RAILWAY STATION',
           'LIGHT RAIL', 'NATURE RESERVE', 'WILDLIFE', 'ZOO', 'GOLF', 'RACECOURSE', 'SKATE PARK',
           'DOG PARK', 'PLAYGROUND', 'MEMORIAL GARDEN', 'CAMPING', 'RIFLE', 'PISTOL', 'ARCHERY',
           'MODEL AIRCRAFT', 'EQUESTRIAN', 'PONY CLUB']
# NSW POI types that are names of areas, not sites
NSW_EXCLUDE_TYPES = ['Suburb', 'City', 'Locality', 'Town', 'Village', 'Railway Station',
                     'Bus Interchange', 'Primary School', 'High School', 'Place Of Worship',
                     'Golf Course', 'Cemetery', 'Post Office', 'Nursing Home', 'Retirement Village',
                     'Caravan Park', 'Industrial Area', 'Bay', 'Point', 'Hill', 'Waterbody']
NSW_KEEP_TYPES = ['Parking Area', 'Shopping Centre', 'Sports Field', 'Showground']
COST_INDEX = {'park / oval': 1.0, 'car park': 1.5, 'shopping centre rooftop': 2.0}

UNITY_ORIGIN = 'Western Sydney International (Nancy Bird Walton) Airport'
AIRPORTS = [
    # name, lon, lat  (same coordinates as vertiport_name_with_coord.xlsx)
    ('Western Sydney International (Nancy Bird Walton) Airport', 150.739722, -33.879444),
    ('Sydney Kingsford Smith International Airport',             151.177002, -33.946098),
]
ROUTES = {'Cabramatta': 'green', 'Bankstown': 'green',
          'Liverpool': 'red', 'Revesby': 'red', 'Hurstville': 'red'}
STATIONS_FILE = 'sydney_poi/stations.csv'   # station list used when SOURCE is not 'nsw' (Sydney only)
CENTRE_MAX_M  = 3000                        # a site belongs to a town centre only within this distance of its station

HEADERS = {'User-Agent': 'vertiport-shuttle-study/1.0 (university research)'}


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(a))


def has_word(name, kw):
    return re.search(rf'(?<![A-Z]){re.escape(kw)}(?![A-Z])', name) is not None


# ── source: NSW Points of Interest ───────────────────────────────────────────
NSW_URL = 'https://maps1.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer/0/query'


def nsw_post(params):
    import requests
    for attempt in range(4):
        r = requests.post(NSW_URL, data={**params, 'f': 'json'}, headers=HEADERS, timeout=120)
        if r.ok and 'error' not in r.json():
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'NSW POI query failed: HTTP {r.status_code} {r.text[:300]}')


def search_nsw():
    keywords = sorted({kw for kws in CATEGORIES.values() for kw in kws})
    like = ' OR '.join(f"UPPER(poiname) LIKE '%{kw}%'" for kw in keywords)
    types = ', '.join(f"'{t}'" for t in NSW_EXCLUDE_TYPES)
    # name matches, plus unnamed features whose POI type is already one we want
    typed = ', '.join(f"'{t}'" for t in NSW_KEEP_TYPES)
    common = {'where': f'(({like}) AND poitype NOT IN ({types})) OR poitype IN ({typed})',
              'geometry': ','.join(map(str, BBOX)), 'geometryType': 'esriGeometryEnvelope',
              'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects'}
    ids = sorted(nsw_post({**common, 'returnIdsOnly': 'true'}).get('objectIds') or [])
    print(f'NSW POI: {len(ids)} name matches, downloading...')
    rows = []
    for k in range(0, len(ids), 500):            # maxRecordCount is 1000; stay well under
        chunk = ids[k:k + 500]
        res = nsw_post({'objectIds': ','.join(map(str, chunk)), 'outFields': 'objectid,poiname,poitype',
                        'returnGeometry': 'true', 'outSR': 4326, 'geometryPrecision': 6})
        for f in res['features']:
            a, g = f['attributes'], f.get('geometry') or {}
            if 'x' in g:
                rows.append({'poiname': a['poiname'] or '', 'poitype': a['poitype'],
                             'longitude': g['x'], 'latitude': g['y'],
                             'source': f"NSW POI objectid {a['objectid']} ({a['poitype']})"})
        print(f'  {min(k + 500, len(ids))}/{len(ids)}')
    return pd.DataFrame(rows)


# ── source: Google Places API (New) Text Search ──────────────────────────────
GOOGLE_URL = 'https://places.googleapis.com/v1/places:searchText'
GOOGLE_QUERIES = {   # query text -> category (Google ranks by relevance, so search per tile)
    'oval': 'park / oval', 'sports ground': 'park / oval', 'park': 'park / oval',
    'shopping centre': 'shopping centre rooftop', 'car park': 'car park',
}
GOOGLE_TILE_DEG = 0.05   # ~5 km tiles; each query returns at most 60 places per tile


def search_google():
    import requests
    key = os.environ.get('GOOGLE_API_KEY')
    if not key:
        raise SystemExit('Set the GOOGLE_API_KEY environment variable to use SOURCE = "google".')
    head = {**HEADERS, 'X-Goog-Api-Key': key,
            'X-Goog-FieldMask': 'places.id,places.displayName,places.location,places.primaryType,nextPageToken'}
    rows = []
    lons = np.arange(BBOX[0], BBOX[2], GOOGLE_TILE_DEG)
    lats = np.arange(BBOX[1], BBOX[3], GOOGLE_TILE_DEG)
    n_req = 0
    for q in GOOGLE_QUERIES:
        for lo in lons:
            for la in lats:
                body = {'textQuery': q, 'pageSize': 20, 'locationRestriction': {'rectangle': {
                    'low': {'latitude': la, 'longitude': lo},
                    'high': {'latitude': la + GOOGLE_TILE_DEG, 'longitude': lo + GOOGLE_TILE_DEG}}}}
                for _ in range(3):                       # up to 3 pages of 20
                    r = requests.post(GOOGLE_URL, json=body, headers=head, timeout=60)
                    n_req += 1
                    r.raise_for_status()
                    js = r.json()
                    for p in js.get('places', []):
                        rows.append({'poiname': p['displayName']['text'].upper(),
                                     'poitype': p.get('primaryType', ''),
                                     'longitude': p['location']['longitude'],
                                     'latitude': p['location']['latitude'],
                                     'source': f"Google place {p['id']}", 'google_query': q})
                    if 'nextPageToken' not in js:
                        break
                    body['pageToken'] = js['nextPageToken']
                    time.sleep(1)
        print(f'Google "{q}": {len(rows)} places so far ({n_req} requests)')
    return pd.DataFrame(rows).drop_duplicates('source')


# ── source: cached CSVs ──────────────────────────────────────────────────────
def search_cache():
    parts = []
    for f, t in [('sports_fields.csv', 'Sports Field'), ('shopping_centres.csv', 'Shopping Centre'),
                 ('parking_areas.csv', 'Parking Area')]:
        d = pd.read_csv(f'sydney_poi/{f}', keep_default_na=False).rename(columns={'x': 'longitude', 'y': 'latitude'})
        d['poitype'] = t
        d['source'] = f'NSW POI cache ({t})'
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ── classify, filter, thin ───────────────────────────────────────────────────
def classify(name, poitype):
    """Return (site_type, matched keyword, keyword rank) or (None, None, None)."""
    if any(has_word(name, x) for x in EXCLUDE):
        return None, None, None
    # car-park words first, so 'CAR PARK' is never read as a park
    for stype in ['car park', 'shopping centre rooftop', 'park / oval']:
        for rank, kw in enumerate(CATEGORIES[stype]):
            if has_word(name, kw):
                return stype, kw, rank
    # unnamed features: fall back on the POI type
    t = (poitype or '').lower()
    if 'parking' in t:
        return 'car park', '(type: parking area)', 99
    if 'shopping' in t:
        return 'shopping centre rooftop', '(type: shopping centre)', 99
    if 'sports field' in t or 'showground' in t:
        return 'park / oval', f'(type: {t})', 99
    return None, None, None


def nice(s):
    return re.sub(r"'S\b", "'s", s.title())


raw = {'nsw': search_nsw, 'google': search_google, 'cache': search_cache}[SOURCE]()
raw = raw[(raw['longitude'].between(BBOX[0], BBOX[2])) & (raw['latitude'].between(BBOX[1], BBOX[3]))]
raw['poiname'] = raw['poiname'].fillna('').str.upper().str.strip()
raw[['site_type', 'keyword', 'rank']] = raw.apply(
    lambda r: pd.Series(classify(r['poiname'], r['poitype'])), axis=1)
raw.to_csv('new_build_raw_matches.csv', index=False)
cand = raw[raw['site_type'].notna()].copy()

# thin: per category, best keyword first, drop anything within THIN_M of a kept site
kept = []
for stype, grp in cand.groupby('site_type'):
    grp = grp.assign(named=grp['poiname'] != '').sort_values(['rank', 'named'], ascending=[True, False])
    klat, klon = [], []
    for idx, r in grp.iterrows():
        if klat and haversine_m(r['latitude'], r['longitude'], np.array(klat), np.array(klon)).min() < THIN_M:
            continue
        kept.append(idx)
        klat.append(r['latitude'])
        klon.append(r['longitude'])
cand = cand.loc[kept].copy()

# nearest station for labels, town centres and routes
def stations_nsw():
    """Railway / metro stations in the search area, live from NSW Points of Interest."""
    common = {'where': "poitype = 'Railway Station'", 'geometry': ','.join(map(str, BBOX)),
              'geometryType': 'esriGeometryEnvelope', 'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects'}
    ids = sorted(nsw_post({**common, 'returnIdsOnly': 'true'}).get('objectIds') or [])
    rows = []
    for k in range(0, len(ids), 500):
        res = nsw_post({'objectIds': ','.join(map(str, ids[k:k + 500])), 'outFields': 'poiname',
                        'returnGeometry': 'true', 'outSR': 4326, 'geometryPrecision': 6})
        rows += [{'poiname': f['attributes']['poiname'] or '', 'x': f['geometry']['x'], 'y': f['geometry']['y']}
                 for f in res['features'] if 'x' in (f.get('geometry') or {})]
    print(f'{len(rows)} stations in the search area')
    return pd.DataFrame(rows)

st = stations_nsw() if SOURCE == 'nsw' else (
    pd.read_csv(STATIONS_FILE, keep_default_na=False) if Path(STATIONS_FILE).exists() else None)
if st is not None and len(st):
    st = st[~st['poiname'].str.contains('LIGHT RAIL|SLR STOP|MONORAIL|TRAMWAY')].reset_index(drop=True)
    D = haversine_m(cand['latitude'].values[:, None], cand['longitude'].values[:, None],
                    st['y'].values[None, :], st['x'].values[None, :])
    j = D.argmin(axis=1)
    cand['station'] = st['poiname'].values[j]
    cand['centre'] = pd.Series(cand['station'].values).str.replace(
        r' (RAILWAY|INTERCHANGE) STATION$', '', regex=True).map(nice).values
    cand['station'] = cand['station'].map(nice)
    cand['dist_station_m'] = D.min(axis=1).round()
    far = cand['dist_station_m'] > CENTRE_MAX_M              # not in a town centre: keep the station for naming only
    cand.loc[far, 'centre'] = ''
else:
    cand['station'], cand['centre'], cand['dist_station_m'] = '', '', np.nan

# names
def site_name(r):
    if r['poiname']:
        base = nice(r['poiname'])
        return f'{base} Rooftop' if r['site_type'] == 'shopping centre rooftop' else base
    label = {'car park': 'Car Park', 'shopping centre rooftop': 'Shopping Centre Rooftop',
             'park / oval': 'Sports Field'}[r['site_type']]
    if r['centre']:
        return f"{r['centre']} {label}"
    return f"{label} near {r['station']}" if r['station'] else label

cand['name'] = cand.apply(site_name, axis=1)

rows = [{'name': n, 'longitude': lo, 'latitude': la, 'role': 'airport', 'site_type': 'passenger airport',
         'new_build': False, 'source': 'vertiport_name_with_coord.xlsx'} for n, lo, la in AIRPORTS]
df = pd.concat([pd.DataFrame(rows), cand.assign(role='feeder', new_build=True)], ignore_index=True)
df[['centre', 'station', 'keyword', 'poitype']] = df[['centre', 'station', 'keyword', 'poitype']].fillna('')
df['cost_index'] = df['site_type'].map(COST_INDEX)
ROUTE_M = 2000   # only tag a site with a corridor route if it is this close to that town's station
df['route'] = df['centre'].map(ROUTES).where(df['dist_station_m'] <= ROUTE_M).fillna('other')
df.loc[~df['new_build'], 'route'] = 'endpoint'

# unique names for the model / Unity: add the centre to repeated names, then number what is left
dup = df['name'].duplicated(keep=False) & df['new_build']
add_centre = dup & ~df.apply(lambda r: bool(r['centre']) and r['centre'] in r['name'], axis=1)
df.loc[add_centre, 'name'] = df.loc[add_centre, 'name'] + ' (' + df.loc[add_centre, 'centre'] + ')'
rep = df.groupby('name').cumcount()
df.loc[rep > 0, 'name'] = df.loc[rep > 0, 'name'] + ' ' + (rep[rep > 0] + 1).astype(str)
assert df['name'].is_unique

for n, lo, la in AIRPORTS:
    short = 'WSI' if 'Western' in n else 'SYD'
    df[f'dist_{short}_km'] = (haversine_m(df['latitude'], df['longitude'], la, lo) / 1000).round(1)
df['short_trip'] = df['new_build'] & (df[['dist_WSI_km', 'dist_SYD_km']].min(axis=1) < SHORT_TRIP_KM)

o_lon, o_lat = next((lo, la) for n, lo, la in AIRPORTS if n == UNITY_ORIGIN)
df['unity_x'] = (np.radians(df['longitude'] - o_lon) * 6371000.0 * np.cos(np.radians(o_lat))).round(1)
df['unity_z'] = (np.radians(df['latitude'] - o_lat) * 6371000.0).round(1)

cols = ['name', 'longitude', 'latitude', 'role', 'site_type', 'new_build', 'cost_index', 'keyword',
        'poitype', 'centre', 'station', 'route', 'dist_station_m', 'dist_WSI_km', 'dist_SYD_km',
        'short_trip', 'unity_x', 'unity_z', 'source']
df = df[cols]
df.to_excel('vertiport_shuttle_new_build_candidates.xlsx', index=False)
df.to_csv('vertiport_shuttle_new_build_candidates.csv', index=False)

nb = df[df['new_build']]
print(f'\nSource: {SOURCE}. {len(raw)} name matches -> {len(nb)} new-build sites after thinning to {THIN_M} m')
print(nb['site_type'].value_counts().to_string())
print(f'short trips (< {SHORT_TRIP_KM} km to an airport): {nb["short_trip"].sum()}')
