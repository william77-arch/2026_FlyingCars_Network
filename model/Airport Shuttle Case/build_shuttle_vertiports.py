"""
Builds an airport-shuttle candidate vertiport file from vertiport_name_with_coord.xlsx.

Two roles:
  airport : passenger airports (shuttle destinations)
  feeder  : public, non-hospital sites where passengers board
            (park / oval / showground HLS, commercial heliports, GA aerodromes)

Excluded: hospital and medical pads, rescue/emergency bases, mines, military/police/
correctional, national parks and dams, private/station/homestead strips, and feeders
beyond eVTOL range of any passenger airport.

Extra helipads are added from nsw_poi_helipads.csv (NSW Spatial Services Points of
Interest) and OurAirports. Put nsw_poi_helipads.csv (and optionally
Emergency_hospitals_with_lat_lon.xlsx) in the same folder.

Output: vertiport_shuttle_candidates.xlsx
  sheet 'candidates' : name, longitude, latitude + role, site_type, access,
                       nearest_airport, dist_airport_km, notes
  sheet 'excluded'   : every dropped site with the reason
The 'name' column is unchanged from the source file so the risk files still join.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE   = 'vertiport_name_with_coord.xlsx'
OUTPUT   = 'vertiport_shuttle_candidates.xlsx'
RANGE_KM = 150   # max feeder-to-airport distance (typical eVTOL range with reserves)
NEAR_M   = 500   # sites closer than this are flagged as possible duplicates

# Passenger airports. 'verify' = check current scheduled service before relying on it.
PASSENGER_AIRPORTS = {
    'Sydney Kingsford Smith International Airport':              'major',
    'Western Sydney International (Nancy Bird Walton) Airport':  'major',
    'Newcastle Airport':                 'regional',
    'Ballina Byron Gateway Airport':     'regional',
    'Coffs Harbour Airport':             'regional',
    'Port Macquarie Airport':            'regional',
    'Albury Airport':                    'regional',
    'Wagga Wagga Airport':               'regional',
    'Dubbo City Regional Airport':       'regional',
    'Tamworth Airport':                  'regional',
    'Armidale Airport':                  'regional',
    'Orange Airport':                    'regional',
    'Moruya Airport':                    'regional',
    'Merimbula Airport':                 'regional',
    'Lismore Airport':                   'regional',
    'Parkes Airport':                    'regional',
    'Griffith Airport':                  'regional',
    'Narrandera Airport':                'regional',
    'Broken Hill Airport':               'regional',
    'Moree Airport':                     'regional',
    'Clarence Valley Regional Airport':  'regional',
    'Lord Howe Island Airport':          'regional',
    'Bathurst Airport':                  'verify',
    'Cooma Snowy Mountains Airport':     'verify',
    'Mudgee Airport':                    'verify',
    'Narrabri Airport':                  'verify',
    'Taree Airport':                     'verify',
    'Inverell Airport':                  'verify',
    'Cobar Airport':                     'verify',
    'Bourke Airport':                    'verify',
    'Walgett Airport':                   'verify',
    'Lightning Ridge Airport':           'verify',
}

# (regex on lower-case name, reason) — first match wins
EXCLUDE_RULES = [
    (r'hospital|health campus|multi purpose service|nets\b|st vincents', 'hospital pad'),
    (r'careflight|westpac base',                          'rescue helicopter base'),
    (r'\bses\b|rural fire|fire station|emergency services|emergency airport', 'emergency services only'),
    (r'police|correctional|raaf|military|naval|barracks|training area', 'police / military / correctional'),
    (r'\bmine\b|colliery|coal|bricks',                      'mine / industrial private'),
    (r'npws|\bnp\b|national park|\bdam\b|scout',          'national park / catchment / scout camp'),
    (r'seaplane',                                         'seaplane base (water)'),
    (r'helibase',                                         'rescue helicopter base'),
    (r'beach|long reef',                                  'surf rescue pad'),
    (r'private|station|homestead|hsd\b|lodge|retreat|wilderness|sanctuary|paddock|farm|house\b|woolshed', 'private / station property'),
    (r'airstrip|airfield|airstrp|strip\b|airpirt',        'unregistered strip, access unknown'),
]
MEDICAL = r'medical'
PUBLIC_LAND = r'oval|park\b|showground|sports|soccer|rugby|ground|reserve'

# names where the keyword rules get it wrong
OVERRIDES = {
    'Coomba Park Helicopter Landing Site': ('feeder', 'helipad / HLS', 'unconfirmed', None),  # town name, not a park
    'Eastern Creek Raceway HLS':           ('feeder', 'helipad / HLS', 'private venue', None),
    'Wolgan Valley Heliport':              ('feeder', 'heliport', 'private resort', None),
}
HELI = r'helipad|heliport|helicopter|hls\b'


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def classify(name):
    """Return (role, site_type, access, exclude_reason)."""
    n = name.lower()
    if name in OVERRIDES:
        return OVERRIDES[name]
    if name in PASSENGER_AIRPORTS:
        return 'airport', 'passenger airport', 'public', None
    for pattern, reason in EXCLUDE_RULES:
        if re.search(pattern, n):
            return None, None, None, reason
    is_heli = re.search(HELI, n) is not None
    if re.search(MEDICAL, n):
        if re.search(PUBLIC_LAND, n):
            return 'feeder', 'park / oval HLS', 'public land, designated medical HLS', None
        return None, None, None, 'medical centre pad'
    if is_heli and re.search(PUBLIC_LAND, n):
        return 'feeder', 'park / oval HLS', 'public land', None
    if re.search(r'heliport', n):
        return 'feeder', 'heliport', 'commercial / unconfirmed', None
    if is_heli:
        return 'feeder', 'helipad / HLS', 'unconfirmed', None
    if re.search(r'airport|aerodrome|aero park', n):
        return 'feeder', 'GA aerodrome', 'check ownership (council or private ALA)', None
    return None, None, None, 'other (gliding, paragliding, fuel depot, unnamed)'


df = pd.read_excel(SOURCE)
df['name'] = df['name'].astype(str)
df[['role', 'site_type', 'access', 'reason']] = df['name'].apply(lambda s: pd.Series(classify(s.strip())))
df['source'] = SOURCE
df['notes'] = ''

# ─────────────────────────────────────────────────────────────
# EXTRA HELIPADS from public databases
#   nsw_poi_helipads.csv : NSW Spatial Services, NSW Points of Interest (poitype = 'Helipad')
#                          https://maps1.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer/0
#   EXTRA_SITES          : OurAirports NSW heliports not already in the source file
# Most POI helipads are unnamed, so they are screened by what they are near.
# ─────────────────────────────────────────────────────────────
POI_FILE      = 'nsw_poi_helipads.csv'
HOSPITAL_FILE = 'Emergency_hospitals_with_lat_lon.xlsx'   # optional: screens POI pads at hospitals
CLUSTER_M  = 150    # POI pads closer than this are one site (e.g. several pads at one base)
MATCH_M    = 300    # POI pad this close to a source site = same site
AIRPORT_M  = 1500   # POI pad this close to a source airport is part of that airport
HOSPITAL_M = 400    # POI pad this close to a hospital = hospital pad

EXTRA_SITES = [
    {'name': 'Eskbank Heliport', 'longitude': 150.164732, 'latitude': -33.476587,
     'source': 'OurAirports AU-0786', 'notes': 'Lithgow'},
]

extra = []
if Path(POI_FILE).exists():
    poi = pd.read_csv(POI_FILE)
    kept = []
    for i, r in poi.iterrows():
        if kept:
            d = haversine_km(r['latitude'], r['longitude'],
                             poi.loc[kept, 'latitude'].values, poi.loc[kept, 'longitude'].values) * 1000
            if d.min() < CLUSTER_M:
                continue
        kept.append(i)
    poi = poi.loc[kept]
    for _, r in poi.iterrows():
        nm = str(r['poiname']).title() if pd.notna(r['poiname']) else ''
        extra.append({
            'name': f'{nm} (NSW POI {r["objectid"]})' if nm else f'NSW POI Helipad {r["objectid"]}',
            'poiname': nm,
            'longitude': r['longitude'], 'latitude': r['latitude'],
            'source': f'NSW POI objectid {r["objectid"]}',
            'notes': f'NSW POI urbanity code {r["urbanity"]}',
        })
else:
    print(f'{POI_FILE} not found: skipping NSW POI helipads')
extra += [dict(s, poiname=s['name']) for s in EXTRA_SITES]
extra = pd.DataFrame(extra)

hospitals = (pd.read_excel(HOSPITAL_FILE)[['Name', 'latitude', 'longitude']]
             if Path(HOSPITAL_FILE).exists() else None)
if hospitals is None:
    print(f'{HOSPITAL_FILE} not found: POI pads are not screened against hospitals')

src_is_airfield = df['name'].str.lower().str.contains('airport|aerodrome|air station')
for i, r in extra.iterrows():
    d_src = haversine_km(r['latitude'], r['longitude'], df['latitude'].values, df['longitude'].values) * 1000
    j = d_src.argmin()
    d_apt = np.where(src_is_airfield, d_src, np.inf)
    reason, role, stype, access = None, 'feeder', 'NSW POI helipad', 'unknown (unnamed in NSW POI)'
    if r['poiname']:
        role, stype, access, reason = classify(r['poiname'])
        if role == 'feeder' and stype == 'helipad / HLS':
            stype = 'NSW POI helipad'
    if reason is None and d_src[j] < MATCH_M:
        reason = f'already in source file as {df["name"].iat[j]}'
    if reason is None and d_apt.min() < AIRPORT_M:
        reason = f'inside existing airport: {df["name"].iat[d_apt.argmin()]}'
    if reason is None and hospitals is not None:
        d_h = haversine_km(r['latitude'], r['longitude'], hospitals['latitude'].values, hospitals['longitude'].values) * 1000
        if d_h.min() < HOSPITAL_M:
            reason = f'hospital pad ({hospitals["Name"].iat[d_h.argmin()]})'
    if reason is None and df['reason'].iat[j] and d_src[j] < 1000:
        reason = f'within 1 km of excluded site {df["name"].iat[j]} ({df["reason"].iat[j]})'
    if reason:
        role = stype = access = None
    extra.loc[i, ['role', 'site_type', 'access', 'reason']] = [role, stype, access, reason]
    extra.loc[i, 'notes'] += f'; nearest known site: {df["name"].iat[j]} ({d_src[j] / 1000:.1f} km)'

# rough ACT box (Queanbeyan excluded): NSW POI includes some pads across the border
in_act = (extra['longitude'].between(148.76, 149.21) & extra['latitude'].between(-35.92, -35.12))
extra.loc[in_act, 'notes'] += '; possibly in ACT, check'

df = pd.concat([df, extra.drop(columns='poiname')], ignore_index=True)
df['notes'] = df['notes'].fillna('')

# airport tier / verification note
for nm, tier in PASSENGER_AIRPORTS.items():
    df.loc[df['name'] == nm, 'notes'] = ('verify current scheduled service' if tier == 'verify' else tier)

# distance from every site to its nearest passenger airport
apt = df[df['role'] == 'airport']
D = haversine_km(df['latitude'].values[:, None], df['longitude'].values[:, None],
                 apt['latitude'].values[None, :], apt['longitude'].values[None, :])
df['nearest_airport'] = apt['name'].values[D.argmin(axis=1)]
df['dist_airport_km'] = D.min(axis=1).round(1)
apt_rows = df['role'] == 'airport'
df.loc[apt_rows, ['nearest_airport', 'dist_airport_km']] = ['(self)', 0.0]

# feeders beyond range
out_of_range = (df['role'] == 'feeder') & (df['dist_airport_km'] > RANGE_KM)
df.loc[out_of_range, 'reason'] = f'more than {RANGE_KM} km from any passenger airport'
df.loc[out_of_range, ['role', 'site_type', 'access']] = None

# ─────────────────────────────────────────────────────────────
# INCLUDE RESTRICTED SITES
# With INCLUDE_RESTRICTED = True every dropped site comes back as a candidate,
# flagged restricted = True with the reason in 'access'. Only true duplicates
# (same pad already in the source file / inside an airport) stay excluded.
# Set to False for the strict public-only list.
# ─────────────────────────────────────────────────────────────
INCLUDE_RESTRICTED = False

def restricted_type(reason):
    r = reason.lower()
    if 'hospital' in r or 'medical' in r or 'rescue helicopter' in r:
        return 'hospital / medical helipad'
    if r.startswith('more than'):
        return f'out of range (>{RANGE_KM} km)'
    if 'unregistered strip' in r or 'private / station' in r:
        return 'private / station strip'
    return 'restricted site'

df['restricted'] = False
if INCLUDE_RESTRICTED:
    is_dup = df['reason'].fillna('').str.match(r'already in source file|inside existing airport')
    back = df['role'].isna() & ~is_dup
    df.loc[back, 'role'] = 'feeder'
    df.loc[back, 'site_type'] = df.loc[back, 'reason'].apply(restricted_type)
    df.loc[back, 'access'] = 'restricted: ' + df.loc[back, 'reason']
    df.loc[back, 'restricted'] = True

# data-quality flags on kept sites
keep = df['role'].notna()
dup_names = df['name'].str.strip().str.lower().duplicated(keep=False) & (df['source'] == SOURCE)
df.loc[keep & dup_names, 'notes'] += '; duplicate name in source'

hosp = df[df['reason'].isin(['hospital pad', 'medical centre pad', 'rescue helicopter base'])]
for idx in df.index[keep]:
    r = df.loc[idx]
    d_all = haversine_km(r['latitude'], r['longitude'], df['latitude'].values, df['longitude'].values) * 1000
    d_all[df.index.get_loc(idx)] = np.inf
    close = df.loc[d_all < NEAR_M]
    for _, c in close.iterrows():
        tag = 'near hospital pad' if c.name in hosp.index else 'within 500 m of'
        df.loc[idx, 'notes'] += f'; {tag}: {c["name"]}'
    if re.search(r'\.\d{0,2}$', f"{r['longitude']}") or len(f"{r['latitude']}".split('.')[-1]) <= 2:
        df.loc[idx, 'notes'] += '; low-precision coordinates'
df['notes'] = df['notes'].str.strip('; ')

# write
cols = ['name', 'longitude', 'latitude', 'role', 'site_type', 'access', 'restricted',
        'nearest_airport', 'dist_airport_km', 'source', 'notes']
cand = df[keep].sort_values(['role', 'site_type', 'name'])[cols].copy()
# the model keys on name: suffix repeats (e.g. the two 'Wilcannia airstrip' rows)
rep = cand.groupby('name').cumcount()
cand.loc[rep > 0, 'notes'] = (cand.loc[rep > 0, 'notes'] + '; renamed: duplicate name in source').str.strip('; ')
cand.loc[rep > 0, 'name'] = cand.loc[rep > 0, 'name'] + ' (' + (rep[rep > 0] + 1).astype(str) + ')'
assert cand['name'].is_unique
excl = df[~keep][['name', 'longitude', 'latitude', 'source', 'reason']].sort_values(['reason', 'name'])

with pd.ExcelWriter(OUTPUT) as xw:
    cand.to_excel(xw, sheet_name='candidates', index=False)
    excl.to_excel(xw, sheet_name='excluded', index=False)

print(f'{len(cand)} candidates written to {OUTPUT}  ({len(excl)} excluded)')
print(cand.groupby(['restricted', 'role', 'site_type']).size().to_string())
print('\nExcluded by reason:')
print(excl['reason'].value_counts().to_string())