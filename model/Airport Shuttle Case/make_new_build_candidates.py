"""
NEW-BUILD airport-shuttle vertiport candidates at town centres on the
Western Sydney International (WSI) <-> Sydney Kingsford Smith (SYD) corridors.

Site types (one or more per centre, where one exists within ~1 km of the station):
  station car park        : commuter / council car park next to the station (rooftop or deck)
  shopping centre rooftop : rooftop car park of the main shopping centre
  park / oval             : open ground-level park or sports field near the station

All coordinates are from NSW Spatial Services, NSW Points of Interest
(https://maps1.six.nsw.gov.au/arcgis/rest/services/public/NSW_POI/MapServer/0).
A POI point is the feature's label point, not a surveyed pad position.

cost_index is a rough relative build cost (assumption, not a quote):
  1.0 park / oval (ground-level pad), 1.5 station car park, 2.0 shopping centre rooftop.

Outputs:
  vertiport_shuttle_new_build_candidates.xlsx
  vertiport_shuttle_new_build_candidates.csv   (for Unity)
Unity columns: unity_x = metres east, unity_z = metres north of UNITY_ORIGIN.
"""

import numpy as np
import pandas as pd

UNITY_ORIGIN = 'Western Sydney International (Nancy Bird Walton) Airport'
COST_INDEX = {'park / oval': 1.0, 'station car park': 1.5, 'shopping centre rooftop': 2.0}

AIRPORTS = [
    # name, lon, lat  (same coordinates as vertiport_name_with_coord.xlsx)
    ('Western Sydney International (Nancy Bird Walton) Airport', 150.739722, -33.879444),
    ('Sydney Kingsford Smith International Airport',             151.177002, -33.946098),
]

# centre: (route, station name, station lon, station lat)
CENTRES = {
    'Cabramatta':   ('green (via Cabramatta, Bankstown)', 'Cabramatta Station',   150.93880753553788, -33.894921019906988),
    'Bankstown':    ('green (via Cabramatta, Bankstown)', 'Bankstown Station',    151.0343178529516,  -33.917810246634069),
    'Liverpool':    ('red (via Liverpool, Revesby, Hurstville)', 'Liverpool Station', 150.9272947915353, -33.924795006661867),
    'Revesby':      ('red (via Liverpool, Revesby, Hurstville)', 'Revesby Station',   151.01514179326259, -33.95250810915612),
    'Hurstville':   ('red (via Liverpool, Revesby, Hurstville)', 'Hurstville Station', 151.10238681954499, -33.967349393944801),
    'Parramatta':   ('other', 'Parramatta Station',   151.0052653251602,  -33.817557356584317),
    'Campbelltown': ('other', 'Campbelltown Station', 150.81435829128358, -34.063710312385027),
}

# centre, site name, site_type, lon, lat, NSW POI name/type
SITES = [
    ('Cabramatta', 'Cabramatta Station Car Park',             'station car park',        150.93762361797715, -33.894658572053473, 'Parking Area (unnamed)'),
    ('Cabramatta', 'Cabravale Memorial Park',                 'park / oval',             150.93848066375884, -33.891023803819962, 'CABRAVALE MEMORIAL PARK'),
    ('Cabramatta', 'Irelands Bridge Reserve',                 'park / oval',             150.94542797185809, -33.903455051082176, 'IRELANDS BRIDGE RESERVE (Sports Field)'),

    ('Bankstown',  'Bankstown Station North Car Park',        'station car park',        151.03429728961646, -33.912288076033313, 'Parking Area (unnamed)'),
    ('Bankstown',  'Bankstown Central Rooftop',               'shopping centre rooftop', 151.03841783397689, -33.915894159968772, 'BANKSTOWN SQUARE SHOPPING CENTRE'),
    ('Bankstown',  'Bankstown Oval',                          'park / oval',             151.032161683369,   -33.922012847331743, 'BANKSTOWN OVAL (Sports Field)'),

    ('Liverpool',  'Telowra Car Park',                        'station car park',        150.92493611743316, -33.926882295883097, 'TELOWRA CAR PARK'),
    ('Liverpool',  'Westfield Liverpool Rooftop',             'shopping centre rooftop', 150.9243234744942,  -33.919371798380425, 'WESTFIELD SHOPPING COMPLEX'),
    ('Liverpool',  'Bigge Park',                              'park / oval',             150.92820575599697, -33.921689041411526, 'BIGGE PARK'),
    ('Liverpool',  'Hillier Oval',                            'park / oval',             150.9168538975668,  -33.926492972365843, 'HILLIER OVAL (Sports Field)'),

    ('Revesby',    'Revesby Station Car Park',                'station car park',        151.01485472224053, -33.953293353983007, 'Parking Area (unnamed)'),
    ('Revesby',    'Johnstone Reserve',                       'park / oval',             151.00977698611391, -33.948163491803918, 'JOHNSTONE RESERVE (Sports Field)'),

    ('Hurstville', 'Westfield Hurstville Rooftop',            'shopping centre rooftop', 151.10412255062514, -33.966425530283075, 'WESTFIELD SHOPPING CENTRE'),
    ('Hurstville', 'Hurstville Park Sports Field',            'park / oval',             151.09938535594128, -33.961208198072597, 'Sports Field in HURSTVILLE PARK'),
    ('Hurstville', 'Kempt Field',                             'park / oval',             151.11265567625509, -33.968173793203114, 'KEMPT FIELD'),

    ('Parramatta', 'Parramatta Station Car Park',             'station car park',        151.00547885649985, -33.816291741296766, 'Parking Area (unnamed)'),
    ('Parramatta', 'Westfield Parramatta Rooftop',            'shopping centre rooftop', 151.00209853920481, -33.817683954698197, 'WESTFIELD PARRAMATTA'),
    ('Parramatta', 'Robin Thomas Reserve',                    'park / oval',             151.01264223278119, -33.817354644870342, 'ROBIN THOMAS RESERVE (Sports Field)'),
    ('Parramatta', 'Old Kings Oval (Parramatta Park)',        'park / oval',             150.99763340199971, -33.808857213287276, 'OLD KINGS OVAL (Sports Field)'),

    ('Campbelltown', 'Campbelltown Mall Rooftop',             'shopping centre rooftop', 150.81020584415407, -34.068804496617325, 'CAMPBELLTOWN MALL'),
    ('Campbelltown', 'Mawson Park',                           'park / oval',             150.81603466348636, -34.065803105847195, 'MAWSON PARK'),
    ('Campbelltown', 'Campbelltown Showground',               'park / oval',             150.8209748648481,  -34.064294198458114, 'CAMPBELLTOWN SHOWGROUND'),
]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


rows = []
for name, lon, lat in AIRPORTS:
    rows.append({'name': name, 'longitude': lon, 'latitude': lat, 'role': 'airport',
                 'centre': '', 'route': 'endpoint', 'site_type': 'passenger airport',
                 'new_build': False, 'cost_index': np.nan, 'dist_station_m': np.nan,
                 'source': 'vertiport_name_with_coord.xlsx'})
for centre, name, stype, lon, lat, poi in SITES:
    route, station, s_lon, s_lat = CENTRES[centre]
    rows.append({'name': name, 'longitude': lon, 'latitude': lat, 'role': 'feeder',
                 'centre': centre, 'route': route, 'site_type': stype,
                 'new_build': True, 'cost_index': COST_INDEX[stype],
                 'dist_station_m': round(haversine_km(lat, lon, s_lat, s_lon) * 1000),
                 'source': f'NSW POI: {poi}'})
df = pd.DataFrame(rows)

# distance to each airport
for name, lon, lat in AIRPORTS:
    short = 'WSI' if 'Western' in name else 'SYD'
    df[f'dist_{short}_km'] = haversine_km(df['latitude'], df['longitude'], lat, lon).round(1)

# Unity local coordinates (metres, equirectangular about the origin; error < 0.1% over this area)
o_lon, o_lat = next((lo, la) for n, lo, la in AIRPORTS if n == UNITY_ORIGIN)
R = 6371000.0
df['unity_x'] = (np.radians(df['longitude'] - o_lon) * R * np.cos(np.radians(o_lat))).round(1)
df['unity_z'] = (np.radians(df['latitude'] - o_lat) * R).round(1)

assert df['name'].is_unique
df.to_excel('vertiport_shuttle_new_build_candidates.xlsx', index=False)
df.to_csv('vertiport_shuttle_new_build_candidates.csv', index=False)

print(f"{len(df)} sites ({df['new_build'].sum()} new-build + {len(AIRPORTS)} airports)")
print(df[['centre', 'name', 'site_type', 'dist_station_m', 'dist_WSI_km', 'dist_SYD_km', 'unity_x', 'unity_z']]
      .to_string(index=False))
