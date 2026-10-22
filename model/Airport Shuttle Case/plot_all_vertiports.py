"""
Plots every airport-shuttle candidate vertiport on a folium map,
using the same base map, measure tool and legend style as parametric_optim_model.py.

Reads vertiport_shuttle_existing_candidates.xlsx (build_shuttle_vertiports_existing_sites.py)
and, if present, vertiport_shuttle_new_build_candidates.xlsx (make_new_build_candidates.py).
Sites are coloured by site_type; new-build sites are drawn larger with a black outline.
"""

import pandas as pd
import folium
from folium.plugins import MeasureControl

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
from pathlib import Path

SOURCE = 'vertiport_shuttle_existing_candidates.xlsx'
NEW_BUILD = 'vertiport_shuttle_new_build_candidates.xlsx'
dfi = pd.read_excel(SOURCE, sheet_name='candidates')
dfi['new_build'] = False

if Path(NEW_BUILD).exists():
    nb = pd.read_excel(NEW_BUILD)
    nb = nb[nb['new_build']].copy()          # airports are already in the existing file
    near_wsi = nb['dist_WSI_km'] <= nb['dist_SYD_km']
    nb['nearest_airport'] = near_wsi.map({True: 'Western Sydney International', False: 'Sydney Kingsford Smith'})
    nb['dist_airport_km'] = nb[['dist_WSI_km', 'dist_SYD_km']].min(axis=1)
    nb['access'] = 'new build'
    nb['notes'] = (nb['centre'] + ', ' + nb['route'] + ' route; cost index ' + nb['cost_index'].astype(str)
                   + '; ' + nb['dist_station_m'].astype(int).astype(str) + ' m from station')
    dfi = pd.concat([dfi, nb[dfi.columns.intersection(nb.columns)]], ignore_index=True)
    SOURCE += f' + {NEW_BUILD}'
dfi['notes'] = dfi['notes'].fillna('')

# ─────────────────────────────────────────────────────────────
# SITE TYPE STYLE
# ─────────────────────────────────────────────────────────────
TYPE_COLOURS = {
    'passenger airport': 'red',
    'park / oval HLS':   'green',
    'heliport':          'purple',
    'helipad / HLS':     'blue',
    'NSW POI helipad':   'cadetblue',
    'GA aerodrome':      'orange',
    'hospital / medical helipad': 'pink',
    'private / station strip':    'beige',
    'restricted site':            'black',
    'out of range (>150 km)':    'lightgray',
    'station car park':           'deeppink',
    'shopping centre rooftop':    'saddlebrown',
    'park / oval':                'lime',
}
TYPE_RADIUS = {'passenger airport': 8}   # airports drawn larger; everything else radius 5

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
m_map = folium.Map(
    location=[-33.0, 151.0],
    zoom_start=7,
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
    attr='Esri, HERE, Garmin, FAO, NOAA, USGS'
)

MeasureControl(
    position='bottomleft',
    primaryLengthUnit='kilometers',
    secondaryLengthUnit='meters',
    activeColor='orange',
    completedColor='green'
).add_to(m_map)

m_map.get_root().html.add_child(folium.Element('''
    <style>.leaflet-interactive:focus { outline: none !important; }</style>
'''))

# draw feeders first so airports sit on top
dfi['_order'] = (dfi['role'] == 'airport').astype(int) + dfi['new_build'].astype(int)
for _, row in dfi.sort_values('_order').iterrows():
    colour = TYPE_COLOURS.get(row['site_type'], 'gray')
    popup_html = f"<b>{row['name']}</b><br>{row['site_type']} ({row['role']})<br>Access: {row['access']}<br>"
    if row['role'] == 'feeder':
        popup_html += f"Nearest airport: {row['nearest_airport']} ({row['dist_airport_km']} km)<br>"
    if row['notes']:
        popup_html += f"<i>{row['notes']}</i><br>"
    popup_html += f"{row['latitude']:.5f}, {row['longitude']:.5f}"
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=7 if row['new_build'] else TYPE_RADIUS.get(row['site_type'], 5),
        color='black' if row['new_build'] else colour, weight=2 if row['new_build'] else 3,
        fill=True, fill_color=colour, fill_opacity=1,
        popup=folium.Popup(popup_html, max_width=300), tooltip=row['name']
    ).add_to(m_map)

# ─────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────
counts = dfi['site_type'].value_counts()
legend_rows = ''.join(
    f'''<p style="margin: 5px 0;"><i style="background:{colour}; width:12px; height:12px;
       display:inline-block; border-radius:50%;"></i> {t} ({counts[t]})</p>'''
    for t, colour in TYPE_COLOURS.items() if t in counts
)
legend_html = f'''
<div style="position: fixed; bottom: 50px; right: 50px; width: 290px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:14px; padding: 10px; border-radius: 5px;">
<p style="margin: 0 0 10px 0; font-weight: bold;">Shuttle Candidates ({len(dfi)})</p>
{legend_rows}
</div>
'''
m_map.get_root().html.add_child(folium.Element(legend_html))

m_map.save('shuttle_vertiports_map.html')
print(f"Plotted {len(dfi)} sites from {SOURCE}. Map saved to shuttle_vertiports_map.html")
print(counts.to_string())
