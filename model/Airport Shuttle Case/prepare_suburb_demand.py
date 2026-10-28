"""
Builds suburbs_with_coord.xlsx (the demand file for shuttle_optim_model.py) from ABS data.

Download (free, ABS, 2021 editions) and put them in this folder (zipped or unzipped):
  1. Suburbs and Localities boundaries   SAL_2021_AUST_GDA2020_SHP.zip
     https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SAL_2021_AUST_GDA2020_SHP.zip
  2. SA4 boundaries                       SA4_2021_AUST_SHP_GDA2020.zip   (region names and the
     Greater Sydney flag GCC_NAME21)
     https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SA4_2021_AUST_SHP_GDA2020.zip
  3. Census population by suburb         https://www.abs.gov.au/census/find-census-data/datapacks
     Choose 2021 > General Community Profile > Suburbs and Localities (SAL) > New South Wales
     (zip, ~34 MB). Leave it zipped: this script finds 2021Census_G01_NSW_SAL.csv inside it
     (Tot_P_P = total persons). An extracted copy of that CSV also works.

Output columns: SAL_NAME21, SAL_CODE21, latitude, longitude, population, SA4_NAME21, GCC_NAME21
latitude/longitude is a point guaranteed to lie inside the suburb (not the plain centroid,
which can fall outside an oddly shaped suburb).

    pip install geopandas pandas openpyxl
    python prepare_suburb_demand.py
"""

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

# each can be the unzipped folder, the .zip, or the file itself (.shp / .csv)
SAL_FILE = 'SAL_2021_AUST_GDA2020_SHP'             # suburb boundaries
SA4_FILE = 'SA4_2021_AUST_SHP_GDA2020'             # optional (None to skip)
POP_FILE = '2021_GCP_SAL_for_NSW_short-header'     # Census DataPack; the G01 NSW SAL table is found inside
OUTPUT   = 'suburbs_with_coord.xlsx'
STATE    = 'New South Wales'
MGA      = 7856       # GDA2020 / MGA zone 56: metres, for placing the inside point accurately


def code_digits(s):
    return re.sub(r'\D', '', str(s))               # 'SAL10001' and '10001' -> '10001'


def find_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    raise SystemExit(f'None of {options} found in columns {list(df.columns)[:15]}...')


def locate(path):
    """The folder, the .zip of the same name, or the file itself - whichever exists."""
    for c in [Path(path), Path(f'{path}.zip')]:
        if c.exists():
            return c
    raise SystemExit(f'{path} (folder, .zip or file) not found in {Path.cwd()}')


def read_shapes(path):
    p = locate(path)
    if p.is_dir():                                      # unzipped folder: find the .shp inside
        shp = sorted(p.rglob('*.shp'))
        if not shp:
            raise SystemExit(f'no .shp file inside {p}')
        print(f'reading {shp[0]}')
        return gpd.read_file(shp[0])
    return gpd.read_file(f'zip://{p}' if p.suffix == '.zip' else p)


sal = read_shapes(SAL_FILE)
name_c = find_col(sal, ['SAL_NAME21', 'SAL_NAME_2021'])
code_c = find_col(sal, ['SAL_CODE21', 'SAL_CODE_2021'])
ste_c = find_col(sal, ['STE_NAME21', 'STE_NAME_2021'])
sal = sal[(sal[ste_c] == STATE) & sal.geometry.notna() & ~sal.geometry.is_empty].copy()
print(f'{len(sal)} suburbs/localities in {STATE}')

# a point inside each suburb, computed in metres, then back to lat/lon
pts = sal.to_crs(MGA).representative_point().to_crs(4326)
sal['longitude'], sal['latitude'] = pts.x.round(6), pts.y.round(6)
sal['code'] = sal[code_c].map(code_digits)

def read_population():
    import zipfile
    p = locate(POP_FILE)
    if p.is_file() and p.suffix == '.csv':
        return pd.read_csv(p)
    if p.is_dir():                                      # unzipped DataPack folder
        hit = sorted(f for f in p.rglob('*.csv') if f.name.lower().endswith('g01_nsw_sal.csv'))
        if hit:
            print(f'reading {hit[0]}')
            return pd.read_csv(hit[0])
    elif p.suffix == '.zip':                            # still-zipped DataPack
        with zipfile.ZipFile(p) as zf:
            hit = [n for n in zf.namelist() if n.lower().endswith('g01_nsw_sal.csv')]
            if hit:
                print(f'reading {hit[0]} from {p.name}')
                with zf.open(hit[0]) as f:
                    return pd.read_csv(f)
    raise SystemExit(f'no 2021Census_G01_NSW_SAL.csv found in {p}')


pop = read_population()
pc = find_col(pop, ['SAL_CODE_2021', 'SAL_CODE21'])
tot = find_col(pop, ['Tot_P_P', 'Tot_P'])
pop = pop.assign(code=pop[pc].map(code_digits))[['code', tot]].rename(columns={tot: 'population'})
sal = sal.merge(pop, on='code', how='left')
missing = sal['population'].isna().sum()
sal['population'] = sal['population'].fillna(0).astype(int)
print(f'population joined: {len(sal) - missing} matched, {missing} without a Census count (set to 0)')

out = sal[[name_c, code_c, 'latitude', 'longitude', 'population']].rename(
    columns={name_c: 'SAL_NAME21', code_c: 'SAL_CODE21'})

if SA4_FILE and (Path(SA4_FILE).exists() or Path(f'{SA4_FILE}.zip').exists()):
    sa4 = read_shapes(SA4_FILE)
    keep = [c for c in ['SA4_NAME21', 'GCC_NAME21'] if c in sa4.columns]
    p_gdf = gpd.GeoDataFrame(out[['SAL_CODE21']], geometry=pts.values, crs=4326)
    joined = gpd.sjoin(p_gdf, sa4[keep + ['geometry']].to_crs(4326), how='left', predicate='within')
    joined = joined[~joined.index.duplicated()]
    for c in keep:
        out[c] = joined[c].values
    gs = (out.get('GCC_NAME21') == 'Greater Sydney')
    print(f'SA4 regions added; {int(gs.sum())} suburbs in Greater Sydney, '
          f'{int(out.loc[gs, "population"].sum()):,} people')
else:
    print('No SA4 file: no region column; the model will use its Greater Sydney box instead')

# the model needs unique names; ABS adds "(NSW)" etc. to repeated names already, but check
dup = out['SAL_NAME21'].duplicated(keep=False)
out.loc[dup, 'SAL_NAME21'] = out.loc[dup, 'SAL_NAME21'] + ' [' + out.loc[dup, 'SAL_CODE21'].astype(str) + ']'

out.to_excel(OUTPUT, index=False)
print(f'Saved {OUTPUT} ({len(out)} rows)')