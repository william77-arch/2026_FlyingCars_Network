import tempfile
import zipfile
import geopandas as gpd
import pandas as pd
import requests

# ---------- FILE PATHS ----------
INPUT_FILE = "NSW_SA2_Area_2021.csv"
OUTPUT_FILE = "NSW_SA2_with_Calculated_Coords.csv"

# SA2 2021 Shapefile Digital boundary files
# Update from https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files
# When needed paste url of zip download here
abs_shapefile_zip_url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip"

try:
    print(f"Reading from '{INPUT_FILE}'...")
    df_nsw = pd.read_csv(INPUT_FILE)

    print("Downloading Shapefile")
    response = requests.get(abs_shapefile_zip_url)
    response.raise_for_status()

    print("Extracting spatial map files safely to a temporary directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = f"{temp_dir}/shapes.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        shp_file_path = f"{temp_dir}/SA2_2021_AUST_GDA2020.shp"

        print("Loading geographic shapes into GeoPandas...")
        gdf_sa2 = gpd.read_file(shp_file_path)

    print("Calculating Lat/Long centroids...")
    # project to a metric CRS first (GDA2020 / Lambert) so the centroid math is accurate,
    # then convert back to lat/long (WGS84)
    gdf_projected = gdf_sa2.to_crs(epsg=7855)
    centroids_projected = gdf_projected.geometry.centroid
    centroids_latlon = centroids_projected.to_crs(epsg=4326)

    gdf_sa2['longitude'] = centroids_latlon.x
    gdf_sa2['latitude'] = centroids_latlon.y
    df_coords = pd.DataFrame(gdf_sa2[['SA2_CODE21', 'latitude', 'longitude']])

    df_nsw['ABS_Code'] = df_nsw['ABS_Code'].astype(str).str.strip()
    df_coords['SA2_CODE21'] = df_coords['SA2_CODE21'].astype(str).str.strip()

    print("Merging spatial coordinates onto your dataset...")
    df_final = pd.merge(df_nsw, df_coords, left_on='ABS_Code', right_on='SA2_CODE21', how='left')

    ordered_columns = ['ABS_Code', 'SA2_Name', 'longitude', 'latitude', 'Area_KM2', 'Equiv_Radius_KM']
    df_final = df_final[ordered_columns]

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Data saved to '{OUTPUT_FILE}'")

except Exception as e:
    print(f"An error occurred: {e}")