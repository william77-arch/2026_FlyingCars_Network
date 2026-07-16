import os
import tempfile
import zipfile
import geopandas as gpd
import pandas as pd
import requests

# --- CONFIGURATION ---
input_file = "NSW_SA2_Area_2021.csv"  # Takes the output from your updated first script
output_file = "NSW_SA2_with_Calculated_Coords.csv"

# Official 2021 ABS SA2 Digital Boundary Zip file URL
abs_shapefile_zip_url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip"
# ---------------------

try:
    # 1. Load the pre-filtered NSW dataset (Columns: ABS_Code, SA2_Name, Area_KM2, Equiv_Radius_KM)
    print(f"Reading pre-filtered data from '{input_file}'...")
    df_nsw = pd.read_csv(input_file)

    # 2. Download and extract the official ABS Shapefile zip to a temporary folder
    print("Downloading official 2021 ABS Shapefile zip (this may take a moment)...")
    response = requests.get(abs_shapefile_zip_url)
    response.raise_for_status()
    
    print("Extracting spatial map files safely to a temporary directory...")
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "shapes.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        shp_file_path = os.path.join(temp_dir, "SA2_2021_AUST_GDA2020.shp")
        
        print("Loading geographic shapes into GeoPandas...")
        gdf_sa2 = gpd.read_file(shp_file_path)

    # 3. Calculate mathematically precise centroids using a meter-based projection
    print("Calculating Lat/Long centroids...")
    
    # Project to standard Australian metric coordinate system (GDA2020 / Lambert - EPSG:7855)
    gdf_projected = gdf_sa2.to_crs(epsg=7855)
    centroids_projected = gdf_projected.geometry.centroid
    
    # Convert the resulting center points back to standard lat/long degrees (WGS84 - EPSG:4326)
    centroids_latlon = centroids_projected.to_crs(epsg=4326)

    # Add the coordinates back to our main spatial dataframe
    gdf_sa2['longitude'] = centroids_latlon.x
    gdf_sa2['latitude'] = centroids_latlon.y

    # Convert to a standard DataFrame holding only the matching code and coordinates
    df_coords = pd.DataFrame(gdf_sa2[['SA2_CODE21', 'latitude', 'longitude']])

    # 4. Standardize keys and join the datasets
    # Updated key to use your new 'ABS_Code' column label from your Excel filtering script
    df_nsw['ABS_Code'] = df_nsw['ABS_Code'].astype(str).str.strip()
    df_coords['SA2_CODE21'] = df_coords['SA2_CODE21'].astype(str).str.strip()

    print("Merging spatial coordinates onto your dataset...")
    df_final = pd.merge(df_nsw, df_coords, left_on='ABS_Code', right_on='SA2_CODE21', how='left')

    # 5. Enforce precise column ordering as requested
    # A=ABS_Code, B=SA2_Name, C=longitude, D=latitude, E=Area_KM2, F=Equiv_Radius_KM
    ordered_columns = [
        'ABS_Code', 
        'SA2_Name', 
        'longitude', 
        'latitude', 
        'Area_KM2', 
        'Equiv_Radius_KM'
    ]
    df_final = df_final[ordered_columns]

    # 6. Save output with lock protection
    counter = 1
    while True:
        try:
            df_final.to_csv(output_file, index=False)
            print(f"Success! Data saved safely to '{output_file}'")
            break
        except PermissionError:
            print(f"Warning: '{output_file}' is locked. Please close the file if open.")
            output_file = f"NSW_SA2_Calculated_Coords_{counter}.csv"
            counter += 1

    print(f"Total NSW records matched with coordinates: {len(df_final)}")

except Exception as e:
    print(f"An error occurred: {e}")