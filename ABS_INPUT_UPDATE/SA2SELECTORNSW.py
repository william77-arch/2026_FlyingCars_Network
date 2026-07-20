import numpy as np
import pandas as pd

# ---------- FILE PATHS ----------
INPUT_FILE = "SA2_2021_AUST.xlsx"
OUTPUT_FILE = "NSW_SA2_Area_2021.csv"

try:
    print(f"Reading Excel with name '{INPUT_FILE}'...")
    df = pd.read_excel(INPUT_FILE, engine='openpyxl')

    # We need only NSWs from this
    nsw_mask = df['STATE_NAME_2021'].astype(str).str.strip() == 'New South Wales'
    df_nsw = df[nsw_mask].copy()

    # Match format of model
    columns_to_keep = {
        'SA2_CODE_2021': 'ABS_Code',
        'SA2_NAME_2021': 'SA2_Name',
        'AREA_ALBERS_SQKM': 'Area_KM2'
    }
    
    
    df_filtered = df_nsw[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

    # Calculations for equiv rad
    df_filtered['Area_KM2'] = pd.to_numeric(df_filtered['Area_KM2'], errors='coerce')
    df_filtered['Equiv_Radius_KM'] = np.sqrt(df_filtered['Area_KM2'] / np.pi)

    df_filtered.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Filtered data saved to '{OUTPUT_FILE}'")

except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_FILE}' in this directory.")
except KeyError as e:
    print(f"Error: Missing expected column in the dataset - {e}")
except ImportError:
    print("Error: Missing Libraries")