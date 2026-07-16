import numpy as np
import os
import pandas as pd

# --- EDIT YOUR FILE PATHS HERE ---
input_file = "SA2_2021_AUST.xlsx"  # Changed extension to .xlsx
output_file = "NSW_SA2_Area_2021.csv"
# ---------------------------------

# Fallback: Look for any XLSX file starting with "SA2_2021" if the exact name isn't found
if not os.path.exists(input_file):
    matching_files = [f for f in os.listdir('.') if f.startswith('SA2_2021') and f.endswith('.xlsx')]
    if matching_files:
        input_file = matching_files[0]
        print(f"Exact file not found. Auto-selected fallback: '{input_file}'")

try:
    # 1. Load the XLSX dataset using the openpyxl engine
    print(f"Reading Excel workbook '{input_file}'...")
    df = pd.read_excel(input_file, engine='openpyxl')
    
    # 2. Filter for New South Wales
    # Using .astype(str) to handle any non-string values safely
    nsw_mask = df['STATE_NAME_2021'].astype(str).str.strip() == 'New South Wales'
    df_nsw = df[nsw_mask].copy()
    
    # 3. Select and rename the required columns
    columns_to_keep = {
        'SA2_CODE_2021': 'ABS_Code',
        'SA2_NAME_2021': 'SA2_Name',
        'AREA_ALBERS_SQKM': 'Area_KM2'
    }
    
    df_filtered = df_nsw[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
    
    # 4. Calculate Equiv_Radius_KM: sqrt(Area_KM2 / pi)
    # Using pd.to_numeric to ensure math safely applies to numeric values
    df_filtered['Area_KM2'] = pd.to_numeric(df_filtered['Area_KM2'], errors='coerce')
    df_filtered['Equiv_Radius_KM'] = np.sqrt(df_filtered['Area_KM2'] / np.pi)
    
    # 5. Save the result to a clean CSV file
    df_filtered.to_csv(output_file, index=False)
    print(f"Success! Filtered data saved to '{output_file}'")
    print(f"Total records processed: {len(df_filtered)}")

except FileNotFoundError:
    print(f"Error: Could not find any suitable .xlsx input file in this directory.")
except KeyError as e:
    print(f"Error: Missing expected column in the dataset - {e}")
except ImportError:
    print("Error: Required libraries are missing. Run: pip install openpyxl pandas numpy")