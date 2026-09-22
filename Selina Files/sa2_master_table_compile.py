import pandas as pd

INPUT_FILE = "Cleaned_ABS_LDH_data.xlsx"
OUTPUT_FILE = "sa2_master_table.xlsx"

xl = pd.ExcelFile(INPUT_FILE)

frames = []
for sheet in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet)
    df["lhd_name"] = sheet   # tag each SA2 with its LHD
    frames.append(df)

sa2_master = pd.concat(frames, ignore_index=True)

sa2_master.to_excel(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved to: {OUTPUT_FILE}")
