import pandas as pd
import requests
import time

INPUT_FILE = "model/sa2_names_with_coord.xlsx"
OUTPUT_FILE = "weather.xlsx"

USERNAME = "abc@gmail.com"

START_DATE = "20200101"
END_DATE = "20251231"

# ==========================
# LOAD SUBURBS
# ==========================

df = pd.read_excel(INPUT_FILE)

# Get column names
df.columns = df.columns.str.strip()

# Add result columns
df["Avg_Temp"] = None
df["Avg_Rainfall"] = None
df["Avg_Vapour_Pressure"] = None
df["Avg_WindSpeed"] = None
df["Avg_WindGust"] = None

# ==========================
# PROCESS EACH SUBURB
# ==========================

for idx, row in df.iterrows():

    # We require coordinates to nearest 0.05 degree
    lat = round(float(row["latitude"]) * 20) / 20
    lon = round(float(row["longitude"]) * 20) / 20

    print(
        f"Processing {idx+1}/{len(df)} "
        f"({row['SA2_Name']})"
    )

    url = (
        "https://www.longpaddock.qld.gov.au/cgi-bin/silo/"
        "DataDrillDataset.php"
    )

    params = {
        "lat": lat,
        "lon": lon,
        "start": START_DATE,
        "finish": END_DATE,
        "format": "alldata",
        "username": USERNAME,
        "password": "apirequest"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        text = response.text

        temps = []
        rainfalls = []
        heavy_rain = 0
        vapour_pressure = []

        for line in text.splitlines():

            line = line.strip()

            # Skip comments/header lines
            if (
                not line
                or line.startswith('"')
                or line.startswith("Date")
                or line.startswith("(")
            ):
                continue

            parts = line.split()

            # Ignore non data rows
            if len(parts) < 16:
                continue

            if not parts[0].isdigit():
                continue

            try:

                # Get information based on index position
                tmax = float(parts[3])
                tmin = float(parts[5])

                rain = float(parts[7])

                vp = float(parts[13])

                avg_day_temp = (tmax + tmin) / 2
                
                temps.append(avg_day_temp)
                rainfalls.append(rain)
                vapour_pressure.append(vp)
                if rain >= 50.0:
                    heavy_rain += 1

            except ValueError:
                continue

        # Calculate averages for each suburb

        if len(temps) > 0:
            df.at[idx, "Avg_Temp"] = (
                sum(temps) / len(temps)
            )

        if len(rainfalls) > 0:
            df.at[idx, "Avg_Rainfall"] = (
                sum(rainfalls) / len(rainfalls)
            )
            
        df.at[idx, "Heavy_Rain_Days"] = heavy_rain

        if len(vapour_pressure) > 0:
            df.at[idx, "Avg_Vapour_Pressure"] = (
                sum(vapour_pressure) / len(vapour_pressure)
            )

    except Exception as e:
        print(
            f"Failed for {row['SA2_Name']}: {e}"
        )

    time.sleep(0.1)
    
    #Wind Data
    wind_url = "https://archive-api.open-meteo.com/v1/archive"

    wind_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "daily": "wind_speed_10m_max,wind_gusts_10m_max"
    }

    try :
        
        wind_response = requests.get(
            wind_url,
            params=wind_params,
            timeout=60
        )

        wind_response.raise_for_status()

        wind_data = wind_response.json()

        winds = wind_data["daily"]["wind_speed_10m_max"]
        gusts = wind_data["daily"]["wind_gusts_10m_max"]

        if len(winds) > 0:
            df.at[idx, "Avg_WindSpeed"] = (
                sum(winds) / len(winds)
            )

        if len(gusts) > 0:
            df.at[idx, "Avg_WindGust"] = (
                sum(gusts) / len(gusts)
            )

    except Exception as e:
        print(
            f"Failed for {row['SA2_Name']}: {e}"
        )

# ==========================
# OVERALL AVERAGES
# ==========================

df["Avg_Temp"] = pd.to_numeric(
    df["Avg_Temp"],
    errors="coerce"
)

df["Avg_Rainfall"] = pd.to_numeric(
    df["Avg_Rainfall"],
    errors="coerce"
)

df["Heavy_Rain_Days"] = pd.to_numeric(
    df["Heavy_Rain_Days"],
    errors="coerce"
)

df["Avg_Vapour_Pressure"] = pd.to_numeric(
    df["Avg_Vapour_Pressure"],
    errors="coerce"
)

df["Avg_WindSpeed"] = pd.to_numeric(
    df["Avg_WindSpeed"],
    errors="coerce"
)

df["Avg_WindGust"] = pd.to_numeric(
    df["Avg_WindGust"],
    errors="coerce"
)

overall_temp = df["Avg_Temp"].mean()
overall_rain = df["Avg_Rainfall"].mean()
overall_vp = df["Avg_Vapour_Pressure"].mean()
overall_wind = df["Avg_WindSpeed"].mean()
overall_gust = df["Avg_WindGust"].mean()

print("\n===== AVERAGES =====")
print(f"Temperature: {overall_temp:.2f} C")
print(f"Rainfall: {overall_rain:.2f} mm")
print(f"VP: {overall_vp:.2f} hPa")
print(f"Wind Speed: {overall_wind:.2f} km/h")
print(f"Wind Gust: {overall_gust:.2f} km/h")

# ==========================
# SAVE RESULTS
# ==========================

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved results to "
    f"{OUTPUT_FILE}"
)
