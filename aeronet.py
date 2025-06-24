
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "data/AERONET/LWN_Level15_All_Points_V3/LWN/LWN15/ALL_POINTS/20020101_20250531_AAOT.LWN_lev15"
# filename = "data/AERONET/LWN_Level20_All_Points_V3/LWN/LWN20/ALL_POINTS/20020101_20250607_AAOT.LWN_lev20"

df = pd.read_csv(filename, skiprows=6)  # Adjust skiprows if needed
lw_cols = [col for col in df.columns if col.startswith("Lw[") and 'Empty' not in col]
df_lw = df[['Date(dd-mm-yyyy)', 'Time(hh:mm:ss)'] + lw_cols].copy()

df_lw_subset = df_lw[df_lw['Date(dd-mm-yyyy)'] == '04:03:2025']
df_lw_subset = df_lw_subset[df_lw_subset['Time(hh:mm:ss)'] == '10:39:40']

lw = df_lw_subset[lw_cols].values[0]
bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in lw_cols])

# Filter bands and lw where lw > 0
mask = lw > 0
bands = bands[mask]
lw = lw[mask]

# Load solar irradiance data
solar_irradiance_file = "Solar_irradiance_Thuillier_2002.csv"
solar_irradiance_df = pd.read_csv(solar_irradiance_file)  # Skip header row
solar_bands = solar_irradiance_df['nm'].values
solar_irradiance = solar_irradiance_df['mW/m2/nm'].values
# convert to 'mW/(cm2*sr-1*um)'
solar_irradiance = solar_irradiance / (100**2*np.pi) * 1000

# Interpolate solar irradiance to match AERONET bands
ed = np.interp(bands, solar_bands, solar_irradiance)



# Calculate Ed (downwelling irradiance)

# Calculate Rrs (remote sensing reflectance)

rrs = lw / ed  # Rrs in sr⁻¹
plt.scatter(bands, rrs, marker='o')
plt.show()

from hypso import Hypso2
l1a_path = "data/HYPSO-2/L1A" 
file = "aeronetvenice_2025-03-04T10-38-05Z-l1a.nc"
satobj_h2 = Hypso2(path=l1a_path+"/"+file, verbose=True)
satobj_h2.run_direct_georeferencing()
latitudes = satobj_h2.latitudes
longitudes = satobj_h2.longitudes

target_lat = 45.314
target_lon = 12.508

# Compute distance to target for each pixel
dist = np.sqrt((latitudes - target_lat)**2 + (longitudes - target_lon)**2)
min_idx = np.unravel_index(np.argmin(dist), dist.shape)
closest_lat = latitudes[min_idx]
closest_lon = longitudes[min_idx]

print(f"Closest pixel index: {min_idx}")
print(f"Closest pixel coordinates: lat={closest_lat}, lon={closest_lon}")

# print(satobj_h2.latitudes.shape, satobj_h2.longitudes.shape)
# print(satobj_h2.latitudes, satobj_h2.longitudes)
# print(satobj_h2.latitudes_direct, satobj_h2.longitudes_direct)

# coordinate Venice: (302, 546)