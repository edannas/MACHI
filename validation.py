"""

Demo of the MACHI algorithm: Minimal Atmospheric Correction for Hyperspectral Imagers.
For MSc thesis, spring 2025. See thesis report for details. 
Author: Edvin Dannäs
Date: 2025-06-17

"""
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import xarray as xr
import os
from main import main
from utils import get_rgb, load_data, apply_saturation_mask
from hypso import Hypso2

# --------------- INPUT -----------------
satellite = "HYPSO-2"  # Satellite name
file = "aeronetvenice_2025-03-04T10-38-05Z-l1d.npz"     # Input file with reflectance data
validation_data = '20020101_20250531_AAOT.LWN_lev15'
validation_site = (302+4, 545-160)                      # Venice validation site coordinates (row, column)
validation_site = (302, 545)                            # Venice validation site coordinates (row, column)
maxIter = 200                                           # Maximum number of iterations 
epsilon = 1e-2                                          # Convergence threshold
batch_size = 100                                        # List of batch sizes to run
crop = None                                             # (y1, y2, x1, x2)
h1 = np.array([1, 0, -1])                               # First-order central difference kernel
h2 = np.array([1, -2, 1])                               # Second-order central difference kernel
h3 = np.array([1, -3, 3, -1])                           # Third-order central difference kernel
h4 = np.array([1, -4, 6, -4, 1])                        # Fourth-order central difference kernel
#h_sg2 = np.array([-1, 16, -30, 16, -1]) / 12
#h_lap7 = np.array([-1, 5, -10, 10, -5, 1])
#h_dog = np.array([0.5, -1, 0.5])
#h_haar = np.array([1, -1])  # 1st difference (Haar wavelet)
#h_wave = np.array([-1, -1, 2, -1, -1])
kernels = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4}      # Dictionary of kernels
# ---------------------------------------

# Set plot layout
color_map = {'h1': 'blue', 'h2': 'm', 'h3': 'orange', 'h4': 'green'}  # Color map for kernels
color_map_light = {'h1': '#4F8CFF', 'h2': "#DB30DB", 'h3': "#DBA333", 'h4': "#2BD32B"}
color_map_dark = {'h1': '#003366', 'h2': '#800080', 'h3': "#8E4E00", 'h4': "#015801"}
# linestyles = {'b100': '-', 'b200': '--', 'b300': '-.', 'b400': ':'}

# Create file path and save folder 
file_path = 'data/{}/ToA-reflectance/{}'.format(satellite, file)  # Path to the input file
if satellite in ["HYPSO-1", "HYPSO-2"]:
    site = file.replace("-l1d.npz", "")
    save_folder = f"output/{satellite}/{site}"
    os.makedirs(save_folder, exist_ok=True)
else:
    raise ValueError("Unsupported satellite. (Currently supported data: HYPSO-1 and HYPSO-2.)")

# Load and process AERONET data
validation_path = "data/AERONET/LWN_Level15_All_Points_V3/LWN/LWN15/ALL_POINTS/"
df = pd.read_csv(validation_path + validation_data, skiprows=6)
lw_cols = [col for col in df.columns if col.startswith("Lw[") and 'Empty' not in col]
sz_cols = [col for col in df.columns if col.startswith("Solar_Zenith_Angle[") and 'Empty' not in col]
aod_cols = [col for col in df.columns if col.startswith("Aerosol_Optical_Depth[") and 'Empty' not in col]
ood_cols = [col for col in df.columns if col.startswith("Ozone_Optical_Depth[") and 'Empty' not in col]
rod_cols = [col for col in df.columns if col.startswith("Rayleigh_Optical_Depth[") and 'Empty' not in col]
nod_cols = [col for col in df.columns if col.startswith("NO2_Optical_Depth[") and 'Empty' not in col]
wod_cols = [col for col in df.columns if col.startswith("Water_Vapor_Optical_Depth[") and 'Empty' not in col]

df_subset = df[lw_cols + sz_cols + aod_cols + ood_cols + rod_cols + nod_cols + wod_cols].copy()
df_subset['datetime'] = pd.to_datetime(
    df['Date(dd-mm-yyyy)'] + ' ' + df['Time(hh:mm:ss)'],
    format='%d:%m:%Y %H:%M:%S'
)
target_lat = df['Site_Latitude(Degrees)'].unique()
target_lon = df['Site_Longitude(Degrees)'].unique()

# Extract capture time from HYPSO filename
filename_time = file.split('_')[1].split('-l1d')[0]
capture_time_str = filename_time.replace('T', ' ').replace('Z', '')
capture_time = datetime.strptime(capture_time_str, "%Y-%m-%d %H-%M-%S")

# Find closest AERONET time to capture_time
closest_idx = (df_subset['datetime'] - capture_time).abs().idxmin()
df_subset = df_subset.loc[closest_idx]
print("AERONET-OC validation datetime:", df_subset['datetime'])

# Extract lw values and bands
lw = df_subset[lw_cols].values.astype(float)
lw_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in lw_cols])
mask = lw > 0
lw_bands = lw_bands[mask][2:]
lw = lw[mask][2:]

# Extract sz, aod, ood, rod for the same bands as lw
sz = df_subset[sz_cols].values.astype(float)
aod = df_subset[aod_cols].values.astype(float)
ood = df_subset[ood_cols].values.astype(float)
rod = df_subset[rod_cols].values.astype(float)
nod = df_subset[nod_cols].values.astype(float)
wod = df_subset[wod_cols].values.astype(float)

# Get band numbers for each column
sz_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in sz_cols])
aod_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in aod_cols])
ood_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in ood_cols])
rod_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in rod_cols])
nod_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in nod_cols])
wod_bands = np.array([int(col.split('[')[1].split('nm]')[0]) for col in wod_cols])

# Interpolate to lw_bands
sz = np.interp(lw_bands, sz_bands, sz)
aod = np.interp(lw_bands, aod_bands, aod)
ood = np.interp(lw_bands, ood_bands, ood)
rod = np.interp(lw_bands, rod_bands, rod)
nod = np.interp(lw_bands, nod_bands, nod)
wod = np.interp(lw_bands, wod_bands, wod)

# Calculate total optical depth (TOD)
tod = aod + ood + rod + nod + wod  # Total optical depth

# Load solar irradiance data
solar_irradiance_file = "Solar_irradiance_Thuillier_2002.csv"
solar_irradiance_df = pd.read_csv(solar_irradiance_file)  # Skip header row
solar_bands = solar_irradiance_df['nm'].values
solar_irradiance = solar_irradiance_df['mW/m2/nm'].values
solar_irradiance = solar_irradiance / (100**2) * 1000 # convert to 'mW/(cm2*um)'
ed = np.interp(lw_bands, solar_bands, solar_irradiance) # Interpolate solar irradiance to match lw bands

# Check if sz is in radians or degrees
# If sz values are mostly below 2*pi, assume radians; if up to ~90, assume degrees
if np.nanmax(sz) > 2 * np.pi:
    print("Converting sz from degrees to radians.")
    sz = np.deg2rad(sz)

# Calculate downwelling irradiance at the surface
ed_surface = ed * np.cos(sz) * np.exp(-tod/np.cos(sz))

# Calculate Rrs (remote sensing reflectance)
rrs = lw / ed_surface  # Rrs (sr-1)
boa_aeronet = rrs * np.pi  # BoA reflectance

# Load georeference data
bands, R_TOA_uncropped, sat_mask = load_data(file_path)
(y,x) = R_TOA_uncropped.shape[:2]
georef_path = 'data/{}/georeference/{}'.format(satellite, file.replace("-l1d.npz", ""))
latitudes_path = os.path.join(georef_path, "latitudes_indirectgeoref.dat")
longitudes_path = os.path.join(georef_path, "longitudes_indirectgeoref.dat")
latitudes = np.fromfile(latitudes_path, dtype=np.float32).reshape((y,x))
longitudes = np.fromfile(longitudes_path, dtype=np.float32).reshape((y,x))

# Find closest pixel to validation site
dist = np.sqrt((latitudes - target_lat)**2 + (longitudes - target_lon)**2)
min_idx = np.unravel_index(np.argmin(dist), dist.shape)
min_idx = (min_idx[0], latitudes.shape[1] - 1 - min_idx[1])
closest_lat = latitudes[min_idx]
closest_lon = longitudes[min_idx]
validation_site = min_idx
print(f"Closest pixel index: {min_idx}")
print(f"Closest pixel coordinates: lat={closest_lat}, lon={closest_lon}")

# Plot 1: Load and plot ToA reflectance data and validation site
R_TOA_uncropped = apply_saturation_mask(R_TOA_uncropped, sat_mask)
scale = 8*1e-3
x_scale = 0.3
plt.figure(figsize=(x*scale*x_scale, y*scale))
plt.imshow(
    np.nan_to_num(get_rgb(R_TOA_uncropped, bands), nan=0),
    extent=[0, x, y, 0],  # Match image to figure size and axes
    aspect='auto'
)
# if crop is not None:
#     y1, y2, x1, x2 = map(int, crop)  # Ensure integers
#     plt.gca().add_patch(
#         plt.Rectangle(
#             (x1, y1),           # (x, y) = (column, row)
#             x2 - x1,            # width
#             y2 - y1,            # height
#             edgecolor='red',
#             facecolor='none',   # No fill
#             linewidth=2
#         )
#     )
plt.scatter(validation_site[1] + 0.5, validation_site[0] + 0.5, color='red', s=80, marker='*', label='Validation site')
plt.legend()
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{save_folder}/RGB_TOA.png", dpi=300)

# Run MACHI
R_TOA = {}
R_BOA = {}
S = {}
T = {}
i = {}
t = {}
P = {}
for h_name, h_function in kernels.items():
    key = f"b{batch_size}_{h_name}"
    R_TOA[key], R_BOA[key], S[key], T[key], bands, i[key], t[key], P[key] = main(file_path, h=h_function, batch_size=batch_size)

# Plot 2: R_BOA for each kernel with validation site
"""fig, axs = plt.subplots(2, 2, figsize=(x*scale*2.5, y*scale*2.5), sharex=True) # sharex='col', sharey='row'
for index, h in enumerate(kernels.keys()):
    key = f'b{batch_size}_{h}'
    rgb_boa = get_rgb(R_BOA[key], bands, ref_image=None)
    column = index % 2
    row = int((index-column)/2)
    axs[row][column].imshow(np.nan_to_num(rgb_boa, nan = 0))
    axs[row][column].set_title("Kernel: " + h)
    axs[row][column].axis("off")
plt.tight_layout()
plt.savefig(f"{save_folder}/RGB_BOA.png", dpi=300)"""
# Plot kernels individually
for h in kernels.keys():
    plt.figure(figsize=(x*scale*x_scale, y*scale))
    key = f'b{batch_size}_{h}'
    rgb_boa = get_rgb(R_BOA[key], bands, ref_image=None)
    plt.imshow(np.nan_to_num(rgb_boa, nan = 0),
    extent=[0, x, y, 0],  # Match image to figure size and axes
    aspect='auto'
    )
    plt.scatter(validation_site[1], validation_site[0], color='red', s=80, marker='*', label='Validation site')
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_folder}/RGB_BOA_{h}.png", dpi=300)

# Plot 3: Final S and T for each kernel
fig, axs = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True) # sharex='col', sharey='row'
global_max_S = max([max(S[f"b{batch_size}_{h}"]) for h in kernels.keys()])
for index, h in enumerate(kernels.keys()):
    column = index % 2
    row = int((index-column)/2)
    key = f'b{batch_size}_{h}'
    ax1 = axs[row, column]
    color_s = color_map_light[h]
    color_t = color_map_dark[h]
    # Plot T on left y-axis
    ax1.plot(bands, T[key], color=color_t, label=f"T ({h})", linewidth=1)
    ax1.set_xlabel("Band [nm]")
    ax1.set_ylabel("T [-]", color=color_t)
    ax1.tick_params(axis='y', labelcolor=color_t)
    ax1.set_title(f"Kernel: {h}")
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    # Create right y-axis for S
    ax2 = ax1.twinx()
    ax2.plot(bands, S[key], color=color_s, label=f"S ({h})", linewidth=1)
    ax2.set_ylabel("S [-]", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_ylim(0, global_max_S)
    # Legends
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2)
plt.tight_layout()
plt.savefig(f"{save_folder}/final_S_T.png", dpi=300)

# Check if POLYMER file exists and load if present
polymer_file = file.replace("l1d.npz", "polymer.nc")
polymer_path = os.path.join("data", "POLYMER", polymer_file)
polymer_data = None
if os.path.exists(polymer_path):
    polymer_data = xr.open_dataset(polymer_path)
else:
    print(f"No POLYMER file found")

if polymer_data is not None:
    rho_w_vars = [var for var in polymer_data.variables if var.startswith('rho_w_')]
    rho_w_vars_sorted = sorted(rho_w_vars, key=lambda v: int(v.split('_')[2]))
    rho_w_stack = np.stack([polymer_data[var].values for var in rho_w_vars_sorted], axis=2)
    rho_w_stack = np.flip(rho_w_stack, axis=1)
    rho_w_polymer = rho_w_stack[validation_site[0], validation_site[1], :]
    polymer_bands = np.array(polymer_data["bands"])

# Plot 4: TOA, BOA and ground truth at validation site
plt.figure(figsize=(8, 6))
R_TOA_val = list(R_TOA.values())[0][validation_site]
y_max = np.max(R_TOA_val) * 1.1  # Set y-axis limit to 10% above max ToA reflectance
plt.plot(bands, R_TOA_val, label="ToA", color="black", linewidth=1.5)
for h in kernels.keys():
    key = f'b{batch_size}_{h}'
    plt.plot(bands, R_BOA[key][validation_site], label=f"BoA: {h}", color = color_map[h], linewidth=1)
plt.scatter(lw_bands, boa_aeronet, label="AERONET-OC", color="red", marker='*', s=50)
if polymer_data is not None:
    plt.plot(polymer_bands, rho_w_polymer, label="POLYMER", color="grey", alpha=0.5)
plt.xlabel("Band [nm]")
plt.ylabel("Reflectance [-]")
plt.legend()
plt.grid(True)
plt.ylim(0, y_max)
plt.xlim(min(bands), max(bands))
plt.tight_layout()
plt.savefig(f"{save_folder}/validation.png", dpi=300)

# Plot 5: BOA and ground truth at validation site (water spectrum)
plt.figure(figsize=(8, 6))
x_max = 58
y_max = np.max([R_BOA[key][validation_site][:x_max] for key in R_BOA.keys()]) * 1.1  # Set y-axis limit to 10% above max BoA reflectance
y_min = np.min([R_BOA[key][validation_site][:x_max] for key in R_BOA.keys()]) * 0.9  # Set y-axis limit to 10% below min BoA reflectance
for h in kernels.keys():
    key = f'b{batch_size}_{h}'
    plt.plot(bands[:x_max], R_BOA[key][validation_site][:x_max], color = color_map[h], label=f"BoA: {h}", linewidth=1)
plt.scatter(lw_bands[:-1], boa_aeronet[:-1], label="AERONET-OC", color="red", marker='*', s=50)
if polymer_data is not None:
    plt.plot(polymer_bands, rho_w_polymer, label="POLYMER", color="grey", alpha=0.5)
plt.xlabel("Band [nm]")
plt.ylabel("Reflectance [-]")
plt.legend()
plt.grid(True)
plt.ylim(0, y_max)
plt.xlim(min(bands[:x_max]), max(bands[:x_max]))  # Limit x-axis to visible bands
plt.tight_layout()
plt.savefig(f"{save_folder}/validation_water.png", dpi=300)