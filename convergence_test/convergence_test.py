"""

Convergence test of the MACHI algorithm: Minimal Atmospheric Compensation for Hyperspectral Imagers.
For MSc thesis, spring 2025. See thesis report for details. 
Author: Edvin Dannäs
Date: 2025-06-17

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import main
from utils import get_rgb, load_data, apply_saturation_mask

# --------------- INPUT -----------------
satellite = "HYPSO-2"  # Satellite name
file = "oslofjord_2025-06-20T10-33-03Z-l1d.npz"  # Input file with reflectance data
maxIter = 200                       # Maximum number of iterations 
epsilon = 1e-2                      # Convergence threshold
batch_size = [100, 200]            # Batch sizes to test
water_pixel = (500, 400)            # Pixel to evaluate (598, 1092)
land_pixel = (310, 780)             # Pixel to evaluate (598, 1092) (302, 546)
crop = None                         # (y1, y2, x1, x2)
h1 = np.array([1, 0, -1])           # First-order central difference kernel
h2 = np.array([1, -2, 1])           # Second-order central difference kernel
h3 = np.array([1, -3, 3, -1])       # Third-order central difference kernel
h4 = np.array([1, -4, 6, -4, 1])    # Fourth-order central difference kernel
kernels = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4}      # Dictionary of kernels
# ---------------------------------------

# Set plot layout
color_map = {'h1': 'blue', 'h2': 'm', 'h3': 'orange', 'h4': 'green'}  # Color map for kernels
linestyles = {'b100': '-', 'b200': '--', 'b300': '-.', 'b400': ':'}

# Create file path and save folder
file_path = 'data/{}/ToA-reflectance/{}'.format(satellite, file)  # Path to the input file
if satellite in ["HYPSO-1", "HYPSO-2"]:
    site = file.replace("-l1d.npz", "")
    save_folder = f"convergence_test/{satellite}/{site}"
    os.makedirs(save_folder, exist_ok=True)
else:
    raise ValueError("Unsupported satellite. (Currently supported data: HYPSO-1 and HYPSO-2.)")

# Plot 1: Load and plot ToA reflectance data with cropping window and evaluation pixels
bands, R_TOA_uncropped, sat_mask = load_data(file_path)
R_TOA_uncropped = apply_saturation_mask(R_TOA_uncropped, sat_mask)
(y,x) = R_TOA_uncropped.shape[:2]
scale = 8*1e-3
x_scale = 0.3
plt.figure(figsize=(x*scale*x_scale, y*scale))
plt.imshow(
    np.nan_to_num(get_rgb(R_TOA_uncropped, bands), nan=0),
    extent=[0, x, y, 0],  # Match image to figure size and axes
    aspect='auto'
)
if crop is not None:
    y1, y2, x1, x2 = map(int, crop)  # Ensure integers
    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1),           # (x, y) = (column, row)
            x2 - x1,            # width
            y2 - y1,            # height
            edgecolor='red',
            facecolor='none',   # No fill
            linewidth=2
        )
    )
plt.scatter(water_pixel[1], water_pixel[0], color='blue', s=80, marker='x', label='Water pixel')
plt.scatter(land_pixel[1], land_pixel[0], color='green', s=80, marker='x', label='Land pixel')
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
    for b in batch_size:
        key = f"b{b}_{h_name}"
        R_TOA[key], R_BOA[key], S[key], T[key], bands, i[key], t[key], P[key] = main(file_path, h=h_function, batch_size=b)

# Plot 2: convergence for each kernel and all batch sizes
fig, axs = plt.subplots(len(kernels), 1, figsize=(9, 2.5*len(kernels)), sharex=True)
for index, h in enumerate(kernels.keys()):
    for b in batch_size:
        key = f'b{b}_{h}'
        axs[index].plot(np.arange(len(P[key])) / 2 * b, P[key] / b, color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index].set_xlabel("Iteration × Batch size")
        axs[index].set_ylabel("Penalty / Batch size")
        axs[index].legend()
        axs[index].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{save_folder}/convergence_batches.png", dpi=300)

# log scale
fig, axs = plt.subplots(len(kernels), 1, figsize=(9, 2.5*len(kernels)), sharex=True)
for index, h in enumerate(kernels.keys()):
    for b in batch_size:
        key = f'b{b}_{h}'
        axs[index].plot(np.arange(len(P[key])) / 2 * b, P[key] / b, color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index].set_yscale("log")
        axs[index].set_xlabel("Iteration × Batch size")
        axs[index].set_ylabel("Penalty / Batch size")
        axs[index].legend()
        axs[index].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{save_folder}/convergence_batches_log.png", dpi=300)

# Plot 3: final S and T for each kernel
fig, axs = plt.subplots(len(kernels), 2, figsize=(9, 2.7*len(kernels)), sharex=True) # sharex='col', sharey='row'
global_max_S = max([max(S[f"b{b}_{h}"]) for h in kernels.keys() for b in batch_size])
for index, h in enumerate(kernels.keys()):
    for b in batch_size:
        key = f'b{b}_{h}'
        # S
        axs[index][0].plot(bands, S[key], color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index][0].set_xlabel("Band [nm]")
        axs[index][0].set_ylabel("Scattering [-]")        
        axs[index][0].set_ylim(0, global_max_S * 1.1)  # Set y-limits to 10% above global max value
        axs[index][0].legend()
        axs[index][0].grid(True, which='both', linestyle='--', linewidth=0.5)
        # T
        axs[index][1].plot(bands, T[key], color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index][1].set_xlabel("Band [nm]")
        axs[index][1].set_ylabel("Transmission [-]")
        axs[index][1].set_ylim(0, 1)
        axs[index][1].legend()
        axs[index][1].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{save_folder}/final_S_T_batches.png", dpi=300)

# Plot 4: BOA for each kernel and all batch sizes
fig, axs = plt.subplots(len(kernels), 2, figsize=(9, 2.7*len(kernels)), sharex=True) # sharex='col', sharey='row'
water_max_R_BOA = max([np.nanmax(R_BOA[f"b{b}_{h}"][water_pixel[0], water_pixel[1], :]) for h in kernels.keys() for b in batch_size])
land_max_R_BOA = max([np.nanmax(R_BOA[f"b{b}_{h}"][land_pixel[0], land_pixel[1], :]) for h in kernels.keys() for b in batch_size])
for index, h in enumerate(kernels.keys()):
    for b in batch_size:
        key = f'b{b}_{h}'
        # water pixel
        axs[index][0].plot(bands, R_BOA[key][water_pixel[0], water_pixel[1], :], color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index][0].set_xlabel("Band [nm]")
        axs[index][0].set_ylabel("BoA reflectance \n (water pixel) [-]")
        axs[index][0].set_ylim(0, water_max_R_BOA * 1.1)  # Set y-limits to 10% above max value
        axs[index][0].legend()
        axs[index][0].grid(True, which='both', linestyle='--', linewidth=0.5)
        # land pixel
        axs[index][1].plot(bands, R_BOA[key][land_pixel[0], land_pixel[1], :], color=color_map[h], linestyle=linestyles[f'b{b}'], label=f"{h} - Batch size {b}", linewidth=1)
        axs[index][1].set_xlabel("Band [nm]")
        axs[index][1].set_ylabel("BoA reflectance \n (land pixel) [-]")
        axs[index][1].set_ylim(0, land_max_R_BOA * 1.1)
        axs[index][1].legend()
        axs[index][1].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{save_folder}/final_BoA_batches.png", dpi=300)