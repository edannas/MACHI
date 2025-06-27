"""

Demo of the MACHI algorithm: Minimal Atmospheric Correction for Hyperspectral Imagers.
For MSc thesis, spring 2025. See thesis report for details. 
Author: Edvin Dannäs
Date: 2025-06-17

"""
import numpy as np
import matplotlib.pyplot as plt
from main import main
from utils import get_rgb, load_data, apply_saturation_mask
import os

# --------------- INPUT -----------------
satellite = "HYPSO-2"  # Satellite name
file = "aeronetvenice_2025-03-04T10-38-05Z-l1d.npz" # Input file with reflectance data
maxIter = 200                                       # Maximum number of iterations 
epsilon = 1e-2                                      # Convergence threshold
batch_size = 100                                    # List of batch sizes to run
validation_site = (500, 400)                        # Pixel to evaluate (598, 1092)
crop = None                                         # (y1, y2, x1, x2)
h1 = np.array([1, 0, -1])                           # First-order central difference kernel
h2 = np.array([1, -2, 1])                           # Second-order central difference kernel
h3 = np.array([1, -3, 3, -1])                       # Third-order central difference kernel
h4 = np.array([1, -4, 6, -4, 1])                    # Fourth-order central difference kernel
kernels = {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4}  # Dictionary of kernels
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

# Plot 1: Load and plot ToA reflectance data and validation site
bands, R_TOA_uncropped, sat_mask = load_data(file_path)
R_TOA_uncropped = apply_saturation_mask(R_TOA_uncropped, sat_mask)
(y,x) = R_TOA_uncropped.shape[:2]
scale = 8*1e-3
plt.figure(figsize=(x*scale, y*scale))
plt.imshow(np.nan_to_num(get_rgb(R_TOA_uncropped, bands), nan = 0))
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
plt.scatter(validation_site[1], validation_site[0], color='red', s=80, marker='*', label='Validation site')
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
    plt.figure(figsize=(x*scale, y*scale))
    key = f'b{batch_size}_{h}'
    rgb_boa = get_rgb(R_BOA[key], bands, ref_image=None)
    plt.imshow(np.nan_to_num(rgb_boa, nan = 0))
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
    ax1.set_title(f"T and S for {h}")
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    # Create right y-axis for S
    ax2 = ax1.twinx()
    ax2.plot(bands, S[key], color=color_s, label=f"S ({h})", linewidth=1)
    ax2.set_ylabel("S [-]", color=color_s)
    ax2.tick_params(axis='y', labelcolor=color_s)
    ax2.set_ylim(0, global_max_S)
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)
plt.tight_layout()
plt.savefig(f"{save_folder}/final_S_T.png", dpi=300)

# Plot 4: TOA, BOA and ground truth at validation site
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot (eval_pixel1)
y_max = np.max([np.max(R_TOA[eval_pixel1]), np.max(R_TOA[eval_pixel2])])
axs[0].plot(bands, R_TOA[eval_pixel1], label="ToA", linewidth=1, color="green")
axs[0].plot(bands, R_BOA_b100_h1[eval_pixel1], color = "blue", label="BoA: h1", linewidth=1)
axs[0].plot(bands, R_BOA_b100_h2[eval_pixel1], color = "m", label="BoA: h2", linewidth=1)
axs[0].set_xlabel("Band [nm]")
axs[0].set_ylabel("Reflectance [-]")
axs[0].legend()
axs[0].set_title("Evaluation Pixel 1")
axs[0].grid(True)
axs[0].set_ylim(0, y_max)
# Second subplot (eval_pixel2)
axs[1].plot(bands, R_TOA[eval_pixel2], label="ToA", linewidth=1, color="red")
axs[1].plot(bands, R_BOA_b100_h1[eval_pixel2], color = "blue", label="BoA: h1", linewidth=1)
axs[1].plot(bands, R_BOA_b100_h2[eval_pixel2], color = "m", label="BoA: h2", linewidth=1)
axs[1].set_xlabel("Band [nm]")
axs[1].set_ylabel("Reflectance [-]")
axs[1].legend()
axs[1].set_title("Evaluation Pixel 2")
axs[1].grid(True)
axs[1].set_ylim(0, y_max)
plt.tight_layout()
plt.savefig(f"{save_folder}/BoA_ToA_b100_h1h2_p1p2.png", dpi=300)


# Plot 5: BOA at validation site (water spectrum)
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot (eval_pixel1)
#axs[0].plot(bands[:58], R_TOA[eval_pixel1][:58], label="ToA", linewidth=1, color="green")
y_max = np.max(np.concatenate((R_BOA_b100_h1[eval_pixel1][:58], R_BOA_b100_h2[eval_pixel1][:58], 
                              R_BOA_b100_h1[eval_pixel2][:58], R_BOA_b100_h2[eval_pixel2][:58]))) * 1.1
axs[0].plot(bands[:58], R_BOA_b100_h1[eval_pixel1][:58], color = "blue", label="BoA: h1", linewidth=1)
axs[0].plot(bands[:58], R_BOA_b100_h2[eval_pixel1][:58], color = "m", label="BoA: h2", linewidth=1)
axs[0].set_xlabel("Band [nm]")
axs[0].set_ylabel("Reflectance [-]")
axs[0].legend()
axs[0].set_title("Evaluation Pixel 1")
axs[0].grid(True)
axs[0].set_ylim(0, y_max)
# Second subplot (eval_pixel2)
#axs[1].plot(bands[:58], R_TOA[eval_pixel2][:58], label="ToA", linewidth=1, color="red")
axs[1].plot(bands[:58], R_BOA_b100_h1[eval_pixel2][:58], color = "blue", label="BoA: h1", linewidth=1)
axs[1].plot(bands[:58], R_BOA_b100_h2[eval_pixel2][:58], color = "m", label="BoA: h2", linewidth=1)
axs[1].set_xlabel("Band [nm]")
axs[1].set_ylabel("Reflectance [-]")
axs[1].legend()
axs[1].set_title("Evaluation Pixel 2")
axs[1].grid(True)
axs[1].set_ylim(0, y_max)
plt.tight_layout()
plt.savefig(f"{save_folder}/BoA_b100_h1h2_p1p2_water.png", dpi=300)