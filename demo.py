"""

Demo of the MACHI algorithm: Minimal Atmospheric Correction for Hyperspectral Imagers.
For MSc thesis, spring 2025. See thesis report for details. 
Author: Edvin Dannäs
Date: 2025-06-17

"""
import numpy as np
import matplotlib.pyplot as plt
from main import main
from utils import get_rgb
import os

# --------------- INPUT -----------------
file = "data/HYPSO-1/ToA-reflectance/mjosa_2024-11-28T10-35-01Z-l1d.npz"  # Input file with reflectance data
file = "data/HYPSO-2/ToA-reflectance/aeronetvenice_2025-03-04T10-38-05Z-l1d.npz" 

maxIter = 200                       # Maximum number of iterations 
epsilon = 1e-2                      # Convergence threshold
# batch_size = None                   # If not None, process data in batches of this size
eval_pixel1 = (400, 500)                # Pixel to evaluate (598, 1092)
eval_pixel1 = (310, 780)                # Pixel to evaluate (598, 1092) (302, 546)
eval_pixel2 = (500, 800)              # Pixel to evaluate (598, 1092)

crop = (360, 375, 550, 580)         # Mjøsa
crop = (250, 350, 500, 800)         # Venice
crop = (200, 400, 400, 900)         # Venice
crop = None         # Venice
# crop = (300, 330, 500, 600)         # Venice
h1 = np.array([1, 0, -1])           # First-order central difference kernel
h2 = np.array([1, -2, 1])           # Second-order central difference kernel
# h = np.array([1, -3, 3, -1])        # Third-order central difference kernel
# ---------------------------------------

# Create save folder from file path
satellite = file.split("/")[1]
site = file.split("/")[3].replace("-l1d.npz", "")
save_folder = f"output/{satellite}/{site}"
os.makedirs(save_folder, exist_ok=True)

# Run MACHI
R_TOA_uncropped, R_TOA, R_BOA_b100_h1, S_b100_h1, T_b100_h1, bands, i_b100_h1, t_b100_h1, p_b100_h1 = main(
    file, maxIter, epsilon, 100, h1, crop = crop
    )
R_TOA_uncropped, R_TOA, R_BOA_b100_h2, S_b100_h2, T_b100_h2, bands, i_b100_h2, t_b100_h2, p_b100_h2 = main(
    file, maxIter, epsilon, 100, h2, crop = crop
    )
R_TOA_uncropped, R_TOA, R_BOA_b200_h1, S_b200_h1, T_b200_h1, bands, i_b200_h1, t_b200_h1, p_b200_h1 = main(
    file, maxIter, epsilon, 200, h1, crop = crop
    )
R_TOA_uncropped, R_TOA, R_BOA_b200_h2, S_b200_h2, T_b200_h2, bands, i_b200_h2, t_b200_h2, p_b200_h2 = main(
    file, maxIter, epsilon, 200, h2, crop = crop
    )
R_TOA_uncropped, R_TOA, R_BOA_b500_h1, S_b500_h1, T_b500_h1, bands, i_b500_h1, t_b500_h1, p_b500_h1 = main(
    file, maxIter, epsilon, 500, h1, crop = crop
    )
R_TOA_uncropped, R_TOA, R_BOA_b500_h2, S_b500_h2, T_b500_h2, bands, i_b500_h2, t_b500_h2, p_b500_h2 = main(
    file, maxIter, epsilon, 500, h2, crop = crop
  )
# R_TOA_uncropped, R_TOA, R_BOA_b1000_h1, S_b1000_h1, T_b1000_h1, bands, i_b1000_h1, t_b1000_h1, p_b1000_h1 = main(
#    file, maxIter, epsilon, 1000, h1, crop = crop
#    )
# R_TOA_uncropped, R_TOA, R_BOA_b1000_h2, S_b1000_h2, T_b1000_h2, bands, i_b1000_h2, t_b1000_h2, p_b1000_h2 = main(
#    file, maxIter, epsilon, 1000, h2, crop = crop
#  )
R_TOA_uncropped, R_TOA, R_BOA_b10000_h1, S_b10000_h1, T_b10000_h1, bands, i_b10000_h1, t_b10000_h1, p_b10000_h1 = main(
   file, maxIter, epsilon, 1000, h1, crop = crop
   )
R_TOA_uncropped, R_TOA, R_BOA_b10000_h2, S_b10000_h2, T_b10000_h2, bands, i_b10000_h2, t_b10000_h2, p_b10000_h2 = main(
   file, maxIter, epsilon, 1000, h2, crop = crop
 )
# R_TOA_uncropped, R_TOA, R_BOA_bAll_h1, S_bAll_h1, T_bAll_h1, bands, i_bAll_h1, t_bAll_h1, p_bAll_h1 = main(
#     file, maxIter, epsilon, All, h1, crop = crop
#     )
# R_TOA_uncropped, R_TOA, R_BOA_bAll_h2, S_bAll_h2, T_bAll_h2, bands, i_bAll_h2, t_bAll_h2, p_bAll_h2 = main(
#     file, maxIter, epsilon, All, h2, crop = crop
#     )

print("Plotting results...")
print(1)
# 1: rgb plot of R_TOA and cropping window
plt.figure(figsize=(10, 5))
plt.imshow(np.nan_to_num(get_rgb(R_TOA_uncropped, bands), nan = 0))
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
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{save_folder}/RGB_TOA_crop.png", dpi=300)

print(2)
# 2: rgb plot of R_TOA and R_BOA, h1 b100
rgb_toa = get_rgb(R_TOA, bands, ref_image=R_TOA_uncropped)#R_TOA_uncropped
rgb_boa = get_rgb(R_BOA_b100_h1, bands, ref_image=None)#R_TOA_uncropped
#rgb_toa_norm = normalize_rgb(rgb_toa)
#rgb_boa_norm = normalize_rgb(rgb_boa, rgb_toa)
plt.figure(figsize=(7, 5))
plt.subplot(1, 2, 1)
plt.title("R_TOA RGB")
plt.imshow(np.nan_to_num(rgb_toa, nan = 0),)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("R_BOA RGB")
plt.imshow(np.nan_to_num(rgb_boa, nan = 0))
plt.axis("off")
plt.tight_layout()
plt.scatter(eval_pixel1[1], eval_pixel1[0], color='green', s=40, marker='o', label='eval_pixel1')
plt.scatter(eval_pixel2[1], eval_pixel2[0], color='red', s=40, marker='o', label='eval_pixel2')
plt.savefig(f"{save_folder}/RGB_TOA_BOA_h1.png", dpi=300)

print(3)
# 3: convergence plots for each kernel and all batch sizes
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# First subplot (h1 kernel)
axs[0].plot(np.arange(len(p_b100_h1)) / 2 * 100, p_b100_h1 / 100, color = "blue", label="h1 - Batch size 100", linewidth=1)
axs[0].plot(np.arange(len(p_b200_h1)) / 2 * 200, p_b200_h1 / 200, color = "blue", linestyle = "dashed", label="h1 - Batch size 200", linewidth=1)
#axs[0].plot(np.arange(len(p_b500_h1)) / 2 * 500, p_b500_h1 / 500, color = "blue", linestyle = "dashdot", label="h1 - Batch size 500", linewidth=1)
# axs[0].plot(np.arange(len(p_b1000_h1)) / 2 * 1000, p_b1000_h1 / 1000, color = "blue", linestyle = "dotted", label="h1 - Batch size 1000", linewidth=1)
axs[0].set_yscale("log")
axs[0].set_ylabel("Penalty / Batch size")
axs[0].legend()
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
# Second subplot (h2 kernel)
axs[1].plot(np.arange(len(p_b100_h2)) / 2 * 100, p_b100_h2 / 100, color = "m", label="h2 - Batch size 100", linewidth=1)
axs[1].plot(np.arange(len(p_b200_h2)) / 2 * 200, p_b200_h2 / 200, color = "m", linestyle = "dashed", label="h2 - Batch size 200", linewidth=1)
#axs[1].plot(np.arange(len(p_b500_h2)) / 2 * 500, p_b500_h2 / 500, color = "m", linestyle = "dashdot", label="h2 - Batch size 500", linewidth=1)
# axs[1].plot(np.arange(len(p_b1000_h2)) / 2 * 1000, p_b1000_h2 / 1000, color = "m", linestyle = "dotted", label="h2 - Batch size 1000", linewidth=1)
axs[1].set_yscale("log")
axs[1].set_xlabel("Iteration × Batch size")
axs[1].set_ylabel("Penalty / Batch size")
axs[1].legend()
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{save_folder}/convergence_batches_h1h2.png", dpi=300)

print(4)
# 4: BOA plots for each kernel and all batch sizes
plt.figure(figsize=(10, 8))
# plt.plot(bands, R_TOA[eval_pixel1], label="ToA", linewidth=1, color="green")
plt.plot(bands, R_BOA_b100_h1[eval_pixel1], color = "blue", label="BoA: h1, batch size 100", linewidth=1)
plt.plot(bands, R_BOA_b200_h1[eval_pixel1], color = "blue", linestyle = "dashed", label="BoA: h1, batch size 200", linewidth=1)
#plt.plot(bands, R_BOA_b500_h1[eval_pixel1], color = "blue", linestyle = "dashdot", label="BoA: h1, batch size 500", linewidth=1)
# plt.plot(bands, R_BOA_b1000_h1[eval_pixel1], color = "blue", linestyle = "dotted", label="BoA: h1, batch size 1000", linewidth=1)
plt.plot(bands, R_BOA_b100_h2[eval_pixel1], color = "m", label="BoA: h2, batch size 100", linewidth=1)
plt.plot(bands, R_BOA_b200_h2[eval_pixel1], color = "m", linestyle = "dashed", label="BoA: h2, batch size 200", linewidth=1)
#plt.plot(bands, R_BOA_b500_h2[eval_pixel1], color = "m", linestyle = "dashdot", label="BoA: h2, batch size 500", linewidth=1)
# plt.plot(bands, R_BOA_b1000_h2[eval_pixel1], color = "m", linestyle = "dotted", label="BoA: h2, batch size 1000", linewidth=1)
plt.xlabel("Band [nm]")
plt.ylabel("Reflectance [-]")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_folder}/BoA_batches_h1h2_p1.png", dpi=300)

print(5)
# 5: BOA plots for each kernel and selected batch size
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

print(6)
# 6: BOA plots for each kernel and selected batch size without ToA
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# First subplot (eval_pixel1)
y_max = np.max(np.concatenate((R_BOA_b100_h1[eval_pixel1], R_BOA_b100_h2[eval_pixel1], 
                               R_BOA_b100_h1[eval_pixel2], R_BOA_b100_h2[eval_pixel2]))) * 1.1
#axs[0].plot(bands, R_TOA[eval_pixel1], label="ToA", linewidth=1, color="green")
axs[0].plot(bands, R_BOA_b100_h1[eval_pixel1], color = "blue", label="BoA: h1", linewidth=1)
axs[0].plot(bands, R_BOA_b100_h2[eval_pixel1], color = "m", label="BoA: h2", linewidth=1)
axs[0].set_xlabel("Band [nm]")
axs[0].set_ylabel("Reflectance [-]")
axs[0].legend()
axs[0].set_title("Evaluation Pixel 1")
axs[0].grid(True)
axs[0].set_ylim(0, y_max)
# Second subplot (eval_pixel2)
#axs[1].plot(bands, R_TOA[eval_pixel2], label="ToA", linewidth=1, color="red")
axs[1].plot(bands, R_BOA_b100_h1[eval_pixel2], color = "blue", label="BoA: h1", linewidth=1)
axs[1].plot(bands, R_BOA_b100_h2[eval_pixel2], color = "m", label="BoA: h2", linewidth=1)
axs[1].set_xlabel("Band [nm]")
axs[1].set_ylabel("Reflectance [-]")
axs[1].legend()
axs[1].set_title("Evaluation Pixel 2")
axs[1].grid(True)
axs[1].set_ylim(0, y_max)
plt.tight_layout()
plt.savefig(f"{save_folder}/BoA_b100_h1h2_p1p2.png", dpi=300)

print(7)
# 7: BOA plots for each kernel and selected batch size, water spectrum
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

print(8)
# 8: Plot S and T for both kernels
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # sharex='col', sharey='row'
# Row 1, scattering S
axs[0].plot(bands, S_b100_h1, label="S: h1", color = "blue", linewidth=1)
axs[0].plot(bands, S_b100_h2, label="S: h2", color = "m", linewidth=1)
axs[0].set_title("Atm. path reflectance (S)")
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(0, max(np.max(S_b100_h1), np.max(S_b100_h2)) * 1.1)
# Row 2, transmission T
axs[1].plot(bands, T_b100_h1, label="T: h1", color = "blue", linewidth=1)
axs[1].plot(bands, T_b100_h2, label="T: h2", color = "m", linewidth=1)
axs[1].set_xlabel("Band [nm]")
axs[1].set_title("Transmission (T)")
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{save_folder}/S_T_h1h2.png", dpi=300)