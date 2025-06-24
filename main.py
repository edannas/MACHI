"""
Implementation of the MACHI algorithm: Minimal Atmospheric Correction for Hyperspectral Imagers.
For MSc thesis, spring 2025. See thesis report for details. 
Author: Edvin Dannäs
Date: 2025-06-17
"""

import numpy as np
import time
from utils import load_data, apply_saturation_mask, initialize_machi_params, crop_image, create_batch, penalty, update_S, update_tildeT

def main(filename, h, batch_size=None, crop=None, maxIter=200, epsilon=1e-2):
    """Main function to run the MACHI algorithm."""
    if h is None:
        raise ValueError("Error: No kernel 'h' provided.")
    # Load data
    bands, R_TOA_uncropped, sat_mask = load_data(filename)

    # Apply saturation mask
    R_TOA_uncropped = apply_saturation_mask(R_TOA_uncropped, sat_mask)

    # Crop image
    R_TOA = crop_image(R_TOA_uncropped, crop)

    # Initialize parameters
    h, L, S, tilde_T, tilde_h, R_TOA_flat, I, N = initialize_machi_params(h, R_TOA)

    # Run algorithm
    time_start = time.time()
    penalties = []
    for iteration in range(maxIter):
        # Batching
        R_TOA_flat_batch, batch_size = create_batch(R_TOA_flat, batch_size, I)

        # Calculate penalty before update
        P_prev = penalty(R_TOA_flat_batch, S, tilde_T, tilde_h, batch_size)

        # Step 1: Update S[n]
        update_S(R_TOA_flat_batch, batch_size, N, S, tilde_T, tilde_h, L)

        # Step 2: Update tilde_T[n]
        update_tildeT(R_TOA_flat_batch, batch_size, N, S, tilde_T, tilde_h, L)

        # Calculate penalty after update
        P_after = penalty(R_TOA_flat_batch, S, tilde_T, tilde_h, batch_size)
        penalties.extend([P_prev, P_after])

        # Step 3: Check for convergence
        delta_P = (P_prev - P_after) / (P_prev + P_after)
        print("iteration {:<6} P_before: {:<25} P_after: {:<25} dP: {}".format(iteration, P_prev, P_after, delta_P))
        
        if delta_P < epsilon:
            break
    else:
        print("Maximum iterations reached without convergence.")

    # Print run time
    time_end = time.time()
    time_elapsed = time_end - time_start
    print(f"Run time: {time_elapsed:.2f} seconds")

    # Output results
    R_BOA = (R_TOA - S) * (1 + tilde_T)
    T = 1 / (1 + tilde_T)

    return R_TOA, R_BOA, S, T, bands, iteration+1, time_elapsed, np.array(penalties)