import numpy as np

def load_data(filename):
    """Load hyperspectral data from file."""
    if filename.lower().endswith(".npz"):
        data = np.load(filename)
        bands = data["bands"]     # shape (N,), numpy.ndarray
        R_TOA = data["cube"]      # shape (Ix, Iy, N), numpy.ndarray
        sat_mask = data.get("sat_mask", None)  # Optional saturation mask
        print(f"Loaded data from {filename}: bands shape {bands.shape}, R_TOA shape {R_TOA.shape}")
        return bands, R_TOA, sat_mask
    else:
        raise ValueError("Unsupported file format. Please provide a .npz file.")

def apply_saturation_mask(image, sat_mask):
    """Apply saturation mask to image."""
    image_masked = image.copy()  # Create a copy of the image to avoid modifying the original
    if sat_mask is not None:
        image_masked[sat_mask] = np.nan  # Set saturated pixels to zero
        print(f"Applied saturation mask. Number of saturated pixels: {np.sum(sat_mask)}")
    else:
        print("No saturation mask found. Using original reflectance data.")

    return image_masked  # Return a copy of the image with saturation applied

def initialize_machi_params(h, R_TOA):
    """Initialize parameters for the MACHI algorithm."""

    # Flatten spatial dimension of R_TOA
    Ix, Iy, N = R_TOA.shape
    I = Ix * Iy
    R_TOA_flat = R_TOA.reshape(I, N)
    
    # Remove rows with NaNs from R_TOA_flat
    nan_mask = ~np.isnan(R_TOA_flat).any(axis=1)
    R_TOA_flat = R_TOA_flat[nan_mask]
    I = R_TOA_flat.shape[0]

    h = h / np.sum(np.abs(h))           # Normalize kernel
    L = len(h)                          # Length of the kernel
    S = np.min(R_TOA_flat, axis=0)      # Dark pixel subtraction
    tilde_T = S / (1 - S)
    tilde_h = np.flip(h)                # Define flipped kernel

    # Print initial data
    print(f"Initialized parameters: h shape {h.shape}, L {L}, S shape {S.shape}, tilde_T shape {tilde_T.shape}, tilde_h shape {tilde_h.shape}, R_TOA_flat shape {R_TOA_flat.shape}, I {I}, N {N}")
    return h, L, S, tilde_T, tilde_h, R_TOA_flat, I, N

def crop_image(image, crop):
    """Crop the image to the specified region."""
    if crop is not None:
        y_start, y_stop, x_start, x_stop = crop
        image_cropped = image[y_start:y_stop, x_start:x_stop, :]
        print(f"Cropping image to region: y({y_start}:{y_stop}), x({x_start}:{x_stop})")
        print(f"Cropped image shape: {image_cropped.shape}")
        return image_cropped
    else:
        print("No cropping applied.")
        return image.copy()

def get_rgb(image, bands, ref_image = None, rgb_bands=(630, 550, 480)): #rgb_bands=(670, 550, 470)
    """ Create RGB array from hyperspectral image using specified bands."""
    # Find closest band indices for R, G, B bands
    idx_r = np.argmin(np.abs(bands - rgb_bands[0]))
    idx_g = np.argmin(np.abs(bands - rgb_bands[1]))
    idx_b = np.argmin(np.abs(bands - rgb_bands[2]))

    rgb = np.stack([
        image[..., idx_r],
        image[..., idx_g],
        image[..., idx_b]
    ], axis=-1)

    # Normalize
    if ref_image is None:
        ref_rgb = rgb
    else:
        ref_rgb = np.stack([
        ref_image[..., idx_r],
        ref_image[..., idx_g],
        ref_image[..., idx_b]
    ], axis=-1)
    rgb_min = np.nanpercentile(rgb, 1)
    rgb_ref_min = np.nanpercentile(ref_rgb, 1)
    rgb_ref_max = np.nanpercentile(ref_rgb, 99)
    rgb_norm = np.clip((rgb-rgb_min) / (rgb_ref_max - rgb_ref_min), 0, 1)
    return rgb_norm.astype(np.float32)

def create_batch(R_TOA_flat, batch_size, I):
    """Create a random batch of data from the flattened R_TOA."""
    if batch_size is None or batch_size >= I:
        return R_TOA_flat, I
    else:
        batch_row_index = np.random.choice(I, size=batch_size, replace=False)
        return R_TOA_flat[batch_row_index, :], batch_size
    
def penalty(R_TOA_flat, S, tilde_T, h, I):
    """Calculate the penalty term."""
    penalty_value = 0
    for i in range(I):
        R_BOA_i = (R_TOA_flat[i] - S) * (1 + tilde_T)
        c_ji = np.convolve(R_BOA_i, h, mode='valid')
        penalty_value += np.sum(c_ji**2)
    return penalty_value

def update_S(R_TOA_flat, I, N, S, tilde_T, tilde_h, L):
    """Update function for the scattering term S."""
    for n in range(N):
        l_hat = max(n - L + 1, 0)
        u_hat = min(n, N - L)
        denom = (1 + tilde_T[n]) * I * sum(tilde_h[n-j]**2 for j in range(l_hat, u_hat + 1))

        num = 0
        for i in range(I):
            R_BOA_i = (R_TOA_flat[i] - S) * (1 + tilde_T)
            for j in range(l_hat, u_hat + 1):
                sum_k = sum(
                    R_BOA_i[j+k] * tilde_h[k]
                    for k in range(L)
                    if k != n-j
                )
                num += tilde_h[n-j] * sum_k
        
        for j in range(l_hat, u_hat + 1):
            num += (1 + tilde_T[n]) * sum(R_TOA_flat[:, n]) * tilde_h[n-j]**2

        S[n] = num / denom
        S[n] = min(S[n], np.min(R_TOA_flat[:, n]))  # Projection onto feasible set

def update_tildeT(R_TOA_flat, I, N, S, tilde_T, tilde_h, L):
    """Update function for the transmission term tilde_T."""
    for n in range(N):
        l_hat = max(n - L + 1, 0)
        u_hat = min(n, N - L)

        denom = 0
        for i in range(I):
            denom += (R_TOA_flat[i, n] - S[n])**2
        denom *= sum(tilde_h[n-j]**2 for j in range(l_hat, u_hat+1))

        num = 0
        for i in range(I):
            R_BOA_i = (R_TOA_flat[i] - S) * (1 + tilde_T)
            
            diff = R_TOA_flat[i, n] - S[n]
            for j in range(l_hat, u_hat+1):
                sum_k = sum(
                    R_BOA_i[j+k] * tilde_h[k]
                    for k in range(L)
                    if k != n-j
                )
                num -= diff * tilde_h[n-j] * sum_k
        for i in range(I):
            diff = R_TOA_flat[i, n] - S[n]
            num -= diff**2 * sum(tilde_h[n-j]**2 for j in range(l_hat, u_hat+1))

        tilde_T[n] = num / denom
        tilde_T[n] = max(tilde_T[n], 0) # Projection onto feasible set