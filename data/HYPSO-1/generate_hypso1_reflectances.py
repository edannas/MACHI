import numpy as np
import os
from hypso import Hypso1

# Create absolute path to L1A directory
script_dir = os.path.dirname(__file__)
l1a_path = os.path.join(script_dir, 'L1A/')
l1a_path = os.path.abspath(l1a_path)

for file in os.listdir(l1a_path):
    if not file.endswith('.nc'):
        continue
    satobj_h1 = Hypso1(path=l1a_path+"/"+file, verbose=True)

    # Create saturation mask (boolean mask where True = saturated pixels in any band)
    sensor_limit = 36855
    saturation_threshold = 0.9 * sensor_limit
    l1a_cube_np = np.array(satobj_h1.l1a_cube)
    saturation_mask = np.any(l1a_cube_np > saturation_threshold, axis=-1)

    # Generate reflectance data
    satobj_h1.generate_l1c_cube()
    satobj_h1.generate_l1d_cube()
    l1d_cube = satobj_h1.l1d_cube[:, :, 12:-2]   # shape (Ix, Iy, N)
    bands = satobj_h1.wavelengths[12:-2]         # shape (N,)
    saturation_mask = np.flip(saturation_mask, axis=0)
    l1d_cube = np.flip(l1d_cube, axis=0)
  
    # Save as .npz file
    out_name = os.path.join(script_dir, "ToA-reflectance", file.replace("l1a.nc", "") + "l1d.npz")
    np.savez(out_name, cube=l1d_cube, sat_mask = saturation_mask, bands=bands)