import rasterio
import numpy as np
from rasterio.enums import Resampling

# Sentinel-2 band metadata (excluding B10)
bands_info = [
    ("B01", 443, "R60m"),
    ("B02", 490, "R10m"),
    ("B03", 560, "R10m"),
    ("B04", 665, "R10m"),
    ("B05", 705, "R20m"),
    ("B06", 740, "R20m"),
    ("B07", 783, "R20m"),
    ("B08", 842, "R10m"),
    ("B8A", 865, "R20m"),
    ("B09", 945, "R60m"),
    ("B11", 1610, "R20m"),
    ("B12", 2190, "R20m"),
]

def load_all_boa_bands(base_path, date_tile, resampling='bilinear'):
    target_resolution = "R10m"
    ref_band = "B02"

    # Build paths and wavelength vector
    band_paths = []
    wavelengths = []

    # Open reference band
    ref_path = f"{base_path}/{target_resolution}/{date_tile}_{ref_band}_10m.jp2"
    with rasterio.open(ref_path) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs

    cube = []
    for band_name, wavelength, resolution in bands_info:
        filename = f"{date_tile}_{band_name}_{resolution[1:]}.jp2"
        path = f"{base_path}/{resolution}/{filename}"
        with rasterio.open(path) as src:
            if (src.height, src.width) != ref_shape:
                data = src.read(
                    out_shape=(1, ref_shape[0], ref_shape[1]),
                    resampling=Resampling[resampling]
                )[0]
            else:
                data = src.read(1)
        cube.append(data)
        wavelengths.append(wavelength)

    boa_cube = np.stack(cube, axis=-1)   # (Ix, Iy, N)
    wavelengths = np.array(wavelengths)  # (N,)
    return boa_cube, wavelengths

base_path = "/path/to/GRANULE/L2A_T31UEA_A042795_20250601T110651/IMG_DATA"
date_tile = "T31UEA_20250601T110651"

boa, wavelengths = load_all_boa_bands(base_path, date_tile)
print("BoA cube shape:", boa.shape)        # e.g., (10980, 10980, 12)
print("Wavelengths:", wavelengths)         # e.g., [443 490 560 ... 2190]
