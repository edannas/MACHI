from hypso import Hypso1, Hypso2
import os
import matplotlib.pyplot as plt

# dir_path = '/home/cameron/Dokumenter/Data/'
dir_path = '/home/ariaa/smallSatLab/data/'

# HYPSO-1 Capture
# h1_l1a_nc_file = os.path.join(dir_path, 'mvco_2025-01-13T14-57-34Z-l1a.nc')
# h1_points_file = os.path.join(dir_path, 'mvco_2025-01-13T14-57-34Z-l1a.points')
h1_l1a_nc_file = os.path.join(dir_path, 'h1/lacrau_2024-12-26T10-24-27Z/lacrau_2024-12-26T10-24-27Z-l1a.nc')
h1_points_file = os.path.join(dir_path, 'h1/lacrau_2024-12-26T10-24-27Z/lacrau_2024-12-26T10-24-27Z-bin3.points')

# HYPSO-2 Capture
# h2_l1a_nc_file = os.path.join(dir_path, 'mvco_2025-01-08T15-59-40Z-l1a.nc')
h2_l1a_nc_file = os.path.join(dir_path, 'h2/lacrau_2024-12-26T11-15-54Z/lacrau_2024-12-26T11-15-54Z-l1a.nc')

# Load HYPSO-1 Capture
satobj_h1 = Hypso1(path=h1_l1a_nc_file, verbose=True)
# Load HYPSO-2 Capture
satobj_h2 = Hypso2(path=h2_l1a_nc_file, verbose=True)
satobj_h2.generate_l1b_cube()
# Run direct georeferencing using ADCS information
satobj_h1.run_direct_georeferencing()
# Run indirect georeferencing using provided ground control points
satobj_h1.run_indirect_georeferencing(points_file_path=h1_points_file, flip=True)
# Generate L1b TOA radiance product
satobj_h1.generate_l1b_cube()
# Generate L1c geolocated TOA radiance product
satobj_h1.generate_l1c_cube()
# Generate L1d TOA reflectance product
satobj_h1.generate_l1d_cube()

from hypso.write import write_l1b_nc_file, write_l1c_nc_file, write_l1d_nc_file

# Write the generated products to NetCDF. Use the 'datacube=False' argument to write each band as a separate variable in the NetCDF file 
write_l1b_nc_file(satobj=satobj_h1, overwrite=True)
write_l1c_nc_file(satobj=satobj_h1, overwrite=True, datacube=False)
write_l1d_nc_file(satobj=satobj_h1, overwrite=True, datacube=False)

# Access datacubes
l1a_cube = satobj_h1.l1a_cube
l1b_cube = satobj_h1.l1b_cube
l1c_cube = satobj_h1.l1c_cube
l1d_cube = satobj_h1.l1d_cube

band = 40

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

# Plot data on each subplot
axs[0, 0].imshow(l1a_cube[:,:,band])
axs[0, 0].set_title('L1a band ' + str(band))

axs[0, 1].imshow(l1b_cube[:,:,band])
axs[0, 1].set_title('L1b band ' + str(band))

axs[1, 0].imshow(l1c_cube[:,:,band])
axs[1, 0].set_title('L1c band ' + str(band))

axs[1, 1].imshow(l1d_cube[:,:,band])
axs[1, 1].set_title('L1d band ' + str(band))

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()



# Access latitudes and longitudes
latitudes = satobj_h1.latitudes
longitudes = satobj_h1.longitudes
latitudes_indirect = satobj_h1.latitudes_indirect
longitudes_indirect = satobj_h1.longitudes_indirect

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

# Plot data on each subplot
axs[0, 0].imshow(latitudes)
axs[0, 0].set_title('Latitudes (Direct)')

axs[0, 1].imshow(longitudes)
axs[0, 1].set_title('Longitudes (Direct)')

axs[1, 0].imshow(latitudes_indirect)
axs[1, 0].set_title('Latitudes (Indirect)')

axs[1, 1].imshow(longitudes_indirect)
axs[1, 1].set_title('Longitudes (Indirect)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
from hypso.spectral_analysis import get_closest_wavelength_index

# Get wavelengths of capture
satobj_h1.wavelengths

# Get band index of wavelength
red_wl = 630
green_wl = 550
blue_wl = 480

print(get_closest_wavelength_index(satobj_h1, red_wl))
print(get_closest_wavelength_index(satobj_h1, green_wl))
print(get_closest_wavelength_index(satobj_h1, blue_wl))
# Get bounding box of capture
satobj_h1.bbox
from hypso.geometry_definition import generate_area_def

new_bbox = (-71.5, 39.8, -69, 43.1)

area_def = generate_area_def(area_id = 'New area',
                            proj_id = 'id',
                            description = 'new area',
                            bbox = new_bbox,
                            height = 512,
                            width = 512
                            )

# Display area information
area_def
from hypso.resample import resample_l1a_cube, \
                            resample_l1b_cube, \
                            resample_l1c_cube, \
                            resample_l1d_cube

# Resample L1a cube
resampled_l1a_cube, \
resampled_latitudes, \
resampled_longitudes = resample_l1a_cube(satobj = satobj_h1,
                    area_def=area_def,
                    use_indirect_georef=True)

longitudes, latitudes = area_def.get_lonlats()

band = 40

# Create a figure and a 1x3 grid of subplots
fig, axs = plt.subplots(1, 3)

# Plot data on each subplot
axs[0].imshow(resampled_l1a_cube[:,:,band])
axs[0].set_title('Resampled L1a band ' + str(band))

axs[1].imshow(resampled_latitudes)
axs[1].set_title('Resampled latitudes')

axs[2].imshow(resampled_longitudes)
axs[2].set_title('Resampled longitudes')

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
from hypso.resample import resample_dataarray_bilinear, resample_dataarray_kd_tree_nearest

# Resampling functions can also be called directly. You can pass any data as an xarray to be resampled.

# Kd tree nearest resampling (recommended for hyperspectral data)
resampled_l1a_cube_kdtree = resample_dataarray_kd_tree_nearest(area_def = area_def, 
                                        data = satobj_h1.l1a_cube,
                                        latitudes = satobj_h1.latitudes,
                                        longitudes = satobj_h1.longitudes,
                                        radius_of_influence=satobj_h1.resolution)


# Bilinear resampling
resampled_l1a_cube_bilinear = output = resample_dataarray_bilinear(area_def = area_def, 
                                        data = satobj_h1.l1a_cube,
                                        latitudes = satobj_h1.latitudes,
                                        longitudes = satobj_h1.longitudes,
                                        )


from hypso.satpy import get_l1a_satpy_scene, \
                        get_l1b_satpy_scene, \
                        get_l1c_satpy_scene, \
                        get_l1d_satpy_scene

# Export HYPSO capture as a SatPy scene
scene = get_l1a_satpy_scene(satobj=satobj_h1, use_indirect_georef=False)

#scene.show('band_40')