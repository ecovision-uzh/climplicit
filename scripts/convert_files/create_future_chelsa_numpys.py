# This script loads the monthly CHELSA rasters and turns them into 12 monthly stacked files

import rioxarray
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from global_land_mask import globe

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
# months = ["01", "02"]
var_names = ["pr", "tas", "tasmax", "tasmin"]
# var_names = ["pr"]
time_frames = ["current", "2011_2040", "2041_2070", "2071_2100"]
# time_frames = ["2011_2040"]
# time_frames = ["current"]
ssps = ["ssp126", "ssp370", "ssp585"]
# ssps = ["ssp126"]

CHELSA_DIR = "/shares/wegner.ics.uzh/CHELSA/"
climatology_dir = CHELSA_DIR + "Future_Climatologies/tifs/"
result_path = CHELSA_DIR + "Future_Climatologies/"
try:
    os.mkdir(result_path)
except:
    pass

refras = rioxarray.open_rasterio(
    climatology_dir
    + "CHELSA_gfdl-esm4_r1i1p1f1_w5e5_ssp585_tasmin_07_2071_2100_norm.tif",
    cache=False,
)
ys = refras.y
xs = refras.x
try:
    land_mask = np.load(result_path + "lsm.npy")
except:
    y_grid, x_grid = np.meshgrid(ys, xs)
    land_mask = globe.is_land(y_grid, x_grid).astype(bool)
    np.save(result_path + "lsm.npy", land_mask)
    land_mask = land_mask

land_mask = land_mask.T.astype(bool)
land_mask[ys < -63] = (
    False  # CHELSA has no values below this point even though there is landmass
)

# These were calculated by taking mean/std across current-climate all months
var_mean = np.array(
    [[[688.4960482]], [[2817.54503453]], [[2862.30705101]], [[2771.25053521]]]
)
var_std = np.array(
    [[[875.89886103]], [[176.02514265]], [[183.50181411]], [[168.30246239]]]
)
all_months = []
for month in months:
    print("Month:", month)
    all_tfs = []
    for tf in tqdm(time_frames):
        all_ssps = []
        for ssp in ssps:
            all_vars = []
            for var in var_names:
                if tf == "current":
                    raster_file_name = (
                        climatology_dir
                        + "CHELSA_"
                        + var
                        + "_"
                        + str(month)
                        + "_1981-2010_V.2.1.tif"
                    )
                else:
                    # CHELSA_gfdl-esm4_r1i1p1f1_w5e5_ssp585_tasmin_07_2071_2100_norm.tif
                    raster_file_name = (
                        climatology_dir
                        + "CHELSA_gfdl-esm4_r1i1p1f1_w5e5_"
                        + ssp
                        + "_"
                        + var
                        + "_"
                        + month
                        + "_"
                        + tf
                        + "_norm.tif"
                    )
                ras = rioxarray.open_rasterio(raster_file_name, cache=False)
                all_vars.append(np.array(ras)[0])
            all_vars = np.stack(all_vars)

            all_vars = (all_vars - var_mean) / var_std
            all_ssps.append(all_vars.astype(np.float16))

        all_ssps = np.stack(all_ssps)
        all_tfs.append(all_ssps)
    all_tfs = np.stack(all_tfs)

    with open(result_path + month + "_monthly_float16.npy", "wb") as f:
        np.save(f, all_tfs)

    with open(result_path + month + "_monthly_float16_land_only.npy", "wb") as f:
        np.save(f, all_tfs[:, :, :, land_mask])

"""   
    all_months.append(all_tfs)
all_months = np.stack(all_months)

var_mean = all_months.mean(axis=(0,1,2,4))
var_std = all_months.std(axis=(0,1,2,4))
print("Means:", var_mean)
print("Stds:", var_std)
"""

# res = np.stack(float16_nps)
# print("Final size:", res.shape)

# res = res[:, ~lsm]
# res = res[:, locs[:, 0], locs[:, 1]]
# with open(result_path + month + "_.npy", "wb") as f:
# np.save(f, res)
