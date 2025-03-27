# This script loads the monthly CHELSA rasters and turns them into 12 monthly stacked files

import rioxarray
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
# months = ["03"]
var_names = [
    "cmi",
    "clt",
    "hurs",
    "pet",
    "pr",
    "rsds",
    "sfcWind",
    "tas",
    "tasmax",
    "tasmin",
    "vpd",
]
# var_names = ["cmi"]

CHELSA_DIR = "/shares/wegner.ics.uzh/CHELSA/"
climatology_dir = CHELSA_DIR + "climatologies/1981-2010/"
result_path = CHELSA_DIR + "climatologies/1981-2010_numpy/"
try:
    os.mkdir(result_path)
except:
    pass

# land_coordinates_file = "/shares/wegner.ics.uzh/CHELSA/input/land_coordinates.npy"
# locs = np.load(land_coordinates_file)

mean = np.array(
    [
        [-264.1493656],
        [3912.44628016],
        [5921.65964573],
        [9385.47468266],
        [697.03653109],
        [15219.37926928],
        [3498.8511804],
        [2819.56006368],
        [2864.08583811],
        [2773.46759638],
        [8039.37322797],
    ]
)
std = np.array(
    [
        [1042.67560332],
        [1767.94018571],
        [1185.91587823],
        [6639.79069994],
        [883.56243405],
        [7843.49167037],
        [1637.09237995],
        [174.43791946],
        [181.69448751],
        [167.07485901],
        [7516.98198719],
    ]
)

all_months = []
for month in months:
    arrays = []
    raster = {}
    for var in var_names:
        if var == "pet":
            raster_file_name = (
                climatology_dir
                + var
                + "/CHELSA_pet_penman_"
                + str(month)
                + "_1981-2010_V.2.1.tif"
            )
        elif var == "rsds":
            raster_file_name = (
                climatology_dir
                + var
                + "/CHELSA_"
                + var
                + "_1981-2010_"
                + str(month)
                + "_V.2.1.tif"
            )
        else:
            raster_file_name = (
                climatology_dir
                + var
                + "/CHELSA_"
                + var
                + "_"
                + str(month)
                + "_1981-2010_V.2.1.tif"
            )

        raster[var] = rioxarray.open_rasterio(raster_file_name, cache=False)
        if var == "clt":
            raster[var] = raster[var].reindex_like(raster["cmi"], method="nearest")
    lsm = np.array(raster["cmi"])[0] > 30000
    # print(lsm.sum(), "values being masked in month", month)
    print("Month:", month)

    def convert_to_float16_np(name, i):
        as_np = raster[name][0].to_numpy()
        as_np = as_np - mean[i, 0]
        as_np = as_np / std[i, 0]
        return as_np.astype("float16")
        # return as_np[~lsm]

    float16_nps = []
    for i in tqdm(range(len(var_names))):
        var = var_names[i]
        float16_nps.append(convert_to_float16_np(var, i))

    res = np.stack(float16_nps)
    print("Final size:", res.shape)
    with open(result_path + month + "_monthly_float16.npy", "wb") as f:
        np.save(f, res)

    res = res[:, ~lsm]
    print("Normalized stats:")
    print(res.mean(1))
    print("Just land shape:", res.shape)
    with open(result_path + month + "_monthly_float16_land_only.npy", "wb") as f:
        np.save(f, res)


"""all_months.append(res)
all_months = np.stack(all_months)
print(all_months.shape)
print(np.expand_dims(all_months.mean((0,2)), (0,2)))
print(np.expand_dims(all_months.std((0,2)), (0,2)))"""

""""""

# res = res[:, ~lsm]
# res = res[:, locs[:, 0], locs[:, 1]]
"""print("Just land shape:", res.shape)
with open(result_path + month + "_monthly_float16_land_only.npy", "wb") as f:
    np.save(f, res)"""
