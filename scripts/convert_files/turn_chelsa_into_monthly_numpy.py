# This script loads the monthly CHELSA rasters and turns them into 12 monthly stacked files

import rioxarray
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
#months = ["03"]
var_names = ["cmi",  "hurs", "pet", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin", "vpd"]
var_names = ["cmi", "clt", "hurs", "pet", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin", "vpd"]
#var_names = ["cmi"]

CHELSA_DIR = "/shares/wegner.ics.uzh/CHELSA/"
climatology_dir = CHELSA_DIR + "climatologies/1981-2010/"
result_path = CHELSA_DIR + "climatologies/1981-2010_numpy/"
try:
    os.mkdir(result_path)
except:
    pass

land_coordinates_file = "/shares/wegner.ics.uzh/CHELSA/input/land_coordinates.npy"
locs = np.load(land_coordinates_file)

for month in months:
    arrays = []
    raster = {}
    for var in var_names:
        if var == "pet":
            raster_file_name = climatology_dir + var + "/CHELSA_pet_penman_" + str(month) + "_1981-2010_V.2.1.tif"
        elif var == "rsds":
            raster_file_name = climatology_dir + var + "/CHELSA_" + var + "_1981-2010_" + str(month) + "_V.2.1.tif"
        else:
            raster_file_name = climatology_dir + var + "/CHELSA_" + var + "_" + str(month) + "_1981-2010_V.2.1.tif"

        raster[var] = rioxarray.open_rasterio(raster_file_name, cache=False)
        if var == "clt":
            raster[var] = raster[var].reindex_like(raster["cmi"], method='nearest')
    lsm = np.array(raster["cmi"])[0]>30000
    print(lsm.sum(), "values being masked in month", month)

    def convert_to_float16_np(name):
        as_np = np.ma.masked_array(raster[name][0].to_numpy(), lsm)
        mean = as_np.mean()
        std = as_np.std()
        as_np = raster[name][0].to_numpy()
        as_np = as_np - mean
        as_np = as_np / std
        return as_np.astype("float16")

    float16_nps = []
    for var in tqdm(var_names):
        float16_nps.append(convert_to_float16_np(var))

    res = np.stack(float16_nps)
    print("Final size:", res.shape)
    with open(result_path + month + "_monthly_float16.npy", "wb") as f:
        np.save(f, res)

    res = res[:, ~lsm]
    #res = res[:, locs[:, 0], locs[:, 1]]
    print("Just land shape:", res.shape)
    with open(result_path + month + "_monthly_float16_land_only.npy", "wb") as f:
        np.save(f, res)




    


    
