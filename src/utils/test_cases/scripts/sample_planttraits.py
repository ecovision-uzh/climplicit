# imports
import rioxarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import torch
from global_land_mask import globe
import os

import rasterio

file_folder = "/shares/wegner.ics.uzh/Global_trait_maps_Moreno_Martinez_2018_Version2_1km_resolution/"
files = [
    "LDMC_1km_v1.tif",
    "LDMC_sd_1km_v1.tif",
    "LNC_1km_v1.tif",
    "LNC_sd_1km_v1.tif",
    "LPC_1km_v1.tif",
    "LPC_sd_1km_v1.tif",
    "SLA_1km_v1.tif",
    "SLA_sd_1km_v1.tif",
]
# files = ["LDMC_1km_v1.tif"]
ras = []
rio_rasters = []
for file in tqdm(files):
    ras.append(rasterio.open(file_folder + file, cache=False).read())
    rio_rasters.append(rasterio.open(file_folder + file, cache=False))

# lsm = ras[0][0]

# lsm = lsm>0.1 # Only sample locations with actual values
# land = np.where(lsm)
# land = np.transpose(np.stack(land))
# print(land.shape)

SAMPLES = 100000
# sample = np.random.choice(len(land), SAMPLES)
# sample = land[sample]
sample = []

while len(sample) < SAMPLES:
    rand_feats_orig = torch.rand(2)
    theta1 = 2.0 * math.pi * rand_feats_orig[0]
    theta2 = torch.acos(2.0 * rand_feats_orig[1] - 1.0)
    lat = 1.0 - 2.0 * theta2 / math.pi
    lon = (theta1 / math.pi) - 1.0
    lat = lat * 90
    lon = lon * 180
    # Also only take samples outside of Antartica
    y, x = rio_rasters[-1].index(lon.item(), lat.item())
    try:
        if globe.is_land(lat, lon) and lat > -63 and ras[-1][0, y, x] >= 0:
            sample.append((lon.numpy(), lat.numpy()))
    except:
        pass
    if len(sample) % (SAMPLES // 10) == 0:
        print(len(sample))

data = np.array(sample)

# sample_ras = rioxarray.open_rasterio(file_folder+files[0], cache=False)
data = np.zeros((len(sample), 10), dtype="float")
for i in tqdm(range(len(sample))):
    # y, x = sample[i]
    # loc = sample_ras[0, y, x]
    # data[i][0] = loc["x"] # Saving as lon/lat
    # data[i][1] = loc["y"]
    lon, lat = sample[i]
    data[i][0] = lon  # Saving as lon/lat
    data[i][1] = lat
    for j in range(len(files)):
        y, x = rio_rasters[j].index(lon.item(), lat.item())
        data[i][2 + j] = float(ras[j][0, y, x])
np.save(
    "/home/jdolli/chelsaCLIP/src/utils/test_cases/data/planttraits_" + str(SAMPLES),
    data,
)

print(data[:3])

# sample_ras.close()
