# imports
import rioxarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

import os

import rasterio

file_path = '/home/jdolli/chelsaCLIP/src/utils/test_cases/data/gpw_v4_population_density_rev10_2015_30_sec.tif'
ras = rasterio.open(file_path, cache=False)
N = 49.61
S = 24.05
E = -65.57
W = -126.04

W, N = ras.index(W, N)
E, S = ras.index(E, S)
print("US-Pixel:", W, N, E, S)

lsm = ras.read()[:, W:E, N:S]
print("Shape reduced to:", lsm.shape, lsm.min())

lsm = lsm[0]>=0
land = np.where(lsm)
land = np.transpose(np.stack(land))
print(land.shape)

SAMPLES = 100000
sample = np.random.choice(len(land), SAMPLES)
sample = land[sample]
print(sample.shape)

ras = rioxarray.open_rasterio(file_path, cache=False)
data = np.zeros((len(sample), 3), dtype="float")
for i in tqdm(range(len(sample))):
    y, x = sample[i]
    loc = ras[0, y, x]
    data[i][0] = loc["x"] # Saving as lon/lat
    data[i][1] = loc["y"]
    data[i][2] = float(loc)
np.save('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/pop_density_us_'+str(SAMPLES), data)

ras.close()