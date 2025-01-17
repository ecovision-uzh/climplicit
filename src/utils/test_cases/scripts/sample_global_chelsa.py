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

file_name = '/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/03_monthly_float16.npy'
ras = np.load(file_name)

ref_path = '/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif'
ref_ras = rasterio.open(ref_path, cache=False)

#lsm = ras[0][0]

#lsm = lsm>0.1 # Only sample locations with actual values
#land = np.where(lsm)
#land = np.transpose(np.stack(land))
#print(land.shape)

SAMPLES = 100000
#sample = np.random.choice(len(land), SAMPLES)
#sample = land[sample]
sample = []

while len(sample) < SAMPLES:
    rand_feats_orig = torch.rand(2)
    theta1 = 2.0*math.pi*rand_feats_orig[0]
    theta2 = torch.acos(2.0*rand_feats_orig[1] - 1.0)
    lat = 1.0 - 2.0*theta2/math.pi
    lon = (theta1/math.pi) - 1.0
    lat = lat * 90
    lon = lon * 180
    # Also only take samples outside of Antartica
    try:
        if globe.is_land(lat, lon) and lat > -63:
            sample.append((lon.numpy(), lat.numpy()))
    except:
        pass
    if len(sample)%(SAMPLES//10) == 0:
        print(len(sample))

data = np.array(sample)

#sample_ras = rioxarray.open_rasterio(file_folder+files[0], cache=False)
data = np.zeros((len(sample), 13), dtype="float")
for i in tqdm(range(len(sample))):
    #y, x = sample[i]
    #loc = sample_ras[0, y, x]
    #data[i][0] = loc["x"] # Saving as lon/lat
    #data[i][1] = loc["y"]
    lon, lat = sample[i]
    data[i][0] = lon # Saving as lon/lat
    data[i][1] = lat  
    y, x = ref_ras.index(lon.item(), lat.item())
    data[i][2:] = ras[:, y, x]
np.save('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/global_chelsa_'+str(SAMPLES), data)

print(data[:3])

#sample_ras.close()