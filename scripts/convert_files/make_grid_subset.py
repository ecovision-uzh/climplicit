# imports
import rioxarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

import os


SAMPLING_DIST = 10

ras = rioxarray.open_rasterio(
    "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif",
    cache=False,
)

# calculate land_coordinates
# calculate point_to_coord
original_inp = "/shares/wegner.ics.uzh/CHELSA/input/"

try:
    os.mkdir("/shares/wegner.ics.uzh/CHELSA/Gridsubset/")
except:
    pass
try:
    os.mkdir("/shares/wegner.ics.uzh/CHELSA/Gridsubset/" + str(SAMPLING_DIST))
except:
    pass

sw_inp = "/shares/wegner.ics.uzh/CHELSA/Gridsubset/" + str(SAMPLING_DIST) + "/"

lsm = np.array(ras)[:, ::SAMPLING_DIST, ::SAMPLING_DIST]
print("Shape reduced to:", lsm.shape, lsm.max())

lsm = lsm[0] < 30000
land = np.where(lsm)
land = np.transpose(np.stack(land))

print("Before fixing transform:", land[-1])
land = (
    land * SAMPLING_DIST
)  # Need to scale back up to get the correct pixels in the data
print("After fixing transform:", land[-1])
# x = lon, y = lat

# Saving the lon/lat for each land-(x,y)
coors = np.zeros_like(land, dtype="float")
for i in tqdm(range(len(land))):
    y, x = land[i]
    loc = ras[0, y, x]
    coors[i][0] = loc["x"]  # Saving as lon/lat
    coors[i][1] = loc["y"]
coors = np.array(coors)
np.save(sw_inp + "point_to_coord.npy", coors)
np.save(sw_inp + "land_coordinates.npy", land)

print(coors[:3])
print(coors[-3:])

land_train = land
coors_train = coors
# Take 10% of train as val
val_index = np.random.choice(
    len(land_train), math.floor(len(land_train) / 10), replace=False
)
land_val = land_train[val_index]
coors_val = coors_train[val_index]
land_train = np.delete(land_train, val_index, 0)
coors_train = np.delete(coors_train, val_index, 0)
# Take 50% of val (thus 5% of train) as test
test_index = np.random.choice(
    len(land_val), math.floor(len(land_val) / 2), replace=False
)
land_test = land_val[test_index]
coors_test = coors_val[test_index]
land_val = np.delete(land_val, test_index, 0)
coors_val = np.delete(coors_val, test_index, 0)
print(land_train.shape, coors_train.shape)
print(land_val.shape, coors_val.shape)
print(land_test.shape, coors_test.shape)

np.save(sw_inp + "land_coordinates_train.npy", land_train)
np.save(sw_inp + "land_coordinates_val.npy", land_val)
np.save(sw_inp + "land_coordinates_test.npy", land_test)
np.save(sw_inp + "point_to_coord_train.npy", coors_train)
np.save(sw_inp + "point_to_coord_val.npy", coors_val)
np.save(sw_inp + "point_to_coord_test.npy", coors_test)
