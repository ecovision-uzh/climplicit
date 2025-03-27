# imports
import rioxarray
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

# define bounds
N = 47.813
W = 5.933
S = 45.806
E = 10.514

NW_y, NW_x = (4340, 22310)
SE_y, SE_x = (4585, 22860)

ras = rioxarray.open_rasterio(
    "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif",
    cache=False,
)
switzerland = ras[:, NW_y:SE_y, NW_x:SE_x]

# climatologies
original_clim = "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/"
sw_clim = "/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/"
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

"""for month in tqdm(months):
    mora = np.load(original_clim+month+"_monthly_float16.npy")
    switzerland = mora[:,NW_y:SE_y,NW_x:SE_x]
    np.save(sw_clim+month+"_monthly_float16.npy", switzerland)"""

# calculate land_coordinates
# calculate point_to_coord
original_inp = "/shares/wegner.ics.uzh/CHELSA/input/"
sw_inp = "/shares/wegner.ics.uzh/CHELSA/Switzerland/input/"

lsm = np.array(switzerland)[0] < 30000
land = np.where(lsm)
land = np.transpose(np.stack(land))

y, x = land[-1]
# x = lon, y = lat

# Saving the lon/lat for each land-(x,y)
coors = np.zeros_like(land, dtype="float")
for i in tqdm(range(len(land))):
    y, x = land[i]
    loc = switzerland[0, y, x]
    coors[i][0] = loc["x"]  # Saving as lon/lat
    coors[i][1] = loc["y"]
coors = np.array(coors)
np.save(sw_inp + "point_to_coord.npy", coors)
np.save(sw_inp + "land_coordinates.npy", land)

print(coors[:5])
print(coors[5:])

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
