import numpy as np
import rioxarray
from geopy.distance import geodesic, distance
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imsave
from joblib import Parallel, delayed, dump, load
import os


SCOPE = "switzerland"
SCOPE = "euro"
SCOPE = "world"

if SCOPE == "switzerland":
    raster_folder = "/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/"
    input_folder = "/shares/wegner.ics.uzh/CHELSA/Switzerland/input/"
    N_y, W_x = (4340, 22310)
    S_y, E_x = (4585, 22860)
    ras = rioxarray.open_rasterio('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)
    ras = ras[0, N_y:S_y,W_x:E_x]
elif SCOPE == "euro":
    raster_folder = "/shares/wegner.ics.uzh/CHELSA/Europe/1981-2010_numpy/"
    input_folder = "/shares/wegner.ics.uzh/CHELSA/Europe/input/"
    W_x = 20300
    N_y = 1200
    E_x = 25300
    S_y = 6200
    ras = rioxarray.open_rasterio('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)
    ras = ras[0, N_y:S_y,W_x:E_x]
else:
    raster_folder = "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/"
    input_folder = "/shares/wegner.ics.uzh/CHELSA/input/"
    ras = rioxarray.open_rasterio('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)[0]

    
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
#months = ["03", "04"]

raster = []
for m in tqdm(months):
    with open(raster_folder + m + "_monthly_float16.npy", "rb") as f:
        raster.append(np.load(f))
#raster = np.stack(raster).astype(np.half)

print("Loaded rasters")

#sharpness = np.zeros((raster.shape[-2], raster.shape[-1]))

BLOCK_LEN = 121

block_ids = []
for i in range((raster[0].shape[-2]//BLOCK_LEN) + 1):
    for j in range((raster[0].shape[-1]//BLOCK_LEN) + 1):
        block_ids.append((i, j))

#dump(raster, "./raster.npy")
#raster = load("./raster.npy", mmap_mode='r')

sharpness = np.memmap("./sharpness.npy", dtype=np.half,
                   shape=(raster[0].shape[-2], raster[0].shape[-1]), mode='w+')

def process(i, j, output, raster):
    raster_block = np.stack([r[:,i*BLOCK_LEN:(i+1)*BLOCK_LEN,j*BLOCK_LEN:(j+1)*BLOCK_LEN] for r in raster]).astype(np.half)
    grads = np.array(np.gradient(raster_block, axis=(0,2,3)))
    y_size, x_size = grads.shape[-2], grads.shape[-1]
    grads = grads.reshape(-1, y_size, x_size)
    gnorm = np.sqrt(grads**2)
    output[i*BLOCK_LEN:(i+1)*BLOCK_LEN,j*BLOCK_LEN:(j+1)*BLOCK_LEN] = np.average(gnorm, axis=0)

Parallel(n_jobs=os.cpu_count())(delayed(process)(i, j, sharpness, raster) for i,j in tqdm(block_ids))        
    
print("Calculated sharpness", sharpness.shape)

# Here we have the issue that coastal tiles have a very significant gradient
# These values need to be located
# print(np.histogram(sharpness.flatten(), bins=20))
# No real sharpness value is above 1
# -> Fill all values above 1 with local interpolations

# Taken from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
to_be_interpolated, x = sharpness>1, lambda z: z.nonzero()[0]
sharpness[to_be_interpolated]= np.interp(x(to_be_interpolated), x(~to_be_interpolated), sharpness[~to_be_interpolated])

print("Remove sharpness coast cases")
print(" ->", sharpness.min(), "to", sharpness.max())

distance_weights = []
print("Calculating distance weights")
for lat in tqdm(np.array(ras["y"])):
    distance_weights.append(geodesic((lat, ras["x"][42]), (lat, ras["x"][43])).km)
distance_weights = np.array(distance_weights)

print(" ->", distance_weights.min(), "to", distance_weights.max())

distance_weights = np.tile(distance_weights, (raster[0].shape[-1],1)).transpose(1,0)
print("Distance weights shape", distance_weights.shape)

weights = sharpness * distance_weights
print("Weights", weights.shape)
print(" ->", weights.min(), "to", weights.max())

lsm = np.array(ras)<30000
y, x = np.where(lsm)
idx_to_weight = weights[y, x]
print("Idx to weight", idx_to_weight.shape)

np.save(input_folder + "idx_to_weight", idx_to_weight)

if SCOPE == "switzerland":
    imsave(input_folder + "weights.jpg", weights)
elif SCOPE == "euro":
    imsave(input_folder + "weights.jpg", weights)
else:
    imsave(input_folder + "weights.jpg", weights[::10,::10])
print("Fitting shape? ->", np.load(input_folder + "land_coordinates.npy").shape)

"""
Switzerland:
Weights (245, 550)
 -> 0.0376755497494984 to 0.31053746319772807
 
Euro:



World:
"""
 