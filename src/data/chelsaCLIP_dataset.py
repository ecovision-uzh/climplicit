from torch.utils.data import Dataset
from time import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import json
import rioxarray
import random


class ChelsaCLIPDataset(Dataset):
    def __init__(self, climatology_dir: str, land_coordinates_file: str, use_bio: bool=False, use_dem: bool=False, verbose: bool=False):
        
        # Load the 11*12 monthly rasters
        start = time()
        
        self.var_names = ["clt", "cmi", "hurs", "pet", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin", "vpd"]
        self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        self.rasters = defaultdict(list)
        for month in tqdm(self.months):
            for var in self.var_names:
                if var == "pet":
                    raster_file_name = climatology_dir + var + "/CHELSA_pet_penman_" + str(month) + "_1981-2010_V.2.1.tif"
                elif var == "rsds":
                    raster_file_name = climatology_dir + var + "/CHELSA_" + var + "_1981-2010_" + str(month) + "_V.2.1.tif"
                else:
                    raster_file_name = climatology_dir + var + "/CHELSA_" + var + "_" + str(month) + "_1981-2010_V.2.1.tif"
                raster = rioxarray.open_rasterio(raster_file_name)
                self.rasters[month].append(raster)
        self.x_to_lon = raster.indexes["x"]
        self.y_to_lat = raster.indexes["y"]
        if verbose:
            print("Loaded monthly rasters in:", time() - start)

        if use_bio:
            raise NotImplementedError()

        if use_dem:
            raise NotImplementedError()
        
        # Load location file
        start = time()
        self.locs = np.load(land_coordinates_file)
        if verbose:
            print("Loaded locations in:", time() - start)

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        pixel_y = self.locs[idx, 0]
        pixel_x = self.locs[idx, 1]

        lat = self.y_to_lat[pixel_y]
        lon = self.x_to_lon[pixel_x]

        month = random.choice(self.months)

        chelsa = []
        for raster in self.rasters[month]:
            print(raster)
            chelsa.append(raster[0, pixel_x, pixel_y])
        chelsa = np.stack(chelsa)

        return (lon, lat, month), chelsa


if __name__ == "__main__":
    from time import time
    start = time()
    ds = ChelsaCLIPDataset(
        climatology_dir="/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_normalized/",
        land_coordinates_file="/shares/wegner.ics.uzh/CHELSA/input/land_coordinates_train.npy",
        verbose=True)
    print(" -> Total loading time:", time() - start)
    print("Dataset of Len:", len(ds))

    for i in tqdm(range(32000)):
        ds[i*1000]



    # Current issues: It is way to slow in loading
    #   There is one raster that is smaller
    # -> Make the files smaller through datatype (currently float64), or load them smaller -> Can load them all into RAM
    # -> Sample the one smaller raster by lat/lon instead
    # -> Or maybe convert it to the bigger size first and then sample it normally
    # -> Why does my normalization code not properly delete stuff after looking at the rasters??

