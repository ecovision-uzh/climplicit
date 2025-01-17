from torch.utils.data import Dataset
from time import time
from tqdm import tqdm
import pickle as pk
import numpy as np
import os
import json
import rioxarray
import random
import torch
import random

from typing import List

class ChelsaCLIPDataset(Dataset):
    def __init__(
        self,
        monthly_arrays,
        land_coordinates_file: str,
        point_to_coord_file: str,
        months: List[str],
        skip_samples: int,
        return_size: int,
        local_multi_sampling: bool,
        whiten_with_pca: bool,
        ):
        
        self.monthly_arrays = monthly_arrays
        if return_size > 1:
            self.y_len, self.x_len = self.monthly_arrays[list(self.monthly_arrays.keys())[0]][0].shape

        self.months = months

        self.return_size = return_size

        self.local_multi_sampling = local_multi_sampling
        
        if local_multi_sampling:
            ras = rioxarray.open_rasterio('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)
            if "Switzerland" in land_coordinates_file:
                NW_y, NW_x = (4340, 22310)
                SE_y, SE_x = (4585, 22860)
                ras = ras[:,NW_y:SE_y,NW_x:SE_x]
            self.x_to_lon = ras["x"]
            self.y_to_lat = ras["y"]
            ras.close()

        if whiten_with_pca:
            self.whiten_pca =  pk.load(open("/shares/wegner.ics.uzh/CHELSA/input/whiten_pca_components.pkl",'rb')) 
        else:
            self.whiten_pca = None

        # Load location file
        if skip_samples is None:
            #self.locs = np.load(land_coordinates_file)
            self.point_to_coord = np.load(point_to_coord_file)
        else:
            #self.locs = np.load(land_coordinates_file)[::skip_samples]
            self.point_to_coord = np.load(point_to_coord_file)[::skip_samples]
            for month in self.monthly_arrays.keys():
                self.monthly_arrays[month][:, ::skip_samples]

    def __len__(self):
        return len(self.point_to_coord)

    def get_idx(self, idx, month):
        #pixel_y, pixel_x = self.locs[idx]
        lonlat = self.point_to_coord[idx]

        if self.return_size > 1:
            side = self.return_size // 2
            if pixel_y-side < 0:
                y_left = 0
                y_right = self.return_size
            elif pixel_y+side >= self.y_len:
                y_left = self.y_len - self.return_size
                y_right = self.y_len
            else:
                y_left = pixel_y-side
                y_right = pixel_y+side+1
            if pixel_x-side < 0:
                x_up = 0
                x_down = self.return_size
            elif pixel_x+side >= self.x_len:
                x_up = self.x_len - self.return_size
                x_down = self.x_len
            else:
                x_up = pixel_x-side
                x_down = pixel_x+side+1
            chelsa = self.monthly_arrays[month][:, y_left:y_right, x_up:x_down]
        else:
            #chelsa = self.monthly_arrays[month][:, pixel_y, pixel_x]
            chelsa = self.monthly_arrays[month][:, idx]
            #print(pixel_y, pixel_x, self.monthly_arrays[month].shape, self.locs.shape)

        if self.whiten_pca:
            chelsa = self.whiten_pca.transform(chelsa.reshape((1,11))).reshape((11)).astype("float16")

        return lonlat, int(month), chelsa

    """def get_idx(self, idx):
        pixel_y, pixel_x = self.locs[idx]
        lonlat = self.point_to_coord[idx]

        if self.return_size > 1:
            raise NotImplementedError()
        else:
            chelsa = []
            for mar in self.monthly_arrays.values():
                chelsa.append(torch.tensor(mar[:, pixel_y, pixel_x]))
        chelsa = torch.cat(chelsa)

        if self.whiten_pca:
            chelsa = self.whiten_pca.transform(chelsa.reshape((1,11))).reshape((11)).astype("float16")

        return lonlat, chelsa"""

    def get_random_adjacent_to_idx(self, idx, month):
        pixel_y, pixel_x = self.locs[idx]

        # Calculate adjacent pixel, and make sure it is in bounds
        # Repeat
        y, x = pixel_y, pixel_x
        while(y == pixel_y and x == pixel_x):
            y, x = pixel_y + random.randint(-1, 1), pixel_x + random.randint(-1, 1)
            y, x = max(y, 0), max(x, 0)
            shape = self.monthly_arrays[month][0].shape
            y, x = min(y, shape[0]-1), min(x, shape[1]-1)

        chelsa = self.monthly_arrays[month][:, y, x]
        lonlat = np.array([self.x_to_lon[x], self.y_to_lat[y]])
        return lonlat, int(month), chelsa
    
    def get_all_adjacent_to_idx(self, idx, month):
        pixel_y, pixel_x = self.locs[idx]

        # Calculate all adjacent pixel, and make sure they are in bounds
        lonlats, months, chelsas = [], [], []
        shape = self.monthly_arrays[month][0].shape
        for i in range(-1, 2):
            for j in range(-1, 2):
                y, x = pixel_y + i, pixel_x + j
                if not (y < 0 or y > shape[0]-1 or x < 0 or x > shape[1]-1 or (x == pixel_x and y == pixel_y)):
                    lonlats.append(torch.tensor(np.array([self.x_to_lon[x], self.y_to_lat[y]])))
                    months.append(torch.tensor(int(month)))
                    chelsas.append(torch.tensor(self.monthly_arrays[month][:, y, x]))
        return lonlats, months, chelsas

    def __getitem__(self, idx):
        if self.local_multi_sampling:
            month = random.choice(self.months)
            l1, m1, c1 = self.get_idx(idx, month)
            mode = "other" #"random_adjacent"
            if mode == "random_adjacent":
                l2, m2, c2 = self.get_random_adjacent_to_idx(idx, month)
                lonlats = torch.stack([torch.tensor(l1), torch.tensor(l2)])
                months = torch.stack([torch.tensor(m1), torch.tensor(m2)])
                chelsas = torch.stack([torch.tensor(c1), torch.tensor(c2)])
            else:
                lonlats, months, chelsas = self.get_all_adjacent_to_idx(idx, month)
                lonlats.append(torch.tensor(l1))
                months.append(torch.tensor(m1))
                chelsas.append(torch.tensor(c1))
                lonlats = torch.stack(lonlats)
                months = torch.stack(months)
                chelsas = torch.stack(chelsas)
            return lonlats, months, chelsas
        else:
            return self.get_idx(idx, month = random.choice(self.months))
            #return self.get_idx(idx)


if __name__ == "__main__":
    from time import time
    start = time()
    ds = ChelsaCLIPDataset(
        climatology_dir="/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/",
        land_coordinates_file="/shares/wegner.ics.uzh/CHELSA/input/land_coordinates_train.npy",
        loc_to_coord_file="/shares/wegner.ics.uzh/CHELSA/input/point_to_coord.npy")
    print(" -> Total loading time:", time() - start)
    print("Dataset of Len:", len(ds))

    for i in tqdm(range(len(ds))):
        ds[i]

