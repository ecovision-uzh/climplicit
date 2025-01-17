import torch
import rioxarray
import numpy as np
from tqdm import tqdm

from global_land_mask import globe

class SwitzerlandDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.N = 47.813
        self.W = 5.933
        self.S = 45.806
        self.E = 10.514

        # originally 342km EW, 223km SN
        # We use the scaling from the original CHELSA data here, amounting to less than 1km in x-direction

        self.x_pixel = 550
        self.y_pixel = 245

    def __len__(self):
        return self.x_pixel * self.y_pixel

    def __getitem__(self, idx): 
        x = idx % self.x_pixel
        y = int(idx / self.x_pixel)

        lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
        lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

        return torch.tensor([lon, lat])

class LakeVictoriaDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.N = 5
        self.W = 28.78
        self.S = -6
        self.E = 40.89

        #self.x_pixel = 120
        #self.y_pixel = 105
        self.x_pixel = 480
        self.y_pixel = 420

        self.land_mask = np.zeros((self.y_pixel, self.x_pixel)).astype(int)

        for idx in range(self.x_pixel * self.y_pixel):
            x = idx % self.x_pixel
            y = int(idx / self.x_pixel)

            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

            self.land_mask[y, x] = int(globe.is_land(lat, lon))

        land = np.transpose(np.stack(np.where(self.land_mask)))
        self.coors = np.zeros_like(land, dtype="float")
        for i in range(len(land)):
            y, x = land[i]
            self.coors[i][0] = self.W + (x/self.x_pixel) * (self.E - self.W) # Saving as lon/lat
            self.coors[i][1] = self.N + (y/self.y_pixel) * (self.S - self.N)
        self.coors = np.array(self.coors).astype("float")

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx): 
        return torch.tensor(self.coors[idx]).float()

    def get_ocean_coords(self):
        land = np.transpose(np.stack(np.where(self.land_mask == 0)))
        res = []
        for yx in land:
            y, x = yx[0], yx[1]
            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N)
            res.append([lon,lat])
        return np.array(res)


class FranceDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.N = 5
        self.W = 28.78
        self.S = -6
        self.E = 40.89

        #self.x_pixel = 120
        #self.y_pixel = 105
        self.x_pixel = 420
        self.y_pixel = 420

        self.land_mask = np.zeros((self.y_pixel, self.x_pixel)).astype(int)

        for idx in range(self.x_pixel * self.y_pixel):
            x = idx % self.x_pixel
            y = int(idx / self.x_pixel)

            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

            self.land_mask[y, x] = int(globe.is_land(lat, lon))

        land = np.transpose(np.stack(np.where(self.land_mask)))
        self.coors = np.zeros_like(land, dtype="float")
        for i in range(len(land)):
            y, x = land[i]
            self.coors[i][0] = self.W + (x/self.x_pixel) * (self.E - self.W) # Saving as lon/lat
            self.coors[i][1] = self.N + (y/self.y_pixel) * (self.S - self.N)
        self.coors = np.array(self.coors).astype("float")

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx): 
        return torch.tensor(self.coors[idx]).float()

    def get_ocean_coords(self):
        land = np.transpose(np.stack(np.where(self.land_mask == 0)))
        res = []
        for yx in land:
            y, x = yx[0], yx[1]
            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N)
            res.append([lon,lat])
        return np.array(res)


class ZurichDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.N = 47.5432
        self.W = 8.3688
        self.S = 46.8696
        self.E = 9.3906

        # originally 77km EW, 75km SN
        # We use a sampling of 0,25km

        self.x_pixel = 308
        self.y_pixel = 300

    def __len__(self):
        return self.x_pixel * self.y_pixel

    def __getitem__(self, idx): 
        x = idx % self.x_pixel
        y = int(idx / self.x_pixel)

        lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
        lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

        return torch.tensor([lon, lat])

class WorldDataset(torch.utils.data.Dataset):
    def __init__(self):

        ras = rioxarray.open_rasterio("/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/sfcWind/CHELSA_sfcWind_03_1981-2010_V.2.1.tif", cache=False)

        self.N = 90
        self.W = -180
        self.S = -63
        self.E = 180

        self.x_pixel = 440
        self.y_pixel = 200
        """self.sample_dist = 10

        # Take EU subset and use every tenth sample
        # NW = -10.83,74.00 ; SE = 32.41,30.75
        # originally 1248km to 3875km EW, 4621km SN
        # we use 500x500 (sample_dist=10) pixel for visuals
        eu = ras[0, N_y:S_y:self.sample_dist, W_x:E_x:self.sample_dist]

        self.land_mask = eu < 65000

        fn = "/home/jdolli/chelsaCLIP/src/utils/test_cases/data/util_ds_europe_coors.npy"
        
        try:
            self.coors = np.load(fn).astype("float")
        except:
            land = np.transpose(np.stack(np.where(self.land_mask)))
            self.coors = np.zeros_like(land, dtype="float")
            print("Creating Europe coors for util dataset at " + fn)
            for i in tqdm(range(len(land))):
                y, x = land[i]
                loc = eu[y, x]
                self.coors[i][0] = loc["x"] # Saving as lon/lat
                self.coors[i][1] = loc["y"]
            self.coors = np.array(self.coors).astype("float")
            np.save(fn, self.coors)

        self.x_pixel = len(eu[0])
        self.y_pixel = len(eu)"""
        self.land_mask = np.zeros((self.y_pixel, self.x_pixel)).astype(int)

        for idx in range(self.x_pixel * self.y_pixel):
            x = idx % self.x_pixel
            y = int(idx / self.x_pixel)

            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

            self.land_mask[y, x] = int(globe.is_land(lat, lon))

        land = np.transpose(np.stack(np.where(self.land_mask)))
        self.coors = np.zeros_like(land, dtype="float")
        for i in range(len(land)):
            y, x = land[i]
            self.coors[i][0] = self.W + (x/self.x_pixel) * (self.E - self.W) # Saving as lon/lat
            self.coors[i][1] = self.N + (y/self.y_pixel) * (self.S - self.N)
        self.coors = np.array(self.coors).astype("float")

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx): 
        return torch.tensor(self.coors[idx]).float()

    def get_ocean_coords(self):
        land = np.transpose(np.stack(np.where(self.land_mask == 0)))
        res = []
        for yx in land:
            y, x = yx[0], yx[1]
            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N)
            res.append([lon,lat])
        return np.array(res)

class EuropeDataset(torch.utils.data.Dataset):
    def __init__(self):

        ras = rioxarray.open_rasterio("/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/sfcWind/CHELSA_sfcWind_03_1981-2010_V.2.1.tif", cache=False)

        W_x = 20300
        N_y = 1200
        E_x = 25300
        S_y = 6200

        self.N = 74.0
        self.W = -10.83
        self.S = 32.41
        self.E = 30.75

        self.x_pixel = 330
        self.y_pixel = 440
        """self.sample_dist = 10

        # Take EU subset and use every tenth sample
        # NW = -10.83,74.00 ; SE = 32.41,30.75
        # originally 1248km to 3875km EW, 4621km SN
        # we use 500x500 (sample_dist=10) pixel for visuals
        eu = ras[0, N_y:S_y:self.sample_dist, W_x:E_x:self.sample_dist]

        self.land_mask = eu < 65000

        fn = "/home/jdolli/chelsaCLIP/src/utils/test_cases/data/util_ds_europe_coors.npy"
        
        try:
            self.coors = np.load(fn).astype("float")
        except:
            land = np.transpose(np.stack(np.where(self.land_mask)))
            self.coors = np.zeros_like(land, dtype="float")
            print("Creating Europe coors for util dataset at " + fn)
            for i in tqdm(range(len(land))):
                y, x = land[i]
                loc = eu[y, x]
                self.coors[i][0] = loc["x"] # Saving as lon/lat
                self.coors[i][1] = loc["y"]
            self.coors = np.array(self.coors).astype("float")
            np.save(fn, self.coors)

        self.x_pixel = len(eu[0])
        self.y_pixel = len(eu)"""
        self.land_mask = np.zeros((self.y_pixel, self.x_pixel)).astype(int)

        for idx in range(self.x_pixel * self.y_pixel):
            x = idx % self.x_pixel
            y = int(idx / self.x_pixel)

            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N) # lat goes from top (0) to bottom

            self.land_mask[y, x] = int(globe.is_land(lat, lon))

        land = np.transpose(np.stack(np.where(self.land_mask)))
        self.coors = np.zeros_like(land, dtype="float")
        for i in range(len(land)):
            y, x = land[i]
            self.coors[i][0] = self.W + (x/self.x_pixel) * (self.E - self.W) # Saving as lon/lat
            self.coors[i][1] = self.N + (y/self.y_pixel) * (self.S - self.N)
        self.coors = np.array(self.coors).astype("float")

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx): 
        return torch.tensor(self.coors[idx]).float()

    def get_ocean_coords(self):
        land = np.transpose(np.stack(np.where(self.land_mask == 0)))
        res = []
        for yx in land:
            y, x = yx[0], yx[1]
            lon = self.W + (x/self.x_pixel) * (self.E - self.W) # lon goes from right (0) to left
            lat = self.N + (y/self.y_pixel) * (self.S - self.N)
            res.append([lon,lat])
        return np.array(res)

"""
class WorldDataset(torch.utils.data.Dataset):
    def __init__(self):

        ras = rioxarray.open_rasterio("/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/sfcWind/CHELSA_sfcWind_03_1981-2010_V.2.1.tif", cache=False)

        W_x = 0
        N_y = 0
        E_x = 43200
        S_y = 20880

        self.N = 90
        self.S = -90
        self.W = -180
        self.E = 180

        self.sample_dist = 50

        # Take world and use every fifthieth sample
        # NW = -180,90 ; SE = 180,-90
        # originally 40.000km EW, 20.000km SN
        # we use 864,418 (sample_dist=25) pixel for visuals
        world = ras[0, N_y:S_y:self.sample_dist, W_x:E_x:self.sample_dist]

        self.land_mask = world < 65000

        fn = "/home/jdolli/chelsaCLIP/src/utils/test_cases/data/util_ds_world_coors_" + str(self.sample_dist) + ".npy"

        try:
            self.coors = np.load(fn)
        except:
            land = np.transpose(np.stack(np.where(self.land_mask)))
            self.coors = np.zeros_like(land, dtype="float")
            print("Creating World coors for util dataset at " +fn)
            for i in tqdm(range(len(land))):
                y, x = land[i]
                loc = world[y, x]
                self.coors[i][0] = loc["x"] # Saving as lon/lat
                self.coors[i][1] = loc["y"]
            self.coors = np.array(self.coors)
            np.save(fn, self.coors)

        self.x_pixel = len(world[0])
        self.y_pixel = len(world)

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx): 
        return torch.tensor(self.coors[idx]).float()
"""

class SwitzerlandDatasetTC(torch.utils.data.Dataset):
    def __init__(self):
        point_to_coord_file="/shares/wegner.ics.uzh/CHELSA/Switzerland/input/point_to_coord.npy"
        self.locs = np.load(point_to_coord_file)
        self.x_pixel = 550
        self.y_pixel = 245

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx): 
        return torch.tensor(self.locs[idx]).float()


if __name__ == "__main__":
    """ds_tc = SwitzerlandDatasetTC()
    ds = SwitzerlandDataset()
    print(ds_tc[0], ds_tc[1])
    print(ds[0], ds[1])
    print(ds_tc[len(ds_tc)-1], ds_tc[len(ds_tc)-2])
    print(ds[len(ds_tc)-1], ds[len(ds_tc)-2])"""
    #ds = SwitzerlandDataset()
    #print(ds[0])
    
    #ds = EuropeDataset()
    #print(ds[0])

    ds = WorldDataset()
    print(ds[0])