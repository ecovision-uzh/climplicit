import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math


class Direct(nn.Module):
    """
    Just turn it into a [-1,1] scaling based on the input range
    """
    def __init__(self, lon_min, lon_max, lat_min, lat_max):
        """
        Args:
        """
        super(Direct, self).__init__()
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def forward(self, coords):
        """"""
        lon, lat = coords[:,0], coords[:,1]
        lon = 2 * (lon - self.lon_min) / (self.lon_max-self.lon_min) - 1
        lat = 2 * (lat - self.lat_min) / (self.lat_max-self.lat_min) - 1
        return torch.stack([lon, lat], dim=1).float()



if __name__ == "__main__":
    from tqdm import tqdm
    BS, EXPS, SPA_EMB, FREQ_NUM = (32000, 10, 40, 64)
    print("Testing for BS", BS, "- EXPS", EXPS, "- FREQ_NUM", FREQ_NUM, "- SPA_EMB", SPA_EMB)
    sg = Direct(lon_min=-90, lon_max=90, lat_min=-90, lat_max=90, device = "cuda")

    # longitude pi and latitude pi/2 
    lonlat = torch.rand([BS,2]) * 180 - 90  # Random values in [-90,90]
    lonlat = lonlat.unsqueeze(dim=1).numpy()
    for i in tqdm(range(EXPS)):
        out = sg(lonlat).squeeze(dim=1)
    print(out[0], len(out[0]), out.shape)