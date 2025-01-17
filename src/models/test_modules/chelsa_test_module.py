import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

import wandb

import pickle
import sys
import rasterio

from tqdm import tqdm
import numpy as np

import sys
sys.path.append('/home/jdolli/chelsaCLIP/src/utils')
from positional_encoding.spheregrid import SphereGridSpatialRelationEncoder

class CHELSA_Loc_Enc(torch.nn.Module):
    def __init__(self, add_location):
        super().__init__()
        chelsa_dir = "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/"

        self.add_location = add_location
        if self.add_location:
            self.loc_addendum = SphereGridSpatialRelationEncoder(
                coord_dim= 2,
                frequency_num= 64,
                max_radius= 360,
                min_radius= 0.0003,
                freq_init= "geometric",
                device= "cuda")

        self.monthly_arrays = {}
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        months = ["03", "06", "09", "12"]
        #months = ["03"]
        for month in tqdm(months):
            #self.monthly_arrays[int(month)] = np.load(chelsa_dir + month + "_monthly_float16.npy", mmap_mode="r")
            self.monthly_arrays[int(month)] = np.load(chelsa_dir + month + "_monthly_float16.npy")
        self.ras = rasterio.open('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif')

    def forward(self, locs, months):
        # Horribly inefficient, but eh
        res = []
        for idx in range(len(locs)):
            lonlat = locs[idx]
            month = months[idx]
            lon, lat = lonlat[0].item(), lonlat[1].item()
            try:
                y, x = self.ras.index(lon, lat)
            except:
                res.append(torch.tensor([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10]))
                continue
            if y == self.monthly_arrays[3].shape[1]:
                y -= 1
            if x == self.monthly_arrays[3].shape[2]:
                x -= 1
            try:
                res.append(torch.tensor(self.monthly_arrays[month.item()][:,y, x]))
            except:
                res.append(torch.tensor(self.monthly_arrays[3][:,y, x]))
        if self.add_location:
            res = torch.stack(res)
            return torch.cat([res, self.loc_addendum(locs).to(res.device).reshape(len(res), -1)], dim=1)
        else:
            return torch.stack(res)
        
        

class CHELSATestModule(LightningModule):
    """
    """

    def __init__(
        self,
        test_cases = None,
        add_location = False,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.pos_embedding = torch.nn.Identity()
        self.location_encoder = CHELSA_Loc_Enc(add_location)

        self.test_cases = test_cases

    def test_step(self, batch, batch_idx):

        # We only use wandb logging
        if batch_idx == 0 and self.logger and self.test_cases:
            wb = self.logger.experiment
            for _, case in self.test_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            case(self.pos_embedding, self.location_encoder, wb)


if __name__ == "__main__":
    loc_enc = CHELSA_Loc_Enc(add_location=True)
    inp = []
    months = []
    for i in range(3):
        inp.append(torch.tensor([170, -30]))
        months.append(torch.tensor([3]))
    print(loc_enc(torch.stack(inp), torch.stack(months)).shape)
