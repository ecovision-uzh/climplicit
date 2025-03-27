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
from positional_encoding.direct import Direct
sys.path.append('/home/jdolli/chelsaCLIP/src/models/components')
from loc_encoder import SirenNet
from residual_net import Residual_Net

class CHELSA_Loc_Enc(torch.nn.Module):
    def __init__(self, freeze, use_chelsa, use_loc, months):
        super().__init__()
        chelsa_dir = "/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/"

        if months=="all":
            self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        elif months=="seasons":
            self.months = ["03", "06", "09", "12"]
        elif months=="march":
            self.months = ["03"]
        else:
            raise NotImplementedError("Incorrect month.")

        self.freeze = freeze
        self.use_chelsa = use_chelsa
        self.use_loc = use_loc
        
        self.reset_model()

        if use_chelsa:
            self.monthly_arrays = {}
            for month in tqdm(self.months):
                #self.monthly_arrays[int(month)] = np.load(chelsa_dir + month + "_monthly_float16.npy", mmap_mode="r")
                self.monthly_arrays[int(month)] = np.load(chelsa_dir + month + "_monthly_float16.npy")
            self.ras = rasterio.open('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif')

    def reset_model(self):
        """self.pos_emb = SphereGridSpatialRelationEncoder(
                coord_dim= 2,
                frequency_num= 64,
                max_radius= 360,
                min_radius= 0.0003,
                freq_init= "geometric",
                device= "cuda")"""

        self.pos_emb = Direct(lon_min=-180,lon_max=180,lat_min=-90,lat_max=90)
        self.loc_enc =  SirenNet(dim_in=2, dim_hidden=512, dim_out=256, num_layers=16, dropout=False, h_siren=True, residual_connections=True)
        self.chelsa_enc = Residual_Net(input_len=11*len(self.months), hidden_dim=64, layers=4, batchnorm=False, out_dim=256)

        if self.freeze:
            for param in self.loc_enc.parameters():
                param.requires_grad = False
            for param in self.chelsa_enc.parameters():
                param.requires_grad = False

    def forward(self, locs):
        # Horribly inefficient, but eh
        if self.use_chelsa:
            res = []
            for idx in range(len(locs)):
                lonlat = locs[idx]
                lon, lat = lonlat[0].item(), lonlat[1].item()
                
                chelsa_enc = []
                try:
                    y, x = self.ras.index(lon, lat)
                except:
                    res.append(torch.tensor([-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10] * len(self.monthly_arrays)))
                    continue
                for mar in self.monthly_arrays.values():
                    if y == mar.shape[1]:
                        y -= 1
                    if x == mar.shape[2]:
                        x -= 1
                    chelsa_enc.append(torch.tensor(mar[:,y, x]))
                res.append(torch.cat(chelsa_enc))
            res = torch.stack(res).to(locs.device)
            res = self.chelsa_enc(res.float())
        else:
            if self.use_loc:
                loc = self.pos_emb(locs).squeeze(dim=1)
                return self.loc_enc(loc)
            else:
                raise ValueError("Need to use either chelsa or loc")
        if self.use_loc:
            loc = self.pos_emb(locs).squeeze(dim=1)
            loc = self.loc_enc(loc)
            res = torch.cat([res, loc], dim=1)
        return res
        
        

class CHELSATestModule(LightningModule):
    """
    """

    def __init__(
        self,
        freeze, use_chelsa, use_loc, months,
        test_cases = None,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.pos_embedding = torch.nn.Identity()
        self.freeze = freeze
        self.location_encoder = CHELSA_Loc_Enc(freeze, use_chelsa, use_loc, months)

        self.test_cases = test_cases

    def test_step(self, batch, batch_idx):

        # We only use wandb logging
        if batch_idx == 0 and self.logger and self.test_cases:
            wb = self.logger.experiment
            for _, case in self.test_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            self.location_encoder.reset_model()
                            case(self.pos_embedding, self.location_encoder, wb)


if __name__ == "__main__":
    loc_enc = CHELSA_Loc_Enc(freeze=False, use_chelsa=True, use_loc=False, months="march").to("cuda")
    inp = []
    for i in range(3):
        inp.append(torch.tensor([170, -30]))
    print(loc_enc(torch.stack(inp).to("cuda")).shape)
