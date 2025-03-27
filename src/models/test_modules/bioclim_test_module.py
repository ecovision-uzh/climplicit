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

sys.path.append("/home/jdolli/sent-sinr/")
from sesi_utils import bilinear_interpolate


class BIOCLIM_Loc_Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bioclim_path = "/shares/wegner.ics.uzh/bioclim+elev/bioclim+elev/bioclim_elevation_scaled.npy"

        context_feats = np.load(bioclim_path)
        self.raster = torch.from_numpy(context_feats)
        self.raster[torch.isnan(self.raster)] = 0.0

    def forward(self, locs):
        locs[:, 1] = (locs[:, 1] + 90) / 180
        locs[:, 0] = (locs[:, 0] + 180) / 360
        locs[:, 0] = locs[:, 0] * 2 - 1
        locs[:, 1] = locs[:, 1] * 2 - 1
        return bilinear_interpolate(locs.to("cpu"), self.raster).to("cuda")


class BIOCLIMTestModule(LightningModule):
    """ """

    def __init__(
        self,
        test_cases=None,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.pos_embedding = torch.nn.Identity()
        self.location_encoder = BIOCLIM_Loc_Enc()

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
    loc_enc = BIOCLIM_Loc_Enc()
    inp = []
    for i in range(2):
        inp.append(torch.tensor([170, -30]))
