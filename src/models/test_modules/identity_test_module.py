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
sys.path.append('/home/jdolli/sent-sinr/')
from sesi_utils import bilinear_interpolate

sys.path.append('/home/jdolli/chelsaCLIP/src/utils')
from positional_encoding.spheregrid import SphereGridSpatialRelationEncoder

class IdentityTestModule(LightningModule):
    """
    """

    def __init__(
        self,
        test_cases = None,
        special_mode = "None",
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if special_mode == "None":
            self.pos_embedding = torch.nn.Identity()
            self.location_encoder = torch.nn.Identity()
        elif special_mode == "SGPlus":
            print("Running SpheregridPlus position embedding")
            self.pos_embedding = SphereGridSpatialRelationEncoder(
                coord_dim= 2,
                frequency_num= 64,
                max_radius= 360,
                min_radius= 0.0003,
                freq_init= "geometric",
                device= "cuda")
            self.location_encoder = torch.nn.Identity()

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

    def forward(self, x):
        return self.location_encoder(self.pos_embedding(x))


if __name__ == "__main__":
    loc_enc = IdentityTestModule(special_mode="SGPlus")
    inp = []
    months = []
    for i in range(8192):
        inp.append(torch.tensor([170, -30]))
        months.append(torch.tensor([3]))
    print(loc_enc(torch.stack(inp)).shape)
