import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

import wandb

import pickle
import sys


class DummyEnc(torch.nn.Module):
    def __init__(self, add_multiples):
        super().__init__()
        self.add_multiples = add_multiples

        sys.path.append("/home/jdolli/satclip/satclip")
        from load import get_satclip

        self.location_encoder = get_satclip(
            "/home/jdolli/chelsaCLIP/src/utils/test_cases/data/satclip-resnet18-l40.ckpt",
            device="cuda",
        )  # Only loads location encoder by default
        self.location_encoder.eval()

    def forward(self, x):
        if isinstance(self.add_multiples, int):
            emb = self.location_encoder(x)
            return torch.cat([emb] * self.add_multiples, dim=1)
        else:
            return self.location_encoder(x)


class SatCLIPTestModule(LightningModule):
    """ """

    def __init__(
        self,
        test_cases=None,
        add_multiples=False,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["location_encoder", "chelsa_encoder", "pos_embedding"]
        )
        self.pos_embedding = torch.nn.Identity()
        self.location_encoder = DummyEnc(add_multiples)

        self.test_cases = test_cases

    def test_step(self, batch, batch_idx):

        # We only use wandb logging
        if batch_idx == 0 and self.logger and self.test_cases:
            wb = self.logger.experiment
            for _, case in self.test_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            pe_copy = self.pos_embedding
                            le_copy = self.location_encoder
                            case(pe_copy, le_copy, wb)


if __name__ == "__main__":
    loc_enc = DummyEnc(add_multiples=4)
    inp = []
    for i in range(3):
        inp.append(torch.tensor([170.0, -30], dtype=torch.double))
    print(loc_enc(torch.stack(inp).to("cuda")).shape)
