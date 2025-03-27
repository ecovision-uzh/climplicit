import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

import wandb

import pickle
import sys


def create_model():
    sys.path.append("/home/jdolli/sinr/")
    from sinr_models import get_model
    import sinr_utils

    file = "/home/jdolli/sinr/pretrained_models/model_an_full_input_enc_sin_cos_hard_cap_num_per_class_1000.pt"
    train_params = torch.load(file, map_location="cuda")
    model = get_model(train_params["params"])
    model.load_state_dict(train_params["state_dict"], strict=True)
    enc = sinr_utils.CoordEncoder(train_params["params"]["input_enc"])

    return enc, model.to("cuda")


class DummyEnc(torch.nn.Module):
    def __init__(self, add_multiples):
        super().__init__()
        self.add_multiples = add_multiples

        self.location_encoder, self.model = create_model()
        self.model.eval()

    def forward(self, x):
        if isinstance(self.add_multiples, int):
            emb = self.location_encoder(x)
            return torch.cat([emb] * self.add_multiples, dim=1)
        else:
            x = self.location_encoder.encode(x.float()).to("cuda")
            return self.model(x, return_feats=True)


class SINRTestModule(LightningModule):
    """ """

    def __init__(
        self,
        test_cases=None,
        add_multiples=None,
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
    loc_enc = DummyEnc(add_multiples=None)
    inp = []
    for i in range(3):
        inp.append(torch.tensor([170.0, -30], dtype=torch.double))
    print(loc_enc(torch.stack(inp).to("cuda")).shape)
