import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

import wandb

import pickle
import sys

def create_model():
    sys.path.append('/home/jdolli/csp/main/')
    import utils as ut
    from module import MultiLayerFeedForwardNN
    from SpatialRelationEncoder import GridCellSpatialRelationEncoder
    ffn = MultiLayerFeedForwardNN(
            input_dim=int(4 * 32),
            output_dim=256,
            num_hidden_layers=1,
            dropout_rate=0.5,
            hidden_dim=512,
            activation="leakyrelu",
            use_layernormalize="T",
            skip_connection = "T",
            context_str = "GridCellSpatialRelationEncoder")
    spa_enc = GridCellSpatialRelationEncoder(
        256, 
        coord_dim = 2, 
        frequency_num = 32, 
        max_radius = 360,
        min_radius = 0.1,
        freq_init = "geometric",
        ffn=ffn,
        device="cuda").to("cuda")

    file = "/home/jdolli/csp/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_contsoftmax_ratio0.050_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar"
    weights = torch.load(file)
    sd = {k[8:]: v for k, v in weights["state_dict"].items() if k.startswith("spa_enc")}
    spa_enc.load_state_dict(sd)


    return spa_enc
    """models.LocationEncoder(
                spa_enc = spa_enc, 
                num_inputs = num_inputs, 
                num_classes = num_classes, 
                num_filts = num_filts, 
                num_users = num_users).to(device)"""


class DummyEnc(torch.nn.Module):
    def __init__(self, add_multiples):
        super().__init__()
        self.add_multiples = add_multiples

        self.location_encoder = create_model()
        self.location_encoder.eval()
    
    def forward(self, x):
        if isinstance(self.add_multiples, int):
            emb = self.location_encoder(x)
            return torch.cat([emb]*self.add_multiples, dim=1)
        else:
            x = x.unsqueeze(dim=1).cpu().numpy()
            return self.location_encoder(x).squeeze(dim=1)


class CSPTestModule(LightningModule):
    """
    """

    def __init__(
        self,
        test_cases = None,
        add_multiples = None,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["location_encoder", "chelsa_encoder", "pos_embedding"])
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
