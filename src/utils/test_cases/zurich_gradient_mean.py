import torch

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import wandb

import numpy as np
import matplotlib.pyplot as plt

import sys

from src.utils.test_cases.util_datasets import *


class ZGM:
    def __init__(self, months, use_months=True, pass_month_to_forward=False):
        self.months = months
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        """
        Creates a reduction to one or three dimenions of abstract embeddings over Switzerland to show level of detail of loc_month_embedder.
        :param pos_embedding: embeds [lon, lat], e.g. SH
        :param location_encoder: network that creates encoding for each location
        :param month: month to be embedded along with the location embedding

        :returns: image of shape [x_pixel, y_pixel, 1 or 3]
        """

        ds = ZurichDataset()

        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=8196,
            num_workers=16,
            shuffle=False,
        )

        all_encs = []
        for m in self.months:
            encodings = []
            for lonlat in dl:
                # Get pos embedding
                # For some reason Spherical Harmonics require double instead of float :8
                loc = pos_embedding(lonlat.double())
                # Put on the same device
                loc = loc.squeeze(dim=1).to("cuda")
                # Append month
                if self.use_months:
                    month = torch.full([len(loc)], m).to("cuda")
                    if self.pass_month_to_forward:
                        encodings.append(location_encoder(loc, month))
                    else:
                        loc = torch.concat(
                            [
                                loc,
                                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                            ],
                            dim=-1,
                        )
                        # Get encoding from network
                        with torch.no_grad():
                            encodings.append(location_encoder(loc))
                else:
                    with torch.no_grad():
                        encodings.append(location_encoder(loc))
            all_encs.append(torch.concat(encodings, dim=0))

        all_encs = torch.stack(all_encs, dim=0).cpu().numpy()
        # Shape: (2, 92400, 256)
        all_encs = all_encs.reshape(
            all_encs.shape[0], ds.y_pixel, ds.x_pixel, all_encs.shape[-1]
        )
        grads = np.array(np.gradient(all_encs, axis=(0, 1, 2)))
        # Shape: (3, 2, 300, 308, 256)
        gnorm = np.sqrt(grads**2)
        sharpness = gnorm.mean(axis=-1).astype("float64")

        log_dict = {
            "sharpness/" + "mean grad zurich": sharpness.mean(),
            "sharpness/" + "std grad zurich": sharpness.std(),
            "sharpness/" + "max grad zurich": sharpness.max(),
        }
        if wb:
            wb.log(log_dict)


if __name__ == "__main__":
    import sys

    sys.path.append("/home/jdolli/chelsaCLIP/src/utils/positional_encoding")
    sys.path.append("/home/jdolli/chelsaCLIP/src/models/components")

    from spheregrid import SphereGridSpatialRelationEncoder
    from location_encoder import SirenNet

    import matplotlib.pyplot as plt

    class FakePosEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass

        def forward(self, x):
            return x

    class FakeLocEnc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass

        def forward(self, x):
            return x

    pos_embedding = SphereGridSpatialRelationEncoder(
        coord_dim=2,
        frequency_num=64,
        max_radius=360,
        min_radius=0.0003,
        freq_init="geometric",
        device="cuda",
    )
    location_encoder = SirenNet(
        dim_in=386, dim_hidden=128, dim_out=32, num_layers=2
    ).to("cuda")
    # These identities showed, that the ordering in the end is indeed correct
    pos_embedding = FakePosEmb()
    location_encoder = FakeLocEnc()

    vis = CreateMapVisual([1], "pca", "zur")
    vis(pos_embedding, location_encoder, None)
    vis = CreateMapVisual([1], "pca", "swi")
    vis(pos_embedding, location_encoder, None)
    vis = CreateMapVisual([1], "pca", "swi_tc")
    vis(pos_embedding, location_encoder, None)
    vis = CreateMapVisual([1], "pca", "euro")
    vis(pos_embedding, location_encoder, None)
    vis = CreateMapVisual([1], "pca", "world")
    vis(pos_embedding, location_encoder, None)
