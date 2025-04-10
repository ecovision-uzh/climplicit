import torch

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

import sys

from src.utils.test_cases.util_datasets import *


class CreateMapVisual:
    def __init__(
        self, months, reduction, scope, use_months=True, pass_month_to_forward=False
    ):
        self.months = months
        self.reduction = reduction
        self.scope = scope
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

        if self.scope == "swi":
            ds = SwitzerlandDataset()
        elif self.scope == "swi_tc":
            ds = SwitzerlandDatasetTC()
        elif self.scope == "zur":
            ds = ZurichDataset()
        elif self.scope == "euro":
            ds = EuropeDataset()
        elif self.scope == "world":
            ds = WorldDataset()

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
                        with torch.no_grad():
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

        all_encs = torch.concat(all_encs, dim=0).cpu().numpy()
        # print("Nans:", np.count_nonzero(np.isnan(all_encs)), all_encs.shape)
        # all_encs = np.nan_to_num(all_encs, nan=1e-5)

        log_dict = {}

        if self.reduction == "first_three":
            red = all_encs[:, :3]
        elif self.reduction == "first":
            red = all_encs[:, 0]
        elif self.reduction == "second":
            red = all_encs[:, 1]
        elif self.reduction == "pca":
            pca = PCA(n_components=3)
            try:
                red = pca.fit_transform(all_encs)
            except:
                print(all_encs.shape)
                raise ValueError()
            if self.scope == "zur" and wb:
                try:
                    pca_2 = PCA(n_components=30)
                    pca_2.fit(all_encs)
                    evrs = pca_2.explained_variance_ratio_
                    evrs = [sum(evrs[:i]) for i in range(len(evrs))]
                    log_dict[section + "zur_pca_first_3_dim_explained_variance"] = sum(
                        pca.explained_variance_ratio_
                    )
                    fig, ax = plt.subplots()
                    ax.plot(evrs)
                    ax.axhline(y=evrs[3], color="r", linestyle="--")
                    fig.savefig("./temp_evrs.png")
                    img = wandb.Image(PIL.Image.open("./temp_evrs.png"))
                    log_dict[section + "zur_pca_explained_variance"] = img
                except:
                    log_dict[section + "zur_pca_first_3_dim_explained_variance"] = sum(
                        pca.explained_variance_ratio_
                    )
                plt.close()
            if self.scope == "world" and wb:
                try:
                    pca_2 = PCA(n_components=256)
                    pca_2.fit(all_encs[:, :256])
                    evrs = pca_2.explained_variance_ratio_
                    evrs = [sum(evrs[:i]) for i in range(len(evrs))]
                    log_dict[section + "world_pca_first_3_dim_explained_variance"] = (
                        sum(pca.explained_variance_ratio_)
                    )
                    fig, ax = plt.subplots()
                    ax.plot(evrs)
                    ax.axhline(y=evrs[3], color="r", linestyle="--")
                    fig.savefig("./temp_evrs.png")
                    img = wandb.Image(PIL.Image.open("./temp_evrs.png"))
                    log_dict[section + "world_pca_explained_variance"] = img
                except:
                    log_dict[section + "world_pca_first_3_dim_explained_variance"] = (
                        sum(pca.explained_variance_ratio_)
                    )
                plt.close()
        elif self.reduction == "tsne":
            red = TSNE(n_components=3).fit_transform(all_encs)

        for i in range(len(self.months)):
            red_month = red[len(ds) * i : len(ds) * (i + 1), :]
            red_month = (red_month - red_month.min(axis=0)) / (
                red_month.max(axis=0) - red_month.min(axis=0)
            )

            if not hasattr(ds, "land_mask"):
                red_month = red_month.reshape(ds.y_pixel, ds.x_pixel, -1)
            else:
                zeros = np.zeros((ds.y_pixel, ds.x_pixel, red_month.shape[-1]))
                zeros[ds.land_mask != 0] = red_month
                red_month = zeros

            if wb:
                fig, ax = plt.subplots(figsize=(12, 8))
                if self.scope == "swi":
                    extent = (5.933, 10.514, 45.806, 47.813)
                elif self.scope == "swi_tc":
                    extent = (5.933, 10.514, 45.806, 47.813)
                elif self.scope == "zur":
                    extent = (8.3688, 9.3906, 46.8696, 47.5432)
                elif self.scope == "euro":
                    extent = (-10.83, 32.41, 30.75, 74.00)
                elif self.scope == "world":
                    extent = (-180, 180, -90, 90)
                ax.set_xlim([extent[0], extent[1]])
                ax.set_ylim([extent[2], extent[3]])

                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                ax.imshow(red_month, extent=extent)

                fig.savefig("./temp_mv.png")
                img = wandb.Image(PIL.Image.open("./temp_mv.png"))
                log_dict[
                    section
                    + str(self.months[i])
                    + "_"
                    + self.reduction
                    + "_"
                    + self.scope
                ] = img

        if wb:
            wb.log(log_dict)
        else:
            plt.imsave(
                "./src/utils/test_cases/data/test_visual_" + self.scope + ".png",
                red_month,
            )


if __name__ == "__main__":
    import sys

    sys.path.append("/home/jdolli/chelsaCLIP/src/utils/positional_encoding")
    sys.path.append("/home/jdolli/chelsaCLIP/src/models/components")

    from spheregrid import SphereGridSpatialRelationEncoder
    from loc_encoder import SirenNet

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
