import torch
from torcheval.metrics import R2Score

import pandas as pd
import math
import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import sys

sys.path.append("/home/jdolli/chelsaCLIP/src/utils/test_cases")
from util_datasets import *

sys.path.append("/home/jdolli/chelsaCLIP/src/models/components")
from residual_net import *


class DS(torch.utils.data.Dataset):
    def __init__(self, num_samples=100000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        month = int(12 * torch.rand(1)) + 1
        while True:
            rand_feats_orig = torch.rand(2)
            theta1 = 2.0 * math.pi * rand_feats_orig[0]
            theta2 = torch.acos(2.0 * rand_feats_orig[1] - 1.0)
            lat = 1.0 - 2.0 * theta2 / math.pi
            lon = (theta1 / math.pi) - 1.0
            lat = lat * 90
            lon = lon * 180
            # Also only take samples outside of Antartica
            if globe.is_land(lat, lon) and lat > -63:
                return torch.tensor([lon, lat]), torch.tensor(month)


class Probe(torch.nn.Module):
    def __init__(
        self,
        mlp_input_len,
        pos_embedding,
        location_encoder,
        use_months,
        pass_month_to_forward=False,
        hidden_dim=64,
        linear_probing=True,
    ):
        super().__init__()
        self.pos_embedding = pos_embedding.to("cuda")
        self.location_encoder = location_encoder.to("cuda")
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        if linear_probing:
            self.mlp = torch.nn.Linear(mlp_input_len, 3).to("cuda")
        else:
            # If the location encoder is month enabled, then we get its ebmedding for ["03", "06, "09, "12"] and concatenate them
            # self.mlp = Residual_Net(mlp_input_len if not use_months else mlp_input_len * 4,
            # hidden_dim = hidden_dim, layers = 2, out_dim=11, batchnorm=True).to("cuda")
            layers = []
            layers += [
                nn.Linear(mlp_input_len, hidden_dim, bias=True),
                nn.ReLU(),
            ]  # Input layer
            layers += [
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
            ] * 2  # Hidden layers
            layers += [nn.Linear(hidden_dim, 3, bias=True)]  # Output layer
            self.mlp = nn.Sequential(*layers)

    def forward(self, lonlat, month):
        loc = self.pos_embedding(lonlat.double()).squeeze(dim=1)

        if self.use_months:
            if self.pass_month_to_forward:
                x = self.location_encoder(loc, month).float()
            else:
                loc_month = torch.concat(
                    [
                        loc,
                        torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                        torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                    ],
                    dim=-1,
                )
                x = self.location_encoder(loc_month).float()
        else:
            x = self.location_encoder(loc).float()

        return self.mlp(x.to("cuda"))


class LMR:
    def __init__(
        self,
        mlp_input_len,
        use_months,
        pass_month_to_forward=False,
        verbose=False,
        linear_probing=True,
        iterations=1,
        map_pca=True,
        train_loc_enc=False,
    ):
        self.mlp_input_len = mlp_input_len
        self.verbose = verbose
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        self.map_pca = map_pca
        self.train_loc_enc = train_loc_enc

    def run_(self, pos_embedding, location_encoder):
        ds = DS()
        self.train = ds
        gn = torch.Generator().manual_seed(42)
        _, self.val, _ = torch.utils.data.random_split(
            ds, [0.5, 0.1, 0.4], generator=gn
        )
        self.test = ds

        EPOCHS = 5000
        if self.verbose:
            EPOCHS = tqdm(range(EPOCHS))
        else:
            EPOCHS = range(EPOCHS)

        patience = 3

        model = Probe(
            self.mlp_input_len,
            pos_embedding,
            location_encoder,
            self.use_months,
            self.pass_month_to_forward,
            linear_probing=self.linear_probing,
        ).to("cuda")

        loss_fn = torch.nn.MSELoss()
        LR = 0.01
        if self.train_loc_enc:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        BATCH_SIZE = 4096

        best_val_loss = 10000
        es_counter = 0
        if not self.train_loc_enc:
            train = torch.utils.data.DataLoader(
                dataset=self.train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
            )
            train_lonlat = []
            train_month = []
            train_x = []
            for lonlat, month in train:
                train_lonlat.append(lonlat)
                train_month.append(month)
                lonlat = lonlat.to("cuda")
                month = month.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    if self.pass_month_to_forward:
                        train_x.append(
                            model.location_encoder(loc, month).float().detach().cpu()
                        )
                    else:
                        loc_month = torch.concat(
                            [
                                loc,
                                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                            ],
                            dim=-1,
                        )
                        train_x.append(
                            model.location_encoder(loc_month).float().detach().cpu()
                        )
                else:
                    train_x.append(model.location_encoder(loc).float().detach().cpu())
            train_lonlat = torch.cat(train_lonlat)
            train_month = torch.cat(train_month)
            train_x = torch.cat(train_x)

            val = torch.utils.data.DataLoader(
                dataset=self.val, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
            )
            val_lonlat = []
            val_month = []
            val_x = []
            for lonlat, month in val:
                val_lonlat.append(lonlat)
                val_month.append(month)
                lonlat = lonlat.to("cuda")
                month = month.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    if self.pass_month_to_forward:
                        val_x.append(
                            model.location_encoder(loc, month).float().detach().cpu()
                        )
                    else:
                        loc_month = torch.concat(
                            [
                                loc,
                                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                            ],
                            dim=-1,
                        )
                        val_x.append(
                            model.location_encoder(loc_month).float().detach().cpu()
                        )
                else:
                    val_x.append(model.location_encoder(loc).float().detach().cpu())
            val_lonlat = torch.cat(val_lonlat)
            val_month = torch.cat(val_month)
            val_x = torch.cat(val_x)

        for _ in EPOCHS:
            if not self.train_loc_enc:
                randperm = torch.randperm(len(train_x))
                train_x = train_x[randperm]
                train_lonlat = train_lonlat[randperm]
                train_month = train_month[randperm]

                for idx in range(len(train_x) // BATCH_SIZE + 1):
                    x = train_x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to("cuda")
                    lonlat = train_lonlat[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    )
                    month = train_month[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    )
                    optimizer.zero_grad()
                    out = model.mlp(x).float().reshape(-1)
                    loss = loss_fn(
                        out,
                        torch.cat([lonlat, month.reshape(-1, 1)], dim=-1).reshape(-1),
                    )
                    loss.backward()
                    optimizer.step()

                model.mlp.eval()
                acc_loss = 0
                for idx in range(len(val_x) // BATCH_SIZE + 1):
                    x = val_x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to("cuda")
                    lonlat = val_lonlat[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    )
                    month = val_month[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    )
                    with torch.no_grad():
                        out = model.mlp(x).float().reshape(-1)
                    acc_loss += (
                        loss_fn(
                            out,
                            torch.cat([lonlat, month.reshape(-1, 1)], dim=-1).reshape(
                                -1
                            ),
                        )
                        .detach()
                        .cpu()
                    )
                acc_loss /= len(val)
                model.mlp.train()
            else:
                train = torch.utils.data.DataLoader(
                    dataset=self.train,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                    shuffle=True,
                )

                for lonlat, month in train:
                    lonlat = lonlat.to("cuda")
                    month = month.to("cuda").reshape(-1)
                    optimizer.zero_grad()
                    out = model(lonlat, month).float().reshape(-1)
                    loss = loss_fn(
                        out,
                        torch.cat([lonlat, month.reshape(-1, 1)], dim=-1).reshape(-1),
                    )
                    loss.backward()
                    optimizer.step()

                val = torch.utils.data.DataLoader(
                    dataset=self.val,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                    shuffle=False,
                )
                model.eval()
                acc_loss = 0
                for lonlat, month in val:
                    lonlat = lonlat.to("cuda")
                    month = month.to("cuda").reshape(-1)
                    with torch.no_grad():
                        out = model(lonlat, month).float().reshape(-1)
                    acc_loss += (
                        loss_fn(
                            out,
                            torch.cat([lonlat, month.reshape(-1, 1)], dim=-1).reshape(
                                -1
                            ),
                        )
                        .detach()
                        .cpu()
                    )
                acc_loss /= len(val)
                model.train()

            scheduler.step(acc_loss)

            if best_val_loss - acc_loss > 0.001:
                es_counter = 0
                best_val_loss = acc_loss
            else:
                es_counter += 1
            if self.verbose:
                EPOCHS.set_description(
                    "Val-loss: %.8f (Best: %.8f - Patience: %i)"
                    % (acc_loss, best_val_loss, es_counter)
                )

            if es_counter == patience:
                break

        model.eval()
        test = torch.utils.data.DataLoader(
            dataset=self.test, batch_size=BATCH_SIZE, num_workers=4, shuffle=False
        )
        lonlat_metric = R2Score()
        month_metric = R2Score()
        # Calculate and log PSNR
        acc_loss = 0  #!Change this for when not using MSE as loss function
        for lonlat, month in test:
            lonlat = lonlat.to("cuda")
            month = month.to("cuda")

            with torch.no_grad():
                out = model(lonlat, month).float()
            lonlat_metric.update(out[:, :2].cpu().reshape(-1), lonlat.cpu().reshape(-1))
            month_metric.update(out[:, 2].cpu().reshape(-1), month.cpu().reshape(-1))
        return lonlat_metric.compute(), month_metric.compute(), model

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        """
        ???
        :param pos_embedding: embeds [lon, lat], e.g. SH
        :param location_encoder: network that creates encoding for each location
        :param month: month to be embedded along with the location embedding

        :returns: ...
        """
        torch.manual_seed(42)

        lonlat_r2s = []
        month_r2s = []

        for _ in range(self.iterations):
            lonlat_r2, month_r2, model = self.run_(pos_embedding, location_encoder)
            lonlat_r2s.append(lonlat_r2)
            month_r2s.append(month_r2)

        lonlat_r2s = np.array(lonlat_r2s)
        month_r2s = np.array(month_r2s)

        if wb:
            wb.log(
                {
                    "Loc-Month Regression/lonlat_r2_mean": lonlat_r2s.mean(),
                    "Loc-Month Regression/lonlat_r2_std": lonlat_r2s.std(),
                    "Loc-Month Regression/month_r2_mean": month_r2s.mean(),
                    "Loc-Month Regression/month_r2_std": month_r2s.std(),
                }
            )
        else:
            print(
                lonlat_r2s.mean(), lonlat_r2s.std(), month_r2s.mean(), month_r2s.std()
            )

        ds = WorldDataset()
        dl = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=8196,
            num_workers=4,
            shuffle=False,
        )

        encodings = []
        for lonlat in dl:
            lonlat = lonlat.to("cuda")
            MONTH = 3
            month = torch.full([len(lonlat)], MONTH).to("cuda")
            with torch.no_grad():
                encodings.append(model(lonlat, month))
        encodings = torch.cat(encodings, dim=0).detach().cpu().float()

        imgs = np.zeros((ds.y_pixel, ds.x_pixel, 3))
        imgs[ds.land_mask != 0] = encodings

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        im = ax.imshow(imgs[:, :, 0], extent=(-180, 180, -90, 90))
        fig.colorbar(im, ax=ax)
        if wb:
            fig.savefig("./temp_lon_r.png")
            img = wandb.Image(PIL.Image.open("./temp_lon_r.png"))
            wb.log({"Loc-Month Regression/lon_map": img})
        else:
            fig.savefig("./temp_lon_r.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        im = ax.imshow(imgs[:, :, 1], extent=(-180, 180, -90, 90))
        fig.colorbar(im, ax=ax)
        if wb:
            fig.savefig("./temp_lat_r.png")
            img = wandb.Image(PIL.Image.open("./temp_lat_r.png"))
            wb.log({"Loc-Month Regression/lat_map": img})
        else:
            fig.savefig("./temp_lat_r.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        im = ax.imshow(imgs[:, :, 2], extent=(-180, 180, -90, 90))
        fig.colorbar(im, ax=ax)
        if wb:
            fig.savefig("./temp_month_r.png")
            img = wandb.Image(PIL.Image.open("./temp_month_r.png"))
            wb.log({"Loc-Month Regression/month_map": img})
        else:
            fig.savefig("./temp_month_r.png")

        plt.close()


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


if __name__ == "__main__":
    pos_embedding = FakePosEmb()
    location_encoder = FakeLocEnc()

    # Add in info where to download this data
    reg = LMR(mlp_input_len=4, use_months=True, verbose=True, iterations=1)
    reg(pos_embedding, location_encoder, None)
