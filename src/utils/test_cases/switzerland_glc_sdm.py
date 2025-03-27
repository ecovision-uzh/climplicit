import torch

import pandas as pd

import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/jdolli/chelsaCLIP/src/utils/test_cases")
from util_datasets import *

sys.path.append("/home/jdolli/chelsaCLIP/src/models/components")
from residual_net import *

from torch import nn


class SW_PO_DS(torch.utils.data.Dataset):
    def __init__(self, PO_path):
        self.data = pd.read_csv(PO_path, sep=";", header="infer", low_memory=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data.iloc[idx]

        return (
            torch.tensor([datum["lon"], datum["lat"]]),
            int(datum["date"].split("-")[1]),
            datum["speciesId"],
        )


def top_k_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Taken from brand90 on https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SDM(torch.nn.Module):
    def __init__(
        self,
        mlp_input_len,
        pos_embedding,
        location_encoder,
        use_months,
        pass_month_to_forward=False,
        hidden_dim=512,
        linear_probing=True,
    ):
        super().__init__()
        self.pos_embedding = pos_embedding.to("cuda")
        self.location_encoder = location_encoder.to("cuda")
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        if linear_probing:
            self.mlp = torch.nn.Linear(mlp_input_len, 10040).to("cuda")
        else:
            # self.mlp = Residual_Net(mlp_input_len, hidden_dim = 256, layers = 2, out_dim=10040, batchnorm=False).to("cuda")
            layers = []
            layers += [
                nn.Linear(mlp_input_len, hidden_dim, bias=True),
                nn.ReLU(),
            ]  # Input layer
            layers += [
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
            ] * 2  # Hidden layers
            layers += [nn.Linear(hidden_dim, 10040, bias=True)]  # Output layer
            self.mlp = nn.Sequential(*layers)

    def forward(self, lonlat, month):
        loc = self.pos_embedding(lonlat.double()).squeeze(dim=1)
        if self.use_months:
            if self.pass_month_to_forward:
                x = self.location_encoder(loc, month).float()
            else:
                loc = torch.concat(
                    [
                        loc,
                        torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                        torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                    ],
                    dim=-1,
                )
                x = self.location_encoder(loc).float()
        else:
            x = self.location_encoder(loc).float()
        return self.mlp(x.to("cuda"))


class SW_SDM:
    def __init__(
        self,
        PO_path,
        mlp_input_len,
        use_months=False,
        verbose=False,
        most_common_species_map=False,
        pass_month_to_forward=False,
        linear_probing=False,
        iterations=1,
        train_loc_enc=False,
    ):
        self.use_months = use_months
        self.mlp_input_len = mlp_input_len
        self.verbose = verbose
        self.most_common_species_map = most_common_species_map
        self.PO_path = PO_path
        self.pass_month_to_forward = pass_month_to_forward
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.train_loc_enc = train_loc_enc

    def run_(self, pos_embedding, location_encoder):
        po_ds = SW_PO_DS(self.PO_path)
        gn = torch.Generator().manual_seed(42)
        self.train, self.val, self.test = torch.utils.data.random_split(
            po_ds, [0.7, 0.05, 0.25], generator=gn
        )

        EPOCHS = 400
        if self.verbose:
            EPOCHS = tqdm(range(EPOCHS))
        else:
            EPOCHS = range(EPOCHS)

        patience = 3

        model = SDM(
            self.mlp_input_len,
            pos_embedding,
            location_encoder,
            self.use_months,
            self.pass_month_to_forward,
            linear_probing=self.linear_probing,
        ).to("cuda")

        def sinr_loss(out, labels, model):
            N = 47.813
            W = 5.933
            S = 45.806
            E = 10.514
            lon = torch.rand(len(out)) * (E - W) + W
            lat = torch.rand(len(out)) * (N - S) + S
            random_lonlat = torch.stack([lon, lat], dim=1).to("cuda")
            random_month = torch.randint(1, 13, (len(out),)).to("cuda")
            rand_pred = torch.sigmoid(model(random_lonlat, random_month))
            loc_pred = torch.sigmoid(out)
            assert rand_pred.shape == loc_pred.shape

            inds = torch.arange(len(labels))

            loss_pos = -torch.log((1 - loc_pred) + 1e-5)
            loss_bg = -torch.log((1 - rand_pred) + 1e-5)
            loss_pos[inds, labels] = 2048 * -torch.log(loc_pred[inds, labels] + 1e-5)

            return loss_pos.mean() + loss_bg.mean()

        loss_fn = sinr_loss
        # loss_fn = torch.nn.CrossEntropyLoss()
        LR = 0.001
        if self.train_loc_enc:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

        BATCH_SIZE = 4096

        best_val_loss = 100
        es_counter = 0
        if not self.train_loc_enc:
            train = torch.utils.data.DataLoader(
                dataset=self.train, batch_size=BATCH_SIZE, num_workers=1, shuffle=True
            )
            train_x = []
            train_y = []
            for lonlat, month, y in train:
                train_y.append(y)
                lonlat = lonlat.to("cuda")
                month = month.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    if self.pass_month_to_forward:
                        train_x.append(
                            model.location_encoder(loc, month).float().detach().cpu()
                        )
                    else:
                        loc = torch.concat(
                            [
                                loc,
                                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                            ],
                            dim=-1,
                        )
                        train_x.append(
                            model.location_encoder(loc).float().detach().cpu()
                        )
                else:
                    train_x.append(model.location_encoder(loc).float().detach().cpu())
            train_x = torch.cat(train_x)
            train_y = torch.cat(train_y)

            val = torch.utils.data.DataLoader(
                dataset=self.val, batch_size=BATCH_SIZE, num_workers=1, shuffle=True
            )
            val_x = []
            val_y = []
            for lonlat, month, y in val:
                val_y.append(y)
                lonlat = lonlat.to("cuda")
                month = month.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    if self.pass_month_to_forward:
                        val_x.append(
                            model.location_encoder(loc, month).float().detach().cpu()
                        )
                    else:
                        loc = torch.concat(
                            [
                                loc,
                                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                            ],
                            dim=-1,
                        )
                        val_x.append(model.location_encoder(loc).float().detach().cpu())
                else:
                    val_x.append(model.location_encoder(loc).float().detach().cpu())
            val_x = torch.cat(val_x)
            val_y = torch.cat(val_y)

        for _ in EPOCHS:
            if not self.train_loc_enc:
                randperm = torch.randperm(len(train_x))
                train_x = train_x[randperm]
                train_y = train_y[randperm]

                for idx in range(len(train_x) // BATCH_SIZE + 1):
                    x, y = train_x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    ), train_y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to("cuda")
                    optimizer.zero_grad()
                    out = model.mlp(x).float()
                    loss = loss_fn(out, y, model)
                    # loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()

                model.mlp.eval()
                acc_loss = 0
                for idx in range(len(val_x) // BATCH_SIZE + 1):
                    x, y = val_x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to(
                        "cuda"
                    ), val_y[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE].to("cuda")
                    with torch.no_grad():
                        out = model.mlp(x).float()
                    acc_loss += loss_fn(out, y, model).detach().cpu()
                    # acc_loss += loss_fn(out, y)
                acc_loss /= len(val)
                model.mlp.train()

            else:
                train = torch.utils.data.DataLoader(
                    dataset=self.train,
                    batch_size=BATCH_SIZE,
                    num_workers=1,
                    shuffle=True,
                )

                for lonlat, month, y in train:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda")
                    optimizer.zero_grad()
                    out = model(lonlat, month).float()
                    loss = loss_fn(out, y, model)
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
                for lonlat, month, y in val:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda")
                    with torch.no_grad():
                        out = model(lonlat, month).float()
                    acc_loss += loss_fn(out, y, model).detach().cpu()
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
        top1, top5, top10 = 0, 0, 0
        test = torch.utils.data.DataLoader(
            dataset=self.test, batch_size=BATCH_SIZE, num_workers=16, shuffle=False
        )
        for lonlat, month, speciesId in test:
            lonlat = lonlat.to("cuda")
            month = month.to("cuda")
            speciesId = speciesId.to("cuda")

            with torch.no_grad():
                out = model(lonlat, month).float()
            tops = top_k_accuracy(out, speciesId, topk=(1, 5, 10))
            top1 += tops[0]
            top5 += tops[1]
            top10 += tops[2]
        top1 /= len(test)
        top5 /= len(test)
        top10 /= len(test)

        return top1, top5, top10, model

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        """
        ???
        :param pos_embedding: embeds [lon, lat], e.g. SH
        :param location_encoder: network that creates encoding for each location
        :param month: month to be embedded along with the location embedding

        :returns: ...
        """
        top1s, top5s, top10s = [], [], []

        for i in range(self.iterations):
            top1, top5, top10, model = self.run_(pos_embedding, location_encoder)
            top1s.append(top1.cpu())
            top5s.append(top5.cpu())
            top10s.append(top10.cpu())

        top1s = np.array(top1s)
        top5s = np.array(top5s)
        top10s = np.array(top10s)

        if wb:
            wb.log(
                {
                    "SW SDM/top1_mean": top1s.mean(),
                    "SW SDM/top5_mean": top5s.mean(),
                    "SW SDM/top10_mean": top10s.mean(),
                    "SW SDM/top1_std": top1s.std(),
                    "SW SDM/top5_std": top5s.std(),
                    "SW SDM/top10_std": top10s.std(),
                }
            )
        else:
            print(
                top1s.mean(),
                top5s.mean(),
                top10s.mean(),
                top1s.std(),
                top5s.std(),
                top10s.std(),
            )

        if self.most_common_species_map:
            ds = SwitzerlandDataset()
            dl = torch.utils.data.DataLoader(
                dataset=ds,
                batch_size=8196,
                num_workers=16,
                shuffle=False,
            )

            po_ds = SW_PO_DS(self.PO_path)

            for m in ["03", "06", "09"]:
                encodings = []
                for lonlat in dl:
                    lonlat = lonlat.to("cuda")
                    month = torch.full([len(lonlat)], int(m)).to("cuda")
                    with torch.no_grad():
                        # Just use model from last iteration to create the map
                        # At some point maybe change this to taking the average between all models
                        encodings.append(
                            model(lonlat, month).sigmoid().detach().cpu().float()
                        )
                encodings = torch.cat(encodings, dim=0)

                if not hasattr(ds, "land_mask"):
                    imgs = encodings.reshape(ds.y_pixel, ds.x_pixel, -1)
                else:
                    imgs = np.zeros((ds.y_pixel, ds.x_pixel, encodings.shape[-1]))
                    imgs[ds.land_mask != 0] = encodings

                sid = 3217  # 3217 most common species in switzerland po data
                img = imgs[:, :, sid]

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_xlim([5.933, 10.514])
                ax.set_ylim([45.806, 47.813])

                im = ax.imshow(
                    img,
                    extent=(5.933, 10.514, 45.806, 47.813),
                    vmin=0,
                    vmax=1,
                    cmap=plt.cm.plasma,
                )
                sp_cond = po_ds.data["speciesId"] == sid
                m_cond = po_ds.data["dayOfYear"] // 31 == int(m)
                ax.scatter(
                    po_ds.data[sp_cond & m_cond]["lon"],
                    po_ds.data[sp_cond & m_cond]["lat"],
                    c="green",
                    alpha=1,
                    s=3,
                )

                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                fig.colorbar(im, ax=ax)

                if wb:
                    fig.savefig("./temp_mcsm.png")
                    img = wandb.Image(PIL.Image.open("./temp_mcsm.png"))
                    wb.log({"SW SDM/" + str(sid) + "_" + str(m): img})
                else:
                    fig.savefig("./temp_mcsm.png")

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

    sw_sdm = SW_SDM(
        "/shares/wegner.ics.uzh/glc23_data/Switzerland_PO.csv",
        mlp_input_len=2,
        use_months=False,
        verbose=True,
        most_common_species_map=True,
        iterations=1,
    )
    sw_sdm(pos_embedding, location_encoder, None)
