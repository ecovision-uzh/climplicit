import torch
from torcheval.metrics import R2Score

import pandas as pd

import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import sys
sys.path.append('/home/jdolli/chelsaCLIP/src/utils/test_cases')
from util_datasets import *
sys.path.append('/home/jdolli/chelsaCLIP/src/models/components')
from residual_net import *


class DS(torch.utils.data.Dataset):
    def __init__(self, ptc_path, chelsa_path):
        self.ptc = np.load(ptc_path)
        self.chelsa = np.load(chelsa_path).transpose(1,2,0).reshape(-1, 11)
    
    def __len__(self):
        return len(self.ptc)

    def __getitem__(self, idx): 
        return torch.tensor(self.ptc[idx]), torch.tensor(self.chelsa[idx], dtype=torch.float)


class Probe(torch.nn.Module):
    def __init__(self, mlp_input_len, pos_embedding, location_encoder, use_months, pass_month_to_forward=False, 
        hidden_dim=64, linear_probing=True):
        super().__init__()
        self.pos_embedding = pos_embedding.to("cuda")
        self.location_encoder = location_encoder.to("cuda")
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        if linear_probing:
            self.mlp = torch.nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, 11).to("cuda")
        else:
            # If the location encoder is month enabled, then we get its ebmedding for ["03", "06, "09, "12"] and concatenate them
            #self.mlp = Residual_Net(mlp_input_len if not use_months else mlp_input_len * 4,
            #hidden_dim = hidden_dim, layers = 2, out_dim=11, batchnorm=True).to("cuda")
            layers = []
            layers += [nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, hidden_dim, bias=True), nn.ReLU()] # Input layer
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()] * 2 # Hidden layers
            layers += [nn.Linear(hidden_dim, 11, bias=True)] # Output layer
            self.mlp = nn.Sequential(*layers)
    
    def forward(self, lonlat):
        loc = self.pos_embedding(lonlat.double()).squeeze(dim=1)

        if self.use_months:
            x = []
            for m in [3,  6,  9, 12]:
                month = torch.full([len(loc)], m).to("cuda")
                if self.pass_month_to_forward:
                    x.append(self.location_encoder(loc, month).float())
                else:
                    loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                    x.append(self.location_encoder(loc_month).float())
            x = torch.cat(x, dim=-1)
        else:
            x = self.location_encoder(loc).float()
        
        return self.mlp(x.to("cuda"))


class CHR():
    def __init__(self, ptc_path, chelsa_path, mlp_input_len, use_months, pass_month_to_forward=False, 
    verbose=False, linear_probing=True, iterations=1, map_pca=True, train_loc_enc=False):
        self.mlp_input_len = mlp_input_len
        self.verbose = verbose
        self.ptc_path = ptc_path
        self.chelsa_path = chelsa_path
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        self.map_pca = map_pca
        self.train_loc_enc = train_loc_enc
        
    def run_(self, pos_embedding, location_encoder):
        ds = DS(self.ptc_path, self.chelsa_path)
        self.train = ds
        gn = torch.Generator().manual_seed(42)
        _, self.val, _ = torch.utils.data.random_split(ds, [0.5, 0.1, 0.4], generator=gn)
        self.test = ds
        
        EPOCHS = 1000
        if self.verbose:
            EPOCHS = tqdm(range(EPOCHS))
        else:
            EPOCHS = range(EPOCHS)
            
        patience = 3
        
        model = Probe(self.mlp_input_len, pos_embedding, location_encoder, self.use_months, self.pass_month_to_forward, 
        linear_probing=self.linear_probing).to("cuda")
        
        loss_fn = torch.nn.MSELoss()
        LR = 0.001
        if self.train_loc_enc:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        BATCH_SIZE = 4096
        
        best_val_loss = 10000
        es_counter = 0
        if not self.train_loc_enc:
            train = torch.utils.data.DataLoader(dataset=self.train, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
            train_x = []
            train_y = []
            for lonlat, y in train:
                train_y.append(y)
                lonlat = lonlat.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    x = []
                    for m in [3,  6,  9, 12]:
                        month = torch.full([len(loc)], m).to("cuda")
                        if self.pass_month_to_forward:
                            x.append(model.location_encoder(loc, month).float())
                        else:
                            loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                            x.append(model.location_encoder(loc_month).float())
                    train_x.append(torch.cat(x, dim=-1).detach().cpu())
                else:
                    train_x.append(model.location_encoder(loc).float().detach().cpu())
            train_x = torch.cat(train_x)
            train_y = torch.cat(train_y)

            val = torch.utils.data.DataLoader(dataset=self.val, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
            val_x = []
            val_y = []
            for lonlat, y in val:
                val_y.append(y)
                lonlat = lonlat.to("cuda")

                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    x = []
                    for m in [3,  6,  9, 12]:
                        month = torch.full([len(loc)], m).to("cuda")
                        if self.pass_month_to_forward:
                            x.append(model.location_encoder(loc, month).float())
                        else:
                            loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                            x.append(model.location_encoder(loc_month).float())
                    val_x.append(torch.cat(x, dim=-1).detach().cpu())
                else:
                    val_x.append(model.location_encoder(loc).float().detach().cpu())
            val_x = torch.cat(val_x)
            val_y = torch.cat(val_y)

        for _ in EPOCHS:
            if not self.train_loc_enc:
                randperm = torch.randperm(len(train_x))
                train_x = train_x[randperm]
                train_y = train_y[randperm]
                
                for idx in range(len(train_x)//BATCH_SIZE + 1):
                    x, y = train_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda"), train_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda").reshape(-1)
                    optimizer.zero_grad()
                    out = model.mlp(x).float().reshape(-1)
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()

                model.mlp.eval()
                acc_loss = 0
                for idx in range(len(val_x)//BATCH_SIZE + 1):
                    x, y = val_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda"), val_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda").reshape(-1)
                    with torch.no_grad():
                        out = model.mlp(x).float().reshape(-1)
                    acc_loss += loss_fn(out, y).detach().cpu()
                acc_loss /= len(val)
                model.mlp.train()
            else:
                train = torch.utils.data.DataLoader(dataset=self.train, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

                for lonlat, y in train:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda").reshape(-1)
                    optimizer.zero_grad()
                    out = model(lonlat).float().reshape(-1)
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()
                
                val = torch.utils.data.DataLoader(dataset=self.val, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
                model.eval()
                acc_loss = 0
                for lonlat, y in val:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda").reshape(-1)
                    with torch.no_grad():
                        out = model(lonlat).float().reshape(-1)
                    acc_loss += loss_fn(out, y).detach().cpu()
                acc_loss /= len(val)
                model.train()

            scheduler.step(acc_loss)
                
            if best_val_loss - acc_loss > 0.0001:
                es_counter = 0
                best_val_loss = acc_loss
            else:
                es_counter += 1
            if self.verbose:
                EPOCHS.set_description("Val-loss: %.8f (Best: %.8f - Patience: %i)" % (acc_loss, best_val_loss, es_counter))

            if es_counter == patience:
                break
        
        model.eval()
        test = torch.utils.data.DataLoader(dataset=self.test, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
        metric = R2Score()
        # Calculate and log PSNR
        acc_loss = 0 #!Change this for when not using MSE as loss function
        for lonlat, temp in test:
            lonlat = lonlat.to("cuda")
            temp = temp.to("cuda").reshape(-1)
            
            with torch.no_grad():
                out = model(lonlat).float().reshape(-1)
            acc_loss += loss_fn(out, temp).detach().cpu()
            metric.update(out.cpu(), temp.cpu())
        max_chelsa = ds.chelsa.max()  # The maximum value of the image is unbound, thus just set to highest value in the original
        PSNR = 10 * torch.log10(np.power(max_chelsa, 2)/(acc_loss/len(test))).detach().cpu()
        return metric.compute(), PSNR, (acc_loss/len(test)).detach().cpu(), model

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        """
            ???
            :param pos_embedding: embeds [lon, lat], e.g. SH
            :param location_encoder: network that creates encoding for each location
            :param month: month to be embedded along with the location embedding

            :returns: ...
        """
        torch.manual_seed(42)

        r2s = []
        psnrs = []
        mses = []

        for _ in range(self.iterations):
            r2, psnr, mse, model = self.run_(pos_embedding, location_encoder)
            r2s.append(r2)
            psnrs.append(psnr)
            mses.append(mse)
        
        r2s = np.array(r2s)
        psnrs = np.array(psnrs)
        mses = np.array(mses)

        if wb:
            wb.log({"CHELSA Regression/chelsa_r2_mean" : r2s.mean(), "CHELSA Regression/chelsa_r2_std" : r2s.std(),
                "CHELSA Regression/chelsa_psnr_mean" : psnrs.mean(), "CHELSA Regression/chelsa_psnr_std" : psnrs.std(),
                "CHELSA Regression/chelsa_mse_mean" : mses.mean(), "CHELSA Regression/chelsa_mse_std" : mses.std()})
        else:
            print(r2s.mean(), r2s.std(), psnrs.mean(), psnrs.std(), mses.mean(), mses.std())

        if self.map_pca:
            ds = ds = SwitzerlandDatasetTC()
            dl = torch.utils.data.DataLoader(
                    dataset=ds,
                    batch_size=8196,
                    num_workers=16,
                    shuffle=False,
                )

            encodings = []
            for lonlat in dl:
                lonlat = lonlat.to("cuda")
                with torch.no_grad():
                    # Just use model from last iteration to create the map
                    # At some point maybe change this to taking the average between all models
                    encodings.append(model(lonlat).sigmoid())
            encodings = torch.cat(encodings, dim=0).detach().cpu().float()

            #print("Before PCA", encodings.min(dim=0), encodings.max(dim=0))
            
            encodings = PCA(n_components=3).fit_transform(encodings.numpy())
            encodings = (encodings - encodings.min(axis=0)) / (encodings.max(axis=0) - encodings.min(axis=0))

            #print("After PCA", encodings.min(axis=0), encodings.max(axis=0))
            
            if not hasattr(ds, 'land_mask'):
                imgs = encodings.reshape(ds.y_pixel, ds.x_pixel, 3)
            else:
                imgs = np.zeros((ds.y_pixel, ds.x_pixel, encodings.shape[-1]))
                imgs[ds.land_mask] = encodings

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim([5.933, 10.514])
            ax.set_ylim([45.806, 47.813])

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            ax.imshow(imgs, extent=(5.933, 10.514, 45.806, 47.813))
                
            if wb:
                fig.savefig("./temp_chr.png")
                img = wandb.Image(PIL.Image.open("./temp_chr.png"))
                wb.log({"CHELSA Regression/output_pca_map": img})
            else:
                fig.savefig("./temp_chr.png")

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
    reg = CHR(ptc_path="/shares/wegner.ics.uzh/CHELSA/Switzerland/input/point_to_coord.npy",
    chelsa_path="/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/03_monthly_float16.npy",
    mlp_input_len=2, use_months=False, verbose=True, iterations=1)
    reg(pos_embedding, location_encoder, None)
    