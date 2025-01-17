import torch
from torcheval.metrics import R2Score

import pandas as pd

import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/jdolli/chelsaCLIP/src/utils/test_cases')
from util_datasets import *
sys.path.append('/home/jdolli/chelsaCLIP/src/models/components')
from residual_net import *

from torch import nn


class DS(torch.utils.data.Dataset):
    def __init__(self, file_path, norm_data=True):
        self.data = np.array(pd.read_csv(file_path))[:,[0,1,8]].astype("float")
        self.data[:,2] = (self.data[:,2] - self.data[:,2].mean()) / self.data[:,2].std()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return torch.tensor(self.data[idx, :2]), torch.tensor(self.data[idx, 2], dtype=torch.float)


class Probe(torch.nn.Module):
    def __init__(self, mlp_input_len, pos_embedding, location_encoder, use_months, pass_month_to_forward=False, 
        hidden_dim=64, linear_probing=True):
        super().__init__()
        self.pos_embedding = pos_embedding.to("cuda")
        self.location_encoder = location_encoder.to("cuda")
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        if linear_probing:
            self.mlp = torch.nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, 1).to("cuda")
        else:
            # If the location encoder is month enabled, then we get its ebmedding for ["03", "06, "09, "12"] and concatenate them
            #self.mlp = Residual_Net(mlp_input_len if not use_months else mlp_input_len * 4,
            #hidden_dim = hidden_dim, layers = 2, out_dim=1, batchnorm=True).to("cuda")
            layers = []
            layers += [nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, hidden_dim, bias=True), nn.ReLU()] # Input layer
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()] * 2 # Hidden layers
            layers += [nn.Linear(hidden_dim, 1, bias=True)] # Output layer
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


class CalR():
    def __init__(self, data_path, mlp_input_len, use_months, pass_month_to_forward=False, 
    verbose=True, linear_probing=True, iterations=1, train_loc_enc=False):
        self.mlp_input_len = mlp_input_len
        self.verbose = verbose
        self.data_path = data_path
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        self.train_loc_enc = train_loc_enc
        
    def run_(self, pos_embedding, location_encoder):
        ds = DS(self.data_path)
        gn = torch.Generator().manual_seed(42)
        self.train, self.val, self.test = torch.utils.data.random_split(ds, [0.5, 0.1, 0.4], generator=gn)
        
        EPOCHS = 5000
        if self.verbose:
            EPOCHS = tqdm(range(EPOCHS))
        else:
            EPOCHS = range(EPOCHS)
            
        patience = 5
        
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
                    x, y = train_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda"), train_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda")
                    optimizer.zero_grad()
                    out = model.mlp(x).float().reshape(-1)
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()

                model.mlp.eval()
                acc_loss = 0
                for idx in range(len(val_x)//BATCH_SIZE + 1):
                    x, y = val_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda"), val_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda")
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
                
            if best_val_loss - acc_loss > 0.001:
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
        for lonlat, temp in test:
            lonlat = lonlat.to("cuda")
            temp = temp.to("cuda").reshape(-1)

            with torch.no_grad():
                out = model(lonlat).float().reshape(-1)
            metric.update(out.cpu(), temp.cpu())
        return metric.compute(), model

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

        for i in range(self.iterations):
            r2, model = self.run_(pos_embedding, location_encoder)
            r2s.append(r2)
        
        r2s = np.array(r2s)

        if wb:
            wb.log({"SatCLIP Regressions/cali_housing_r2_mean" : r2s.mean(), "SatCLIP Regressions/cali_housing_r2_std" : r2s.std()})
        else:
            print(r2s.mean(), r2s.std())
        
            
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
    reg = CalR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/housing.csv', mlp_input_len=4,
    use_months=True, verbose=True, iterations=2)
    reg(pos_embedding, location_encoder, None)
    