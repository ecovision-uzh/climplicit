import torch

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

import matplotlib.lines as mlines

from torch import nn

import ignite


class DS(torch.utils.data.Dataset):
    def __init__(self, file_path, mode):
        self.data = pd.read_csv(file_path, low_memory=False).fillna(55)
        self.mode = mode
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        datum = self.data.iloc[idx]
        if self.mode == "biomes":
            Id =  int(datum["BIOME"])
        else:
            Id = int(datum["ECO_ID"])
            if Id == -9999:
                Id = 0
            elif Id == -9998:
                Id = 1
        return torch.tensor([datum["LON"], datum["LAT"]]), torch.tensor(Id)

class Sample_DS(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.data = samples
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        datum = self.data.iloc[idx]
        Id =  int(datum["BIOME"])
        return torch.tensor([datum["LON"], datum["LAT"]]), torch.tensor(Id)

    
def top_k_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Taken from brand90 on https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b"""
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

class PerClassMetrics():
    def __init__(self, num_classes):
        self.pc_tp = torch.zeros((num_classes)).to("cuda")
        self.pc_true = torch.zeros((num_classes)).to("cuda")
        self.pc_positive = torch.zeros((num_classes)).to("cuda")
        self.num_samples = 0

    def update(self, out, Id):
        label = torch.zeros_like(out).to("cuda")
        label[np.arange(len(label)), Id] = 1
        out_max = torch.zeros_like(out).to("cuda")
        out_max[np.arange(len(label)), torch.argmax(out, dim=1)] = 1
        self.pc_tp += (label * out_max).sum(dim=0) 
        self.pc_true += label.sum(dim=0)
        self.pc_positive += out_max.sum(dim=0)
        self.num_samples += len(out)

    def calculate(self):
        recall = self.pc_tp/self.pc_true
        precision = self.pc_tp/self.pc_positive
        #recall = recall[self.pc_true > 0]
        #precision = precision[self.pc_positive > 0]
        return 2 * (precision * recall) / (precision + recall)
        #return recall, precision

class Probe(torch.nn.Module):
    def __init__(self, mlp_input_len, pos_embedding, location_encoder, use_months, out_dim, pass_month_to_forward=False, 
        hidden_dim=64, linear_probing=True):
        super().__init__()
        self.pos_embedding = pos_embedding.to("cuda")
        self.location_encoder = location_encoder.to("cuda")
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        if linear_probing:
            self.mlp = torch.nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, out_dim).to("cuda")
        else:
            # If the location encoder is month enabled, then we get its ebmedding for ["03", "06, "09, "12"] and concatenate them
            #self.mlp = Residual_Net(mlp_input_len if not use_months else mlp_input_len * 4,
            #hidden_dim = hidden_dim, layers = 2, out_dim=out_dim, batchnorm=True).to("cuda")
            layers = []
            layers += [nn.Linear(mlp_input_len if not use_months else mlp_input_len * 4, hidden_dim, bias=True), nn.ReLU()] # Input layer
            layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()] * 2 # Hidden layers
            layers += [nn.Linear(hidden_dim, out_dim, bias=True)] # Output layer
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
            """# TODO: Throwing out the four month append
            month = torch.full([len(loc)], MONTH).to("cuda")
            if self.pass_month_to_forward:
                x = self.location_encoder(loc, month).float()
            else:
                loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                x = self.location_encoder(loc_month).float()"""
        else:
            x = self.location_encoder(loc).float()
        
        return self.mlp(x.to("cuda"))


class BEC():
    def __init__(self, file_path, mode, mlp_input_len, use_months=False, verbose=False,
    lake_victoria_map = True, plot_confusion_matrix=True, pass_month_to_forward=False, linear_probing=True, iterations=1, epochs=1000, biome_tsne = True,
    track_failure_areas = True, train_loc_enc=False):
        self.use_months = use_months
        self.mlp_input_len = mlp_input_len
        self.mode = mode
        self.verbose = verbose
        self.lake_victoria_map = lake_victoria_map
        self.file_path = file_path
        self.pass_month_to_forward = pass_month_to_forward
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.epochs = epochs
        self.biome_tsne = biome_tsne
        self.track_failure_areas = track_failure_areas
        self.plot_confusion_matrix = plot_confusion_matrix
        self.train_loc_enc = train_loc_enc

        self.labels =    ["Tropical & Subtropical Moist Broadleaf Forests",
            "Tropical & Subtropical Dry Broadleaf Forests",
            "Tropical & Subtropical Coniferous Forests",
            "Temperate Broadleaf & Mixed Forests",
            "Temperate Conifer Forests",
            "Boreal Forests/Taiga",
            "Tropical & Subtropical Grasslands, Savannas & Shrublands",
            "Temperate Grasslands, Savannas & Shrublands",
            "Flooded Grasslands & Savannas",
            "Montane Grasslands & Shrublands",
            "Tundra",
            "Mediterranean Forests, Woodlands & Scrub",
            "Deserts & Xeric Shrublands",
            "Mangroves",
            "Ocean", "Lake", "Rock and Ice"]
        self.short_labels =    ["T&ST MBF",
            "T&ST DBF",
            "T&ST CF",
            "Temp B&MF",
            "Temp CF",
            "BF/Taiga",
            "T&ST GrSaShrub",
            "Temp GrSaSh",
            "Fl GrSa",
            "Mont GrSh",
            "Tundra",
            "Med FWS",
            "Des&XS",
            "Mangroves",
            "Ocean", "Lake", "RIce"]
        self.vals = [1, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13 ,14 ,55 ,98, 99]
        self.vals_to_labels = {self.vals[i]:self.labels[i] for i in range(len(self.vals))}
        
    def run_(self, pos_embedding, location_encoder):
        ds = DS(self.file_path, self.mode)
        gn = torch.Generator().manual_seed(42)
        self.train, self.val, self.test = torch.utils.data.random_split(ds, [0.5, 0.1, 0.4], generator=gn)
        
        EPOCHS = self.epochs
        if self.verbose:
            EPOCHS = tqdm(range(EPOCHS))
        else:
            EPOCHS = range(EPOCHS)
            
        patience = 5
        
        if self.mode == "biomes":
            out_dim = 100
        else:
            out_dim = 81334
        model = Probe(self.mlp_input_len, pos_embedding, location_encoder, self.use_months, out_dim, self.pass_month_to_forward, linear_probing=self.linear_probing).to("cuda")
        
        loss_fn = torch.nn.CrossEntropyLoss()
        LR = 0.001
        if self.train_loc_enc:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        else:
            optimizer = torch.optim.Adam(model.mlp.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        BATCH_SIZE = 4096
        
        best_val_loss = 100
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
                    """# TODO: Throwing out the four month append
                    month = torch.full([len(loc)], MONTH).to("cuda")
                    if self.pass_month_to_forward:
                        train_x.append(model.location_encoder(loc, month).float().detach().cpu())
                    else:
                        loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                        train_x.append(model.location_encoder(loc_month).float().detach().cpu())"""
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
                    """# TODO: Throwing out the four month append
                    month = torch.full([len(loc)], MONTH).to("cuda")
                    if self.pass_month_to_forward:
                        val_x.append(model.location_encoder(loc, month).float().detach().cpu())
                    else:
                        loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                        val_x.append(model.location_encoder(loc_month).float().detach().cpu())"""
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
                    out = model.mlp(x).float()
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()

                model.mlp.eval()
                acc_loss = 0
                for idx in range(len(val_x)//BATCH_SIZE + 1):
                    x, y = val_x[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda"), val_y[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to("cuda")
                    with torch.no_grad():
                        out = model.mlp(x).float()
                    acc_loss += loss_fn(out, y).detach().cpu()
                acc_loss /= len(val)
                model.mlp.train()

            else:
                train = torch.utils.data.DataLoader(dataset=self.train, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

                for lonlat, y in train:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda")
                    optimizer.zero_grad()
                    out = model(lonlat).float()
                    loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()
                
                val = torch.utils.data.DataLoader(dataset=self.val, batch_size=BATCH_SIZE, num_workers=1, shuffle=False)
                model.eval()
                acc_loss = 0
                for lonlat, y in val:
                    lonlat = lonlat.to("cuda")
                    y = y.to("cuda")
                    with torch.no_grad():
                        out = model(lonlat).float()
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
        top1, top5, top10 = 0, 0, 0
        test = torch.utils.data.DataLoader(dataset=self.test, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)
        if self.mode == "biomes":
            pcm = PerClassMetrics(100)
        else:
            pcm = PerClassMetrics(81334)
        for lonlat, Id in test:
            lonlat = lonlat.to("cuda")
            Id = Id.to("cuda")

            with torch.no_grad():
                out = model(lonlat).float()
            tops = top_k_accuracy(out, Id, topk=(1, 5, 10))      
            top1 += tops[0]
            top5 += tops[1]
            top10 += tops[2]
            pcm.update(out, Id)
        top1 /= len(test)
        top5 /= len(test)
        top10 /= len(test)

        return top1, top5, top10, pcm, model

    def __call__(self, pos_embedding, location_encoder, wb, section="test/"):
        """
            ???
            :param pos_embedding: embeds [lon, lat], e.g. SH
            :param location_encoder: network that creates encoding for each location
            :param month: month to be embedded along with the location embedding

            :returns: ...
        """
        torch.manual_seed(42)

        top1s, top5s, top10s = [], [], []
        pcms = []

        for i in range(self.iterations):
            top1, top5, top10, pcm, model = self.run_(pos_embedding, location_encoder)
            top1s.append(top1.cpu())
            top5s.append(top5.cpu())
            top10s.append(top10.cpu())
            pcms.append(pcm)
        
        top1s = np.array(top1s)

        if self.mode == "biomes":
            f1s = []
            for pcm in pcms:
                f1 = pcm.calculate()[self.vals].cpu()
                f1s.append(torch.nan_to_num(f1, nan=0))
            f1s = torch.stack(f1s)       

            biomes_means = f1s.mean(dim=0)
            biomes_stds = f1s.std(dim=0)
            log_dict = {"BIOMES C/top1_mean" : top1s.mean(), "BIOMES C/top1_std" : top1s.std()}
            for i in range(len(self.labels)):
                biome = self.labels[i]
                log_dict["BIOMES C/" + biome + "_f1_mean"] = biomes_means[i].item() * 100
                log_dict["BIOMES C/" + biome + "_f1_std"] = biomes_stds[i].item() * 100
            
            macro_f1s = f1s.mean(dim=1)
            log_dict["BIOMES C/macro_f1_mean"] = macro_f1s.mean().item() * 100
            log_dict["BIOMES C/macro_f1_std"] = macro_f1s.std().item() * 100

            value_counts = torch.zeros((17))
            data = pd.read_csv(self.file_path, low_memory=False).fillna(55)
            vc = data["BIOME"].value_counts()
            for i in range(len(self.vals)):
                val = self.vals[i]
                value_counts[i] = vc.get(val)
            f1s = f1s * (value_counts/value_counts.sum())
            f1s = f1s.sum(dim=1)
            log_dict["BIOMES C/weighted_f1_mean"] = f1s.mean().item() * 100
            log_dict["BIOMES C/weighted_f1_std"] = f1s.std().item() * 100

            if wb:
                wb.log(log_dict)
            else:
                print(log_dict)
        else:
            wb.log({"ECOREGIONS C/top1_mean" : top1s.mean(), "ECOREGIONS C/top1_std" : top1s.std()})
        
        font = {'weight' : 'bold',
                'size'   : 16}
        import matplotlib
        matplotlib.rc('font', **font)

        if self.mode == "biomes" and self.plot_confusion_matrix:
            cf = ignite.metrics.confusion_matrix.ConfusionMatrix(100)
            ds = DS(self.file_path, self.mode)
            dl = torch.utils.data.DataLoader(dataset=ds, batch_size=8096, num_workers=0, shuffle=True)
            if self.verbose:
                print("Calculating confusion matrix")
            pred_counts = torch.zeros((17))
            for lonlat, y in dl:
                lonlat = lonlat.to("cuda")
                loc = model.pos_embedding(lonlat.double()).squeeze(dim=1)
                if self.use_months:
                    x = []
                    for m in [3, 6, 9, 12]:
                        month = torch.full([len(loc)], m).to("cuda")
                        if self.pass_month_to_forward:
                            x.append(model.location_encoder(loc, month).float())
                        else:
                            loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                            x.append(model.location_encoder(loc_month).float())
                    x = torch.cat(x, dim=-1)
                else:
                    x = model.location_encoder(loc).float()
                with torch.no_grad():
                    pred = model.mlp(x).float().detach().cpu()
                
                out_max = torch.zeros_like(pred)
                out_max[np.arange(len(y)), torch.argmax(pred, dim=1)] = 1
                pred_counts += out_max.sum(dim=0)[self.vals]
                cf.update((pred, y))

            cf = cf.compute()[self.vals][:, self.vals]
            cf_tn = cf / value_counts
            plt.close()
            import seaborn as sn
            df_cm = pd.DataFrame(cf_tn, self.short_labels, self.short_labels)
            ax = sn.heatmap(df_cm, annot_kws={"size": 10})
            ax.set(xlabel="Pred", ylabel="True")
            fig = ax.get_figure()
            fig.savefig("./temp_bcm_tn.png", bbox_inches="tight")
            if wb:
                img = wandb.Image(PIL.Image.open("./temp_bcm_tn.png"))
                wb.log({"BIOMES C/biomes_confusion_matrix_true_norm": img})
            
            #cf_pn = (cf.T / torch.flip(pred_counts, [0])).T
            cf_pn = cf / pred_counts.unsqueeze(0).expand_as(cf)
            cf_pn = torch.nan_to_num(cf_pn, nan=0, posinf=0)
            plt.close()
            import seaborn as sn
            df_cm = pd.DataFrame(cf_pn, self.short_labels, self.short_labels)
            ax = sn.heatmap(df_cm, annot_kws={"size": 10})
            ax.set(xlabel="Pred", ylabel="True")
            fig = ax.get_figure()
            fig.savefig("./temp_bcm_pn.png", bbox_inches="tight")
            if wb:
                img = wandb.Image(PIL.Image.open("./temp_bcm_pn.png"))
                wb.log({"BIOMES C/biomes_confusion_matrix_pred_norm": img})


        if self.mode == "biomes" and self.track_failure_areas:
            split_lat = 34
            split_lon = 80
            data = DS(self.file_path, self.mode).data.iloc[self.test.indices]
            #data = self.test.dataset.data
            diversity = np.zeros((split_lat, split_lon))
            sample_density = np.zeros((split_lat, split_lon))
            accuracy = np.ones((split_lat, split_lon)) * -10
            macro_f1 = np.ones((split_lat, split_lon)) * -10
            print("Calculating global failure areas")
            for i in range(split_lat):
                N = -63 + (i + 1) * (153/split_lat)
                S = -63 + i * (153/split_lat)
                for j in range(split_lon):
                    W = -180 + (j + 1) * (360/split_lon)
                    E = -180 + j * (360/split_lon)
                    samples = data[data["LON"] > E]
                    samples = samples[samples["LON"] < W]
                    samples = samples[samples["LAT"] > S]
                    samples = samples[samples["LAT"] < N]
                    if len(samples) > 0:
                        #print(S, N, E, W, len(samples))
                        #print("    ", [self.vals_to_labels[i] for i in samples["BIOME"].unique()])
                        diversity[i, j] = samples["BIOME"].nunique()
                        sample_density[i, j] = len(samples)

                        top1 = 0
                        test = torch.utils.data.DataLoader(dataset=Sample_DS(samples), batch_size=2048, num_workers=0, shuffle=False)
                        pcm = PerClassMetrics(100)
                        for lonlat, Id in test:
                            lonlat = lonlat.to("cuda")
                            Id = Id.to("cuda")

                            with torch.no_grad():
                                out = model(lonlat).float()
                            tops = top_k_accuracy(out, Id, topk=(1, 5))      
                            top1 += tops[0]
                            pcm.update(out, Id)

                        accuracy[i,j] = top1 / len(test)
                        macro_f1s = np.nan_to_num(pcm.calculate().cpu().detach(), nan=0)
                        macro_f1s = macro_f1s[np.where(macro_f1s>0)]
                        if len(macro_f1s) == 0:
                            macro_f1[i,j] = 0
                        else:
                            macro_f1[i,j] =  macro_f1s.mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([-180,180])
            ax.set_ylim([-63,90])
            im = ax.imshow(np.flip(diversity, axis=0), extent=(-180, 180, -63, 90))
            fig.colorbar(im, ax=ax)
            if wb:
                fig.savefig("./temp_gbd.png")
                img = wandb.Image(PIL.Image.open("./temp_gbd.png"))
                wb.log({"BIOMES C/global_biome_diversity": img})
            else:
                fig.savefig("./temp_gbd.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([-180,180])
            ax.set_ylim([-63,90])
            im = ax.imshow(np.flip(sample_density, axis=0), extent=(-180, 180, -63, 90))
            fig.colorbar(im, ax=ax)
            if wb:
                fig.savefig("./temp_bsd.png")
                img = wandb.Image(PIL.Image.open("./temp_bsd.png"))
                wb.log({"BIOMES C/sample_density": img})
            else:
                fig.savefig("./temp_bsd.png")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([-180,180])
            ax.set_ylim([-63,90])
            im = ax.imshow(np.flip(accuracy, axis=0), extent=(-180, 180, -63, 90), vmin=-10, vmax=100)
            fig.colorbar(im, ax=ax)
            if wb:
                fig.savefig("./temp_gba.png")
                img = wandb.Image(PIL.Image.open("./temp_gba.png"))
                wb.log({"BIOMES C/accuracy": img})
            else:
                fig.savefig("./temp_gba.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([-180,180])
            ax.set_ylim([-63,90])
            im = ax.imshow(np.flip(macro_f1, axis=0), extent=(-180, 180, -63, 90), vmin=-10, vmax=100)
            fig.colorbar(im, ax=ax)
            if wb:
                fig.savefig("./temp_gbr.png")
                img = wandb.Image(PIL.Image.open("./temp_gbr.png"))
                wb.log({"BIOMES C/macro_f1": img})
            else:
                fig.savefig("./temp_gbr.png")
     
        if self.lake_victoria_map and self.mode == "biomes":
            def map_res(scope):
                if scope == "sw":
                    ds = SwitzerlandDataset()
                elif scope == "eu":
                    ds = EuropeDataset()
                elif scope == "world":
                    ds = WorldDataset()
                elif scope == "lv":
                    ds = LakeVictoriaDataset()
                dl = torch.utils.data.DataLoader(
                        dataset=ds,
                        batch_size=8196,
                        num_workers=0,
                        shuffle=False,
                    )

                encodings = []
                x = []
                y = []
                if self.verbose:
                    print("Creating", scope, "map")
                for lonlat in (tqdm(dl) if self.verbose else dl):
                    x.append(lonlat[:,0])
                    y.append(lonlat[:,1])
                    lonlat = lonlat.to("cuda")

                    loc = model.pos_embedding(lonlat.double()).squeeze(dim=1).to("cuda")
                    with torch.no_grad():
                        if self.use_months:
                            temp = []
                            for m in [3,  6,  9, 12]:
                                month = torch.full([len(loc)], m).to("cuda")
                                if self.pass_month_to_forward:
                                    temp.append(model.location_encoder(loc, month).float())
                                else:
                                    loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                                    temp.append(model.location_encoder(loc_month).float())
                            temp = torch.cat(temp, dim=-1).to("cuda")
                            """# TODO: Throwing out four months
                            month = torch.full([len(loc)], MONTH).to("cuda")
                            if self.pass_month_to_forward:
                                temp = model.location_encoder(loc, month).float()
                            else:
                                loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                                temp = model.location_encoder(loc_month).float()"""
                            encodings.append(model.mlp(temp).detach().cpu())
                        else:
                            encodings.append(model.mlp(model.location_encoder(loc).float()).detach().cpu())
                encodings = torch.cat(encodings, dim=0)
                x = torch.cat(x, dim=0).detach().cpu().numpy()
                y = torch.cat(y, dim=0).detach().cpu().numpy()
                
                """if not hasattr(ds, 'land_mask'):
                    imgs = encodings.reshape(ds.y_pixel, ds.x_pixel, -1)
                else:
                    imgs = np.zeros((ds.y_pixel, ds.x_pixel, encodings.shape[-1]))
                    imgs[ds.land_mask] = encodings"""

                if scope == "lv":
                    m = nn.LogSoftmax(dim=1)
                    encodings = m(encodings)
                    #encodings = (encodings - encodings.min()) / (encodings.max() - encodings.min())
                    for biome in [7, 11, 12, 13]:
                        fig, ax = plt.subplots(figsize=(12, 11))
                        ax.set_xlim([ds.W, ds.E])
                        ax.set_ylim([ds.S, ds.N])
                        imgs = np.zeros((ds.y_pixel, ds.x_pixel, 1))
                        imgs[dl.dataset.land_mask != 0] = encodings[:, biome].reshape(-1, 1)
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        #im = ax.imshow(imgs, extent=(ds.W, ds.E, ds.S, ds.N), vmin=0, vmax=1)
                        im = ax.imshow(imgs, extent=(ds.W, ds.E, ds.S, ds.N))
                        fig.colorbar(im, ax=ax)
                        if wb:
                            fig.savefig("./temp_lvb_" + str(biome) + ".png")
                            img = wandb.Image(PIL.Image.open("./temp_lvb_" + str(biome) + ".png"))
                            wb.log({"BIOMES C/lake_victoria_" + self.vals_to_labels[biome]: img})
                            img = wandb.Image(PIL.Image.open("./temp_lvb_" + str(biome) + "_gt.png"))
                            wb.log({"BIOMES C/lake_victoria_" + self.vals_to_labels[biome] + "_gt": img})
                        else:
                            fig.savefig("./temp_lvb_" + str(biome) + ".png")
                        plt.close()
                
                img = np.argmax(encodings, axis=1)

                if scope == "lv":
                    fig, ax = plt.subplots(figsize=(12, 12))
                elif scope == "sw":
                    fig, ax = plt.subplots(figsize=(12, 6))
                elif scope == "eu":
                    fig, ax = plt.subplots(figsize=(9, 12))
                elif scope == "world":
                    fig, ax = plt.subplots(figsize=(12, 7))
                ax.set_xlim([ds.W, ds.E])
                ax.set_ylim([ds.S, ds.N])

                #im = ax.imshow(img, extent=(ds.E, ds.W, ds.S, ds.S))
                #[ 1  7  9 10 13 14 55 98]
                self.vals =      [ 1, 2, 3, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13 ,14 ,55 ,98, 99]
                colors =  ["xkcd:forest green", "darkgoldenrod", "red", "orange", "darkcyan", "indigo", "xkcd:kelly green", "slategrey", "xkcd:blue green", "xkcd:carmine", "greenyellow", "lightcoral", "xkcd:sand", "m", "c", "b", "k"]
                #["xkcd:forest green", "g", "g", "g", "g", "g", "xkcd:kelly green", "r", "xkcd:blue green", "xkcd:carmine", "w", "k", "xkcd:sand", "m", "c", "b"]
                for i, c, l in zip(self.vals, colors, self.labels):
                    if len(x[img==i]) > 0:
                        plt.plot(x[img==i], y[img==i], 
                        marker="o", markersize=2, ls="None", color=c, 
                        label=l)

                if scope != "sw":
                    ocean = dl.dataset.get_ocean_coords()
                    plt.plot(ocean[:,0], ocean[:,1], 
                            marker="o", markersize=2, ls="None", color="c")

                if scope != "world":#im = ax.scatter(x, y, c=img)
                    lons = []
                    lats = []
                    for lonlat, y in self.train:
                        lon, lat = lonlat[0], lonlat[1]
                        if ds.W <= lon <= ds.E and ds.S <= lat <= ds.N:
                            lons.append(lon)
                            lats.append(lat)
                    plt.plot(lons, lats, marker="+", ls="None", c="k", alpha=1, markersize=5)
                    lons = []
                    lats = []
                    for lonlat, y in self.test:
                        lon, lat = lonlat[0], lonlat[1]
                        if ds.W <= lon <= ds.E and ds.S <= lat <= ds.N:
                            lons.append(lon)
                            lats.append(lat)
                    plt.plot(lons, lats, marker="+", ls="None", c="red", alpha=1, markersize=5)
                """sample_ds = DS(self.file_path, self.mode)
                lv_samples = sample_ds.data[sample_ds.data["LON"]>=ds.W]
                lv_samples = lv_samples[lv_samples["LON"]<=ds.E]
                lv_samples = lv_samples[lv_samples["LAT"]>=ds.S]
                lv_samples = lv_samples[lv_samples["LAT"]<=ds.N]
                plt.plot(lv_samples["LON"], lv_samples["LAT"], marker="+", ls="None", c="k", alpha=1, markersize=5)"""

                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")

                import matplotlib.patches as mpatches
                handles = []
                for i in range(len(self.vals)):
                    if len(x[img==self.vals[i]]) > 0:
                        handles.append(mpatches.Patch(color=colors[i], label=self.labels[i]))
                handles.append(mlines.Line2D([], [], color='k', marker="+", markersize=10, label='Training samples'))
                handles.append(mlines.Line2D([], [], color='red', marker="+", markersize=10, label='Test samples'))
                fig.legend(handles = handles, framealpha=1)

                if scope == "sw": 
                    if wb:
                        fig.savefig("./temp_swb.png")
                        img = wandb.Image(PIL.Image.open("./temp_swb.png"))
                        wb.log({"BIOMES C/switzerland_biomes": img})
                        img = wandb.Image(PIL.Image.open("./temp_swb_gt.png"))
                        wb.log({"BIOMES C/switzerland_biomes_gt": img})
                    else:
                        fig.savefig("./temp_swb.png")
                elif scope == "eu": 
                    if wb:
                        fig.savefig("./temp_eub.png")
                        img = wandb.Image(PIL.Image.open("./temp_eub.png"))
                        wb.log({"BIOMES C/europe_biomes": img})
                        img = wandb.Image(PIL.Image.open("./temp_eub_gt.png"))
                        wb.log({"BIOMES C/europe__biomes_gt": img})
                    else:
                        fig.savefig("./temp_eub.png")
                elif scope == "lv": 
                    if wb:
                        fig.savefig("./temp_lvb.png")
                        img = wandb.Image(PIL.Image.open("./temp_lvb.png"))
                        wb.log({"BIOMES C/lake_victoria_biomes": img})
                        img = wandb.Image(PIL.Image.open("./temp_lvb_gt.png"))
                        wb.log({"BIOMES C/lake_victoria_biomes_gt": img})
                    else:
                        fig.savefig("./temp_lvb.png")
                elif scope == "world": 
                    if wb:
                        fig.savefig("./temp_worldb.png")
                        img = wandb.Image(PIL.Image.open("./temp_worldb.png"))
                        wb.log({"BIOMES C/world_biomes": img})
                        img = wandb.Image(PIL.Image.open("./temp_worldb_gt.png"))
                        wb.log({"BIOMES C/world_biomes_gt": img})
                    else:
                        fig.savefig("./temp_worldb.png")

                plt.close()

            map_res("world")
            map_res("lv")
            map_res("sw")
            map_res("eu")

        if self.biome_tsne and self.mode == "biomes":
            ds = DS(self.file_path, self.mode)
            SAMPLES = 2048

            sample = ds.data.sample(SAMPLES)

            sample = sample[sample["BIOME"]<50]
            lonlat = torch.stack([torch.tensor(sample["LON"].to_numpy()), torch.tensor(sample["LAT"].to_numpy())]).transpose(1,0).to("cuda")

            loc = model.pos_embedding(lonlat.double()).squeeze(dim=1).to("cuda")
            if self.use_months:
                temp = []
                for m in [3,  6,  9, 12]:
                    month = torch.full([len(loc)], m).to("cuda")
                    if self.pass_month_to_forward:
                        temp.append(model.location_encoder(loc, month).float())
                    else:
                        loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                        temp.append(model.location_encoder(loc_month).float())
                temp = torch.cat(temp, dim=-1).to("cuda")
                encodings = model.mlp(temp).detach().cpu()
                """# TODO: Throwing out four months
                month = torch.full([len(loc)], MONTH).to("cuda")
                if self.pass_month_to_forward:
                    encodings = model.location_encoder(loc, month).float().detach().cpu()
                else:
                    loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
                    encodings = model.location_encoder(loc_month).float().detach().cpu()"""
            else:
                encodings = model.location_encoder(loc).float().detach().cpu()

            if self.verbose:
                print("Creating tsne")
            from bhtsne import tsne
            from sklearn.manifold import TSNE
            #red = TSNE(n_components=2, method="exact").fit_transform(encodings)
            red = tsne(np.float64(encodings))

            fig, ax = plt.subplots(figsize=(12, 11))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            colors =    ["xkcd:forest green", "darkgoldenrod", "red", "orange", "darkcyan", "indigo", "xkcd:kelly green", "slategrey", "xkcd:blue green", "xkcd:carmine", "greenyellow", "lightcoral", "xkcd:sand", "m", "k", "k", "k"]
            val_to_color = {self.vals[i]:colors[i] for i in range(len(self.vals))}

            ax.scatter(red[:,0], red[:,1], marker="o", c=[val_to_color[int(i)] for i in sample["BIOME"]], alpha=1, s=40)

            ax.set_xlabel("T-SNE 1")
            ax.set_ylabel("T-SNE 2")

            import matplotlib.patches as mpatches
            handles = []
            for i in range(len(self.vals)):
                if len(sample[sample["BIOME"]==self.vals[i]]) > 0:
                    handles.append(mpatches.Patch(color=colors[i], label=self.labels[i]))
            fig.legend(handles = handles, framealpha=1)
                
            if wb:
                fig.savefig("./temp_btsne.png")
                img = wandb.Image(PIL.Image.open("./temp_btsne.png"))
                wb.log({"BIOMES C/biome_tsne": img})
            else:
                fig.savefig("./temp_btsne.png")

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
import geopandas as gpd
import pandas as pd
class EcoregionsLocEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eco = gpd.read_file("/home/jdolli/chelsaCLIP/src/utils/test_cases/data/data/commondata/data0/wwf_terr_ecos.shp")
    def forward(self, x):

        pts = pd.DataFrame(x.cpu().numpy())

        # Rename the columns of the CSV for clarity
        pts.columns = ['LON', 'LAT']
        # Convert the points to a GeoDataFrame
        pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['LON'], pts['LAT'], crs="EPSG:4326"))
        # Ensure both GeoDataFrames have the same CRS
        pts_gdf.to_crs(self.eco.crs, inplace=True)
        # Perform spatial join to extract information
        ep_all = gpd.sjoin(pts_gdf, self.eco, how='left')
        # Select the desired columns
        ep = ep_all[['BIOME']]

        x = torch.zeros((len(ep),100))
        ep_labels = ep.to_numpy().reshape(-1)
        ep_labels[ep_labels<0] = 55
        try:
            x[torch.arange(len(x)), ep_labels] = 1
        except:
            ep_labels = np.nan_to_num(ep_labels, nan=55)
            x[torch.arange(len(x)), ep_labels] = 1

        return x.to("cuda")

if __name__ == "__main__":
    pos_embedding = FakePosEmb()
    location_encoder = EcoregionsLocEnc()
    #location_encoder = FakeLocEnc()
    
    sw_sdm = BEC('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/ecobiomes_100000.csv',
    mode="biomes",
    mlp_input_len=100, use_months=False, verbose=True, plot_confusion_matrix=False, track_failure_areas=False, lake_victoria_map=True, biome_tsne=False, iterations=1, epochs=5000, linear_probing=True)
    sw_sdm(pos_embedding, location_encoder, None)
    