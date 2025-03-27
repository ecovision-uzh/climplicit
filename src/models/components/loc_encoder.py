from torch import nn, optim
import math
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

"""Sinusoidal Representation Network (SIREN)"""
"""Where did I copy this one from?"""
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, dropout = False,
    residual_connections=False, h_siren=False, return_hidden_embs=None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.return_hidden_embs = return_hidden_embs

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                h_siren = h_siren,
                dropout = dropout,
                residual_connections=residual_connections
            ))

        final_activation = nn.Identity() if not final_activation else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation, dropout = False)

    def forward(self, x, mods = None):

        res = []
        gaussian = None
        # Passing the gaussian (vector after dot product and before sin) between layers as residual connection
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, gaussian = layer(x, gaussian)
            if self.return_hidden_embs is not None:
                if i in self.return_hidden_embs:
                    res.append(x)

        x, _ = self.last_layer(x, gaussian)

        if len(res) > 0:
            res.append(x)
            x = torch.cat(res, dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = act
    def forward(self, x):
        return x + self.act(self.linear(x))

class FFN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, activation, residual_connections):
        super().__init__()
        if activation == "ReLU":
            act = nn.ReLU()
        elif activation == "GELU":
            act = nn.GELU()
        
        if residual_connections:
            layers = [nn.Linear(dim_in, dim_hidden), act]
            for i in range(num_layers-1):
                layers.append(ResBlock(dim_hidden, dim_hidden, act))
            layers.append(nn.Linear(dim_hidden, dim_out))
        else:
            layers = [nn.Linear(dim_in, dim_hidden), act]
            for i in range(num_layers-1):
                layers.append(nn.Linear(dim_hidden, dim_hidden))
                layers.append(act)
            layers.append(nn.Linear(dim_hidden, dim_out))

        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None, dropout = False,
    residual_connections = False, h_siren = False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout
        self.h_siren = h_siren

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

        self.residual_connections = residual_connections

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if not bias is None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, prev_gaussian = None):
        out =  F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        if self.residual_connections and prev_gaussian is not None and out.shape == prev_gaussian.shape:
            out = (out + prev_gaussian) / 2
        gaussian = out
        if self.h_siren and self.is_first:
            out = torch.sinh(2 * out)
        out = self.activation(out)
        return out, gaussian
    

if __name__ == "__main__":
    from tqdm import tqdm
    BS, HID_DIM, HID_NUM, DIM_IN, DIM_OUT, EXPS  = (32000, 256, 2, 384, 128, 100)
    print("Testing for BS", BS, "- HID_DIM", HID_DIM, "- HID_NUM", HID_NUM, "- DIM_IN", DIM_IN, "- DIM_OUT", DIM_OUT)
    sg = SirenNet(DIM_IN, HID_DIM, DIM_OUT, HID_NUM, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, dropout = True)
    sg = sg.to("cuda")
    print(sg)

    locs = torch.rand(BS, DIM_IN).to("cuda")
    for i in tqdm(range(EXPS)):
        out = sg(locs)
    print(out[0], out.shape)