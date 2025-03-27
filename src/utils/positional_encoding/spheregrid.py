import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math


class SphereGridSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    Taken from https://github.com/gengchenmai/sphere2vec/blob/main/main/SpatialRelationEncoder.py
    """

    def __init__(
        self,
        spa_embed_dim=None,
        coord_dim=2,
        frequency_num=16,
        max_radius=10000,
        min_radius=10,
        freq_init="geometric",
        ffn=None,
        device="cuda",
    ):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(SphereGridSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.input_embed_dim = self.cal_input_dim()

        self.ffn = ffn
        self.device = device

    def cal_elementwise_angle(self, coord, cur_freq):
        """
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        """
        return coord / (
            np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1))
        )

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        if self.freq_init == "random":
            # the frequence we use for each block, alpha in ICLR paper
            # self.freq_list shape: (frequency_num)
            self.freq_list = (
                np.random.random(size=[self.frequency_num]) * self.max_radius
            )
        elif self.freq_init == "geometric":
            """self.freq_list = []
            for cur_freq in range(self.frequency_num):
                #base = 1.0/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
                base =(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
                self.freq_list.append(base)

            self.freq_list = np.asarray(self.freq_list)"""
            log_timescale_increment = math.log(
                float(self.max_radius) / float(self.min_radius)
            ) / (self.frequency_num * 1.0 - 1)

            timescales = self.min_radius * np.exp(
                np.arange(self.frequency_num).astype(float) * log_timescale_increment
            )

            self.freq_list = 1.0 / timescales
        # self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception(
                "Unknown coords data type for GridCellSpatialRelationEncoder"
            )

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # convert to radius
        spr_embeds = spr_embeds * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 6)
        spr_embeds_ = np.concatenate(
            [lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_cos, lat_cos * lon_sin],
            axis=-1,
        )

        # (batch_size, num_context_pt, 2*frequency_num*6)
        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        coords = coords.unsqueeze(dim=1).cpu().numpy()

        spr_embeds = self.make_input_embeds(coords)

        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        # return sprenc
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


if __name__ == "__main__":
    from tqdm import tqdm

    BS, EXPS, SPA_EMB, FREQ_NUM = (32000, 10, 40, 64)
    print(
        "Testing for BS",
        BS,
        "- EXPS",
        EXPS,
        "- FREQ_NUM",
        FREQ_NUM,
        "- SPA_EMB",
        SPA_EMB,
    )
    sg = SphereGridSpatialRelationEncoder(
        SPA_EMB,
        coord_dim=2,
        frequency_num=FREQ_NUM,
        max_radius=100000,
        min_radius=10,
        freq_init="geometric",
        ffn=None,
        device="cuda",
    )

    # longitude pi and latitude pi/2
    lonlat = torch.rand([BS, 2]) * 180 - 90  # Random values in [-90,90]
    # lonlat = torch.rand([BS,2]) * math.pi - math.pi/2  # Random values in [-pi/2,pi/2]
    lonlat = lonlat.unsqueeze(dim=1).numpy()
    for i in tqdm(range(EXPS)):
        out = sg(lonlat).squeeze(dim=1)
    print(out[0], len(out[0]), out.shape)
