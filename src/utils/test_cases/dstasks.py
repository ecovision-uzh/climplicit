import torch
from torcheval.metrics import R2Score

import pandas as pd

import wandb
import PIL

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/jdolli/chelsaCLIP/src/utils/test_cases/downstreams')
from air_temp_regression import ATR
from chelsa_regression import CHR
from cali_housing_regression import CalR
from median_income_regression import MIR
from pop_density_regression import PDR
from ecobiome_classification import BEC
from switzerland_glc_sdm import SW_SDM
from planttraits_regression import PTR
from loc_month_regression import LMR
from europe_glc_sdm import EU_SDM
from global_chelsa_regression import GCR


class DSTs():
    def __init__(self, mlp_input_len, use_months, pass_month_to_forward=False,
    verbose=True, linear_probing=True, iterations=10, deactivate=False, train_loc_enc=False):
        self.mlp_input_len = mlp_input_len
        self.verbose = verbose
        self.linear_probing = linear_probing
        self.iterations = iterations
        self.use_months = use_months
        self.pass_month_to_forward = pass_month_to_forward
        self.deactivate = deactivate
        self.train_loc_enc = train_loc_enc

    def __call__(self, pos_embedding, location_encoder, wb, section=""):
        if self.deactivate:
            print("Downstream-Tasks deactivated")
            return

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Air-Temperature Regression:")
        atr = ATR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/stationDataAll.csv', mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        atr(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("CHELSA Regression:")
        chre = CHR(ptc_path="/shares/wegner.ics.uzh/CHELSA/Switzerland/input/point_to_coord.npy",
                chelsa_path="/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/03_monthly_float16.npy",
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing,
                map_pca=True, train_loc_enc=self.train_loc_enc)
        chre(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Location-Month Regression:")
        lmr =LMR(mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing,
                map_pca=True, train_loc_enc=self.train_loc_enc)
        lmr(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Global Chelsa Regression:")
        gcr = GCR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/global_chelsa_100000.npy',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        gcr(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Cali Housing Regression:")
        calr = CalR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/housing.csv',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        calr(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Pop density Regression:")
        pdr = PDR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/pop_density_us_100000.npy',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        pdr(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Median income Regression:")
        mir = MIR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/Unemployment.xlsx',
                '/home/jdolli/chelsaCLIP/src/utils/test_cases/data/us_county_latlng.csv',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        mir(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Biome Classification:")
        bc = BEC('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/ecobiomes_100000.csv',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, mode = "biomes", pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, lake_victoria_map=True, biome_tsne=True, train_loc_enc=self.train_loc_enc)
        bc(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Ecoregion Classification:")
        ec = BEC('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/ecobiomes_100000.csv',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, mode = "ecoregions", pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        #ec(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Species Distribution Modeling:")
        sdm = SW_SDM('/shares/wegner.ics.uzh/glc23_data/Switzerland_PO.csv',
            mlp_input_len=self.mlp_input_len,
            use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
            verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing,
            most_common_species_map=True, train_loc_enc=self.train_loc_enc)
        #sdm(pos_embedding, location_encoder, wb)
        if self.train_loc_enc:
            location_encoder.reset_model()
        sdm = EU_SDM('/shares/wegner.ics.uzh/glc23_data/Pot_10_to_1000.csv',
            mlp_input_len=self.mlp_input_len,
            use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
            verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing,
            most_common_species_map=True, train_loc_enc=self.train_loc_enc)
        sdm(pos_embedding, location_encoder, wb)

        if self.train_loc_enc:
            location_encoder.reset_model()
        if self.verbose:
            print("Planttraits Regression:")
        ptr = PTR('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/planttraits_100000.npy',
                mlp_input_len=self.mlp_input_len,
                use_months=self.use_months, pass_month_to_forward=self.pass_month_to_forward,
                verbose=self.verbose, iterations=self.iterations, linear_probing=self.linear_probing, train_loc_enc=self.train_loc_enc)
        ptr(pos_embedding, location_encoder, wb)
    
        
            
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
    
    dsts = DSTs(2, verbose=True, use_months=False, iterations=1)
    dsts(pos_embedding, location_encoder, None)
    