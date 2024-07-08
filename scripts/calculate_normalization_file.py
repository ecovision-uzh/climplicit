import rioxarray
import os
import json
from tqdm import tqdm

CHELSA_DIR = "/shares/wegner.ics.uzh/CHELSA"
monthly_raster_names = os.listdir(CHELSA_DIR + "/climatologies/1981-2010/")
monthly_raster_names.remove("bio")
bio_files = []
for file in os.listdir(CHELSA_DIR + "/climatologies/1981-2010/bio"):
    if "bio" in file:
        bio_files.append(file)
print(monthly_raster_names)
print(len(bio_files))

# Calculating the normalization values for all files
normalization_dict = {}
for bio in tqdm(bio_files):
    bio_file_name = CHELSA_DIR + "/climatologies/1981-2010/bio/" + bio
    bio_file = rioxarray.open_rasterio(bio_file_name)
    normalization_dict[bio_file_name] = (bio_file.mean(), bio_file.std())
    del bio_file
print(len(normalization_dict), bio_file_name, normalization_dict[bio_file_name])

for monthly in tqdm(monthly_raster_names):
    for mfn in os.listdir(CHELSA_DIR + "/climatologies/1981-2010/" + monthly):
        monthly_file_name = CHELSA_DIR + "/climatologies/1981-2010/" + monthly + "/" + mfn
        monthly_file = rioxarray.open_rasterio(monthly_file_name)
        normalization_dict[monthly_file_name] = (monthly_file.mean(), monthly_file.std())
        del monthly_file
print(len(normalization_dict), monthly_file_name, normalization_dict[monthly_file_name])

for key in normalization_dict.keys():
    mean, std = normalization_dict[key]
    normalization_dict[key] = (float(mean), float(std))
print(normalization_dict[key])

with open(CHELSA_DIR + "/climatologies/1981-2010/" + "normalization_values.json", "w") as f:
    json.dump(normalization_dict, f)