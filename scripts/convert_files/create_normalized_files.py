import rioxarray
import os
import json
import traceback
from tqdm import tqdm

normalization_file = (
    "/shares/wegner.ics.uzh/CHELSA"
    + "/climatologies/1981-2010/"
    + "normalization_values.json"
)
old_climatology_name = "1981-2010/"
new_climatology_name = "1981-2010_normalized/"

try:
    dir_name = "/".join(normalization_file.split("/")[:-1]) + "/"
    os.mkdir(dir_name.replace(old_climatology_name, new_climatology_name))
except:
    pass

with open(normalization_file, "r") as f:
    norm = json.load(f)

try:
    for file_name in tqdm(norm.keys()):
        if not os.path.isfile(
            file_name.replace(old_climatology_name, new_climatology_name)
        ):
            file = rioxarray.open_rasterio(file_name)
            mean, std = norm[file_name]
            try:
                dir_name = "/".join(file_name.split("/")[:-1])
                os.mkdir(dir_name.replace(old_climatology_name, new_climatology_name))
            except:
                pass
            file = (file - mean) / std
            file.rio.to_raster(
                file_name.replace(old_climatology_name, new_climatology_name)
            )
            file.close()
            del file

except:
    traceback.print_exc()
