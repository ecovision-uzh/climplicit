import geopandas as gpd
import pandas as pd
# Load the vector data using GeoPandas
eco = gpd.read_file("/home/jdolli/chelsaCLIP/src/utils/test_cases/data/data/commondata/data0/wwf_terr_ecos.shp")

import torch
import math
import rasterio

from global_land_mask import globe

# Create point sample
#pts = pd.read_csv(".../random_points.csv",  header=None)
import rioxarray
from tqdm import tqdm
import numpy as np
from global_land_mask import globe
#ras = rioxarray.open_rasterio('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)
#ras = rasterio.open('/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010/cmi/CHELSA_cmi_01_1981-2010_V.2.1.tif', cache=False)

SAMPLES = 100000
"""#lsm = np.array(ras)[0]<30000
lsm = np.array(ras)[0]<300000 # To also sample the ocean
land = np.where(lsm)
land = np.transpose(np.stack(land))

sample = np.random.choice(len(land), SAMPLES)
sample = land[sample]"""


sample = []

while len(sample) < SAMPLES:
    rand_feats_orig = torch.rand(2)
    theta1 = 2.0*math.pi*rand_feats_orig[0]
    theta2 = torch.acos(2.0*rand_feats_orig[1] - 1.0)
    lat = 1.0 - 2.0*theta2/math.pi
    lon = (theta1/math.pi) - 1.0
    lat = lat * 90
    lon = lon * 180
    # Also only take samples outside of Antartica
    if globe.is_land(lat, lon) and lat > -63:
        sample.append((lon.numpy(), lat.numpy()))
    if len(sample)%(SAMPLES//10) == 0:
        print(len(sample))

data = np.array(sample)
print(data.shape)

"""data = np.zeros((len(sample), 2), dtype="float")
for i in tqdm(range(len(sample))):
    #y, x = sample[i]
    lon, lat = sample[i]
    #y, x = ras.index(lon, lat)
    #loc = ras[0, y, x]
    data[i][0] = lon #loc["x"] # Saving as lon/lat
    data[i][1] = lat #loc["y"] """

pts = pd.DataFrame(data)

# Rename the columns of the CSV for clarity
pts.columns = ['LON', 'LAT']
# Convert the points to a GeoDataFrame
pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts['LON'], pts['LAT'], crs="EPSG:4326"))
# Ensure both GeoDataFrames have the same CRS
pts_gdf.to_crs(eco.crs, inplace=True)
# Perform spatial join to extract information
ep_all = gpd.sjoin(pts_gdf, eco, how='left')
# Select the desired columns
ep = ep_all[['LON', 'LAT', 'ECO_NAME', 'BIOME', 'ECO_NUM', 'ECO_ID']]
# Display the resulting DataFrame
print(ep)
print(ep['ECO_NAME'].nunique())
print(ep['BIOME'].nunique())
print(ep['ECO_NUM'].nunique())
print(ep['ECO_ID'].nunique())

ep.to_csv('/home/jdolli/chelsaCLIP/src/utils/test_cases/data/ecobiomes_'+str(SAMPLES)+'.csv')