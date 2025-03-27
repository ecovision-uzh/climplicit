from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch


months = ["03", "06", "09", "12"]

all_encs = []

for month in months:
    encs = []
    raster = np.load(
        "/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/"
        + month
        + "_monthly_float16.npy"
    )
    for i in range(11):
        encs.append(raster[i].reshape(245 * 550, 1))
    all_encs.append(np.concatenate(encs, axis=1))

all_encs = np.concatenate(all_encs, axis=0)

red = PCA(n_components=3).fit_transform(all_encs)

for i in range(len(months)):
    img = red[i * 245 * 550 : (i + 1) * 245 * 550, :].reshape(245, 550, 3)
    img = (img - img.min()) / (img.max() - img.min())
    plt.imsave("./" + str(int(months[i])) + "_pca_chelsa.png", img)

# Also do it for Zurich
x, y = (165, 80)
len = 75
raster = (
    np.load(
        "/shares/wegner.ics.uzh/CHELSA/Switzerland/1981-2010_numpy/03_monthly_float16.npy"
    )[:, y : y + len, x : x + len]
    .transpose(1, 2, 0)
    .reshape(len * len, 11)
)
# encs = []
# for i in range(11):
#    encs.append(raster[i].reshape(245*550, 1))
red = PCA(n_components=3).fit_transform(raster)

img = red.reshape(len, len, 3)
img = (img - img.min()) / (img.max() - img.min())
plt.imsave("./03_pca_chelsa_zurich.png", img)
