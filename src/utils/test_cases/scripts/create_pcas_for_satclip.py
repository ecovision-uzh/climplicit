import numpy as np

# import sys
# sys.path.append('/home/jdolli/chelsaCLIP/src/utils/test_cases')
try:
    from src.utils.test_cases.util_datasets import *
except:
    from util_datasets import *


def create_intermediate_npy_satCLIP(scope):

    import torch
    from tqdm import tqdm

    sys.path.append("/home/jdolli/satclip/satclip")
    from load import get_satclip

    model = get_satclip(
        "/home/jdolli/chelsaCLIP/src/utils/test_cases/scripts/satclip-resnet18-l40.ckpt",
        device="cuda",
    )  # Only loads location encoder by default
    model.eval()

    if scope == "swi":
        ds = SwitzerlandDataset()
    elif scope == "swi_tc":
        ds = SwitzerlandDatasetTC()
    elif scope == "zur":
        ds = ZurichDataset()
    elif scope == "euro":
        ds = EuropeDataset()
    elif scope == "world":
        ds = WorldDataset()

    dl = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=1024,
        num_workers=16,
        shuffle=False,
    )

    encodings = []
    for lonlat in tqdm(dl):
        with torch.no_grad():
            encodings.append(model(lonlat.to("cuda").double()).detach().cpu())
    all_encs = torch.concat(encodings, dim=0).cpu().numpy()

    np.save("./utils/test_cases/data/satCLIP_intermediate_" + scope + ".npy", all_encs)


def create_map_visual_satCLIP(reduction, scope):

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    all_encs = np.load("./utils/test_cases/data/satCLIP_intermediate_" + scope + ".npy")

    if reduction == "first_three":
        red = all_encs[:, :3]
    elif reduction == "first":
        red = all_encs[:, 0]
    elif reduction == "second":
        red = all_encs[:, 1]
    elif reduction == "pca":
        pca = PCA(n_components=3).fit(all_encs)
        red = pca.transform(all_encs)
    elif reduction == "tsne":
        red = TSNE(n_components=3).fit_transform(all_encs)

    print("Done with reduction:", reduction, "on", scope)

    if scope == "swi":
        ds = SwitzerlandDataset()
    elif scope == "swi_tc":
        ds = SwitzerlandDatasetTC()
    elif scope == "zur":
        ds = ZurichDataset()
    elif scope == "euro":
        ds = EuropeDataset()
    elif scope == "world":
        ds = WorldDataset()

    try:
        red = red.reshape(ds.y_pixel, ds.x_pixel, 3)
    except:
        red = red.reshape(ds.y_pixel, ds.x_pixel, 1)
    red = (red - red.min()) / (red.max() - red.min())
    plt.imsave(
        "./utils/test_cases/data/satCLIP_" + reduction + "_" + scope + ".png", red
    )


if __name__ == "__main__":
    # create_intermediate_npy_satCLIP("zur")
    # create_intermediate_npy_satCLIP("swi_tc")
    # create_intermediate_npy_satCLIP("euro")
    # create_intermediate_npy_satCLIP("world")

    create_map_visual_satCLIP("pca", "zur")
    create_map_visual_satCLIP("pca", "swi_tc")
    create_map_visual_satCLIP("pca", "euro")
    create_map_visual_satCLIP("pca", "world")
