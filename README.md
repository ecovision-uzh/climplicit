
# Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks
This repository contains the code for [**Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks**](*).
The work extends the [SIREN](https://arxiv.org/abs/2006.09661) backbone with residual connections (**ReSIREN**) and trains it on the dense and global climatic raster [CHELSA](https://chelsa-climate.org/). The code uses the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework with [Hydra](https://hydra.cc/docs/intro/).

Note1: This work started out as a contrastive framework inspired by [SatCLIP](https://github.com/microsoft/satclip) and Climplicit only emerged from the ablation of regressing the climatic data directly. Therefore the code structure is still contrastive, and everything is named ChelsaCLIP.

## [ArXiv](https://arxiv.org/abs/2504.05089) and [Project Page](https://ecovision-uzh.github.io/climplicit/)

# Quickstart
The *quickstart* folder contains the least amount of files required to showcase the usage of the pretrained Climplicit embeddings:
```bash
git clone https://github.com/ecovision-uzh/climplicit.git
cd climplicit/quickstart
conda env create -f environment.yml
conda activate climplicit-quickstart
python climplicit.py
```
A tutorial on how to use such pretrained embeddings for downstream learning can be found in the [SatCLIP](https://github.com/microsoft/satclip) codebase.

# Pretraining
To setup the directory run:
```bash
git clone https://github.com/ecovision-uzh/climplicit.git
cd climplicit
conda config --set channel_priority strict
conda env create -f environment.yml
conda activate climplicit
```

To download all monthly CHELSA rasters for the 1981-2010 climatology use:
```bash
mkdir data
cd data
wget --no-host-directories --force-directories --input-file=envidatS3paths_all_monthly.txt
cd ..
```

Now adapt the paths in ```scripts/convert_files/turn_chelsa_into_monthly_numpy.py``` and run the script:
```bash
python scripts/convert_files/turn_chelsa_into_monthly_numpy.py
```

Now change all paths in ```configs/experiments/climplicit.yml``` and ```configs/paths/default.yml``` to the correct root. Finally to start the pretraining run:
```bash
python src/train.py experiment=climplicit
```

# Citation
```
@article{dollinger2025climplicit,
  title={Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks},
  author={Dollinger, Johannes and Robert, Damien and Plekhanova, Elena and Drees, Lukas and Wegner, Jan Dirk},
  journal={International Conference on Learning Representations (ICLR) Workshops},
  year={2025}
}
```
