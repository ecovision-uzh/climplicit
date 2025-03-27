# Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks
This repository contains the code for [**Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks**](*).
The work extends the [SIREN](https://arxiv.org/abs/2006.09661) backbone with residual connections (ReSIREN) and trains it on the dense and global climatic raster [CHELSA](https://chelsa-climate.org/). The code uses the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) framework with [Hydra](https://hydra.cc/docs/intro/).

Note1: his work started out as a contrastive framework inspired by [SatCLIP](https://github.com/microsoft/satclip) and Climplicit only emerged from the ablation of regressing the climatic data directly. Therefore the code structure is still contrastive, and everything is named ChelsaCLIP.

Note2: This project was done in a very explorative fashion, and the code does not lend itself to easily replicate the whole training process, as the input data was quite heavily preprocessed. This codebase is supposed to inspire adaptions, but should not itself be used directly. To use the pretrained Climplicit embeddings, please follow the Quickstart tutorial below.
# Quickstart
The quickstart folder contains the least amount of files required to showcase the usage of the pretrained climplicit embeddings:
```bash
git clone https://github.com/ecovision-uzh/climplicit.git
cd climplicit/quickstart
conda env create -f environment.yml
conda activate climplicit
python climplicit.py
```
A advanced tutorial on how to use such pretrained embeddings for downstream learning can be found in the [SatCLIP](https://github.com/microsoft/satclip) codebase.

# Citation
...
