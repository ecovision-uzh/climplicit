#!/bin/bash
##SBATCH -J lcd
##SBATCH -p wegner
##SBATCH --time=48:0:0
##SBATCH -C foo
##SBATCH -c 16
##SBATCH -n 1
##SBATCH --gres=gpu:1
##SBATCH --constraint="GPUMEM16GB"
##SBATCH --mem=42G

python src/train.py experiment=lcd logger=wandb
