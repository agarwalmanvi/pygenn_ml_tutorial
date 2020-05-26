#!/bin/bash
#SBATCH -o ./out/output.%a.out
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB
python ./fusi_par.py