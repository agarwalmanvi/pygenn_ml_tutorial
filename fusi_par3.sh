#!/bin/bash
#SBATCH -o ./out/outputg.%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=1GB
python ./fusi_par3.py
