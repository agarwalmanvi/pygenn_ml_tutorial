#!/bin/bash
#SBATCH -o ./out/output_class.%a.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=1GB
python ./fusi_par4.py
