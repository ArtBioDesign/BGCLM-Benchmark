#!/bin/bash
#SBATCH --job-name=magabyte
#SBATCH --partition=qgpu_a40
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err



python extract_emb.py