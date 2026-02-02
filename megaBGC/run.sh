#!/bin/bash
#SBATCH --job-name=magabyte
#SBATCH --partition=qgpu_5090
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem=50G
#SBATCH --gres=gpu:8
#SBATCH --mail-type=end
#SBATCH --mail-user=YOU@MAIL.COM
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH -w gnode21
#SBATCH --time=2-00:00:00



echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(f'Cuda available: {torch.cuda.is_available()}'); print(f'Cuda device count: {torch.cuda.device_count()}')"

python megabgc_pretrain.py --resume_from_checkpoint ./pretraining/checkpoint-23200/