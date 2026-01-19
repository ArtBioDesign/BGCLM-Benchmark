#!/bin/bash
#SBATCH --job-name=hyenadna_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/hyenadna_%j.out
#SBATCH --error=logs/hyenadna_%j.err

# ============================================================
# HyenaDNA Training Script for SLURM Cluster
# Usage: sbatch run_train.sh
# ============================================================

# Create logs directory
mkdir -p logs

# Set environment
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "============================================"

# Activate conda environment
source ~/.bashrc
conda activate evo2

# Set paths
WORK_DIR=/hpcfs/fhome/yangchh/genome_lms/megaDNA/hyenaDNA
DATA_FILE=/hpcfs/fhome/yangchh/genome_lms/megaDNA/data/filtered_65536_clean.fasta
OUTPUT_DIR=${WORK_DIR}/hyena_training_$(date +%Y%m%d_%H%M%S)

cd $WORK_DIR

# Set distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training with DeepSpeed
echo "Starting distributed training with 4 GPUs..."

torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_hyena.py \
    --data_file $DATA_FILE \
    --output_dir $OUTPUT_DIR \
    --deepspeed ds_config.json \
    --d_model 512 \
    --n_layer 8 \
    --max_seq_length 65536 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_epochs 100 \
    --learning_rate 6e-4 \
    --warmup_ratio 0.03 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --bf16

echo "============================================"
echo "Training completed at: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================"
