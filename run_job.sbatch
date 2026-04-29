#!/bin/bash
#SBATCH --job-name=dual_stream_cnn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load modules
module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

# Activate virtual environment
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Print GPU info for debugging
nvidia-smi

# Run training
python train.py --epochs 50 --use_10_class

# Deactivate
deactivate
