#!/bin/bash
#SBATCH --job-name=dual_alpha08
#SBATCH --output=logs/alpha08_%j.out
#SBATCH --error=logs/alpha08_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf
source venv/bin/activate
mkdir -p logs

python train.py --epochs 40 --batch_size 128 --num_classes 100 --alpha 0.8 --save_dir ./models

deactivate
