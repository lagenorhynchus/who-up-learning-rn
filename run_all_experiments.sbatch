#!/bin/bash
#SBATCH --job-name=all_experiments
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err
#SBATCH --time=72:00:00
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

# Create directories for each experiment
mkdir -p experiments/exp1_default
mkdir -p experiments/exp2_shape_biased
mkdir -p experiments/exp3_shape_biased_learnable
mkdir -p experiments/exp4_extreme_shape
mkdir -p logs

echo "=========================================="
echo "Starting Experiment 1: Default (Alpha=0.5, Learnable)"
echo "=========================================="
python train.py \
    --epochs 40 \
    --batch_size 128 \
    --num_classes 100 \
    --learnable_alpha \
    --save_dir ./experiments/exp1_default

echo "=========================================="
echo "Starting Experiment 2: Shape-Biased (Alpha=0.8, Fixed)"
echo "=========================================="
python train.py \
    --epochs 40 \
    --batch_size 128 \
    --num_classes 100 \
    --alpha 0.8 \
    --save_dir ./experiments/exp2_shape_biased

echo "=========================================="
echo "Starting Experiment 3: Shape-Biased Learnable (Alpha=0.8 start)"
echo "=========================================="
python train.py \
    --epochs 40 \
    --batch_size 128 \
    --num_classes 100 \
    --alpha 0.8 \
    --learnable_alpha \
    --save_dir ./experiments/exp3_shape_biased_learnable

echo "=========================================="
echo "Starting Experiment 4: Extreme Shape (Alpha=0.9, Fixed)"
echo "=========================================="
python train.py \
    --epochs 40 \
    --batch_size 128 \
    --num_classes 100 \
    --alpha 0.9 \
    --save_dir ./experiments/exp4_extreme_shape

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="

deactivate
