#!/bin/bash
#SBATCH --gres=gpu:a5000:1,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_only_linear_bigger_with_dropout.txt
#SBATCH --output=logs/only_linear.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py \
    --limit_size=100000 \
    --validation_size=1024 \
    --batch_size=32 \
    --run_name=only_linear_bigger_with_dropout \
    --epochs=20 \
    --num_workers=12 \
    --description="only linear bigger model , lr=0.00001, with more drop-out and batch size 32" 