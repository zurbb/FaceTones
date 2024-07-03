#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_0207_back_to_best_bigger_lr.txt
#SBATCH --output=logs/0207_back_to_best_bigger_lr.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py \
    --limit_size=100000 \
    --validation_size=1024 \
    --batch_size=16 \
    --run_name=0207_back_to_best_bigger_lr \
    --epochs=30 \
    --num_workers=11 \
    --description="Back to the best model, but with a bigger learning rate and param num"