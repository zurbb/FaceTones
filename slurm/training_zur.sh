#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_yossi_debug.txt
#SBATCH --output=logs/entropy_yossi_debug.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py \
    --limit_size=100000 \
    --validation_size=1024 \
    --batch_size=16 \
    --run_name=2706_yossi_debug \
    --epochs=20 \
    --num_workers=16\
    --description="with positive pusishment and entropy learned, upper bond 0.9, lr 0.00001"