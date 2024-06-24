#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_entropy_learned_2206_upper_bond_09_lr_00001.txt
#SBATCH --output=logs/entropy_learned_2206_upper_bond_09_lr_00001.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py \
    --limit_size=100000 \
    --validation_size=1024 \
    --batch_size=16 \
    --run_name=2206_postive_punish \
    --epochs=20 \
    --num_workers=16 \
    --description="added mean of the sim matrix diagonal to the loss (-1) to ensure punishment of distance from the positive vs negative. margin can be "