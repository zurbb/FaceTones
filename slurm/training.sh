#!/bin/bash
#SBATCH --gres=gpu:a5000:1,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c32
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_log_gelu_0307.txt
#SBATCH --output=logs/log_training_gelu_0307.txt
#SBATCH --job-name=training_02
umask 022
/cs/ep/120/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/training.py --limit_size=150000 --validation_size=1024 --batch_size=16 --run_name=gelu_lr_0001 --epochs=20 --description="removed layernorm and attention, replaced relu with gelu, lr=0.0001, batch 16, max_margin 0.92"
