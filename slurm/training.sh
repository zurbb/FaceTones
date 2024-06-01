#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c16
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_log_training_dino_31_05.txt
#SBATCH --output=logs/log_training_dino_31_05.txt
#SBATCH --job-name=training_02
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/training.py --limit_size=16000 --validation_size=1024 --batch_size=32 --run_name=dino_31_05 --epochs=10