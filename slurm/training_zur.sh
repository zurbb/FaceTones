#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c32
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_log_training_dino_3_06_zur.txt
#SBATCH --output=logs/log_training_dino_3_06_zur.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py --limit_size=100000 --validation_size=10000 --batch_size=32 --run_name=3_06_zur_only_one_dropout --epochs=20