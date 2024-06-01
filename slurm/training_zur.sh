#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c32
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_log_training_dino_31_05_zur_2.txt
#SBATCH --output=logs/log_training_dino_31_05_zur_2.txt
#SBATCH --job-name=training_021
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py --limit_size=100000 --validation_size=10000 --batch_size=32 --run_name=dino_31_05_zur_2 --epochs=20