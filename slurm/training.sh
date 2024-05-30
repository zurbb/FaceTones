#!/bin/bash
#SBATCH --gres=gpu:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c48
#SBATCH --time=12:00:00
#SBATCH --error=logs/error_log_training_dino_30_05.txt
#SBATCH --output=logs/log_training_dino_30_05.txt
#SBATCH --job-name=training_02
/cs/ep/120/playground/Voice-Image-Classifier/.env/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/training.py --limit_size=16384 --validation_size=1024 --batch_size=32 --run_name=dino_30_05 --epochs=10