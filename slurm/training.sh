#!/bin/bash
#SBATCH --gres=gpu:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c48
#SBATCH --time=07:00:00
#SBATCH --error=error_log_training_30_05.txt
#SBATCH --output=log_training_30_05.txt
#SBATCH --job-name=training_01
/cs/ep/120/playground/Voice-Image-Classifier/.env/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/training.py