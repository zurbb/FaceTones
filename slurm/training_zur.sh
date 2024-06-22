#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c32
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_entopy_09_2106.txt
#SBATCH --output=logs/entopy_09_2106.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py --limit_size=100000 --validation_size=1024 --batch_size=32 --run_name=entopy_09_2106 --epochs=20