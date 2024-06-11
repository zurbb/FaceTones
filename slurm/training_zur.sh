#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:16g
#SBATCH --mem=32gb
#SBATCH -c32
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_1106_entropy_learned_m.txt
#SBATCH --output=logs/1106_entropy_learned_m.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py --limit_size=100000 --validation_size=1024 --batch_size=32 --run_name=entopy_zur_margin_learned --epochs=20