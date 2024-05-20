#!/bin/bash
#SBATCH --mem=32gb
#SBATCH -c48
#SBATCH --time=4-00:00:00
#SBATCH --error=error_log_train_20_05.txt
#SBATCH --output=log_train_20_05.txt
#SBATCH --job-name=dataset_create_train
#SBATCH --killable
#SBATCH --requeue
/cs/ep/120/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/data/youtube_downloader.py
