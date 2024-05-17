#!/bin/bash
#SBATCH --mem=16gb
#SBATCH -c32
#SBATCH --time=2:0:0
#SBATCH --error=error_log.txt
#SBATCH --output=log.txt
#SBATCH --job-name=dataset_create
#SBATCH --killable
#SBATCH --requeue
/cs/usr/yedidyat/Project/Voice-Image-Classifier/.venv/bin/python3 /cs/usr/yedidyat/Project/Voice-Image-Classifier/data/youtube_downloader.py