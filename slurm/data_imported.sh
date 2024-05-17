#!/bin/bash
#SBATCH --mem=32gb
#SBATCH -c8
#SBATCH --time=2:0:0
#SBATCH --error=error_log.txt
#SBATCH --output=log.txt
#SBATCH --job-name=dataset_create
#SBATCH --killable
#SBATCH --requeue
/cs/usr/zurbb/Voice-Image-Classifier/.env/bin/python3 /cs/usr/zurbb/Voice-Image-Classifier/data/youtube_downloader.py