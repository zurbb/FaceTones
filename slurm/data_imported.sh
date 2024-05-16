#!/bin/bash
#SBATCH --mem=8gb
#SBATCH -c4
#SBATCH --time=0:12:0
#SBATCH --error=error_log.txt
#SBATCH --output=log.txt
#SBATCH --job-name=dataset_create
/cs/usr/zurbb/Voice-Image-Classifier/.env/bin/python3 /cs/usr/zurbb/Voice-Image-Classifier/data/youtube_downloader.py