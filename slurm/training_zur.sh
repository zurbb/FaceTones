#!/bin/bash
#SBATCH --gres=gpu:a5000:3,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_0107_vgg_style.txt
#SBATCH --output=logs/107_vgg_style.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/models/training.py \
    --limit_size=100000 \
    --validation_size=1024 \
    --batch_size=16 \
    --run_name=0107_vgg_style \
    --epochs=30 \
    --num_workers=11 \
    --description="used vgg inspired nn, upper bond 0.9, lr 0.00001, removed dropout in the end"