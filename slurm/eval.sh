#!/bin/bash
#SBATCH --gres=gpu:a5000:1,vmem:24g
#SBATCH --mem=48gb
#SBATCH -c48
#SBATCH --time=4-12:00:00
#SBATCH --error=logs/error_eval_sbs_all.txt
#SBATCH --output=logs/eval_sbs_all.txt
#SBATCH --job-name=training_01
umask 022
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/playground/Voice-Image-Classifier/eval/eval_sbs_all.py \
    --run_name=only_linear_bigger_with_dropout_32_BS \
    --validation_size=50 \
    --batch_size=50 \
    --num_workers=2