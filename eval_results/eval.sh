#!/bin/bash

# Set only:
folder_name="eval_results"
model_checkpoint='/cs/ep/120/Voice-Image-Classifier/checkpoints/checkpoint_19.pth'
validation_size=20
dir_path="/cs/ep/120/playground/Voice-Image-Classifier/$folder_name"

# Don't change
umask 022
mkdir -p "$dir_path"
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 models/generate_audio_similarity_heatmap.py \
--model_checkpoint="$model_checkpoint" \
--result_file_path="$dir_path/heatmap.png" \
--use_dino \
--validation_size="$validation_size"

/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 models/eval_sxs.py \
--model_checkpoint="$model_checkpoint" \
--result_file_path="$dir_path/results.txt" \
--use_dino\
--validation_size="$validation_size"
