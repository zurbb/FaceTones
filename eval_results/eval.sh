#!/bin/bash

# set only :
folder_name="eval_results"
model_checkpoint='/cs/ep/120/Voice-Image-Classifier/checkpoints/dino_31_05.pth'
use_dino=True
validation_size=1024


# dont change
umask 022
dir_path="/cs/ep/120/Voice-Image-Classifier/$folder_name"
mkdir "$dir_path"
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/generate_audio_similarity_heatmap.py 
--model_checkpoint="$model_checkpoint"
--result_file_path="$dir_path/heatmap.png" 
--use_dino="$use_dino" 
--validation_size="$validation_size"
# 
#
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/eval_sxs.py 
--model_checkpoint="$model_checkpoint"
--result_file_path="$dir_path/results.txt" 
--use_dino="$use_dino" 
--validation_size="$validation_size"
