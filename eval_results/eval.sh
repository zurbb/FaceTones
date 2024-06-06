#!/bin/bash

<<<<<<< HEAD
# Set only:
folder_name="eval_results"
model_checkpoint='/cs/ep/120/Voice-Image-Classifier/checkpoints/checkpoint_19.pth'
validation_size=20
dir_path="/cs/ep/120/playground/Voice-Image-Classifier/$folder_name"
=======
# set only :
folder_name="drop_01_checkpoint_7"
model_checkpoint='/cs/ep/120/Voice-Image-Classifier/trained_models/dropout_01/checkpoint_7.pth'
use_dino=True
validation_size=10
>>>>>>> 3f83e566866bbbdf9a25d0e76b642e9030a7e976

# Don't change
umask 022
<<<<<<< HEAD
mkdir -p "$dir_path"
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 models/generate_audio_similarity_heatmap.py \
--model_checkpoint="$model_checkpoint" \
--result_file_path="$dir_path/heatmap.png" \
--use_dino \
=======
dir_path="/cs/ep/120/Voice-Image-Classifier/eval_results/$folder_name"
mkdir "$dir_path"
/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 /cs/ep/120/Voice-Image-Classifier/models/generate_audio_similarity_heatmap.py 
--model_checkpoint="$model_checkpoint"
--result_file_path="$dir_path/heatmap.png" 
--use_dino="$use_dino" 
>>>>>>> 3f83e566866bbbdf9a25d0e76b642e9030a7e976
--validation_size="$validation_size"

/cs/ep/120/playground/Voice-Image-Classifier/.venv/bin/python3 models/eval_sxs.py \
--model_checkpoint="$model_checkpoint" \
--result_file_path="$dir_path/results.txt" \
--use_dino\
--validation_size="$validation_size"


