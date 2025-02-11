import argparse
import os
import sys
from time import sleep
import numpy as np
import torch
import tqdm
import logging
import coloredlogs
sys.path.append(os.path.abspath(os.getcwd()))
import models.eval_lib as lib

logger = logging.getLogger()
coloredlogs.install()

IMAGE_DIR = "data/test/images/"
AUDIO_DIR = "data/test/audio/"

ROOT_DIR = "/cs/ep/120/Voice-Image-Classifier/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_size", type=int, default=100, help="Validation size of the dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (txt)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    args = parser.parse_args()
    return args

def write_results(results: dict, args):
    """
    Write the evaluation results to a file.

    Args:
        results (dict): A dictionary containing the evaluation results.
        args: The command-line arguments passed to the script.

    Returns:
        None
    """
    
    experiment_args = {
        "validation_size": args.validation_size,
        "model_checkpoint": args.model_checkpoint,
        "batch_Size": args.batch_size
    }
    
    image_dir = ROOT_DIR + IMAGE_DIR
    voice_dir = ROOT_DIR + AUDIO_DIR
    scores = [value[0] for value in results.values()]
    true_names = [value[1] for value in results.values()]
    best_false_names = [value[2] for value in results.values()]
    worst_false_names = [value[3] for value in results.values()]
    true_scores = [value[4] for value in results.values()]
    best_false_scores = [value[5] for value in results.values()]
    worst_false_scores = [value[6] for value in results.values()]
    
    with open(args.result_file_path, "w") as f:
        f.write("Experiment arguments:\n")
        f.write("\n".join([f"{key}: {value}" for key, value in experiment_args.items()]))
        f.write("\n\nResults:\n")
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        variance = np.var(scores)
        f.write(f"Mean score: {mean_score}\n")
        f.write(f"Median score: {median_score}\n")
        f.write(f"Variance: {variance}\n")
        f.write("\n\nIndividual scores:\n")
        for i, score in enumerate(scores):
            f.write(f"{i}: {score}\n")
            f.write(f"# [True Image - Similarity: {true_scores[i]}]({image_dir+true_names[i]})\n")
            f.write(f"# [True Voice]({voice_dir+true_names[i].replace('_0.jpg','.mp3')})\n")
            if best_false_scores[i] >= true_scores[i]:
                f.write(f"# [Fail Image - Similarity: {best_false_scores[i]}]({image_dir+best_false_names[i]})\n")
                f.write(f"# [Fail Voice]({voice_dir+best_false_names[i].replace('_0.jpg','.mp3')})\n")
            if worst_false_scores[i] <= true_scores[i]:
                f.write(f"# [Success Image - Similarity: {worst_false_scores[i]}]({image_dir+worst_false_names[i]})\n")
                f.write(f"# [Success Voice]({voice_dir+worst_false_names[i].replace('_0.jpg','.mp3')})\n")

        
            f.write("\n")


def main(args):
    """
    Load the model and evaluate it on the validation data.
    Evaluation method: two-images-one-voice. Given two images and a voice of one of them,
    check which one is more similar to the voice. If the model predicts the correct image, it is a success.
    Calcualate the success rate of the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lib.load_model_by_checkpoint(f"{args.model_checkpoint}").to(device)
    model.eval()
    logger.info(f"eval checkpoint: {args.model_checkpoint}")
    logger.info(f"loss margin: {model.loss_func.learnable_param.item()}")
    with torch.no_grad():
        logger.info("Loading validation data")
        batch_size = args.batch_size
        validation_data = lib.load_validation_data(limit_size=args.validation_size, batch_size=batch_size, use_dino=True,num_workers=args.num_workers)
        with tqdm.tqdm(total=np.ceil(args.validation_size), desc="Processing", bar_format="{l_bar}{bar}{r_bar}", ncols=80, colour='green') as pbar:
            results = {}
            true_scores = []
            false_scores = []
            for batch_num, images_and_voices in enumerate(validation_data):
                N = len(images_and_voices[0])
                for i in range(N):
                    pbar.set_description(f"Processing ID: {batch_num}")
                    image = images_and_voices[0][i].unsqueeze(0).to(device)
                    true_voice = images_and_voices[1][i].unsqueeze(0).to(device)
                    predict_voice = model(image)
                    success = 0
                    true_score = lib.cosine_similarity(predict_voice, true_voice).item()
                    true_scores.append(true_score)
                    best_false_score, best_false_id = 0, 0
                    worst_false_score, worst_false_id = 1, 0
                    for z in range(N):
                        if i != z:
                            false_image = images_and_voices[0][z].unsqueeze(0).to(device)
                            false_predict_voice = model(false_image)
                            false_score = lib.cosine_similarity(false_predict_voice, true_voice).item()
                            false_scores.append(false_score)
                            if true_score > false_score:
                                success += 1
                            if false_score > best_false_score:
                                best_false_score = false_score
                                best_false_id = z
                            if false_score < worst_false_score:
                                worst_false_score = false_score
                                worst_false_id = z
                    pbar.update(1)
                    results[batch_num*batch_size + i] = (success/(N-1), images_and_voices[2][i], 
                    images_and_voices[2][best_false_id], images_and_voices[2][worst_false_id], 
                    true_score, best_false_score, worst_false_score)
                      # success rate, true image name, best false image name, worst false image name, true score, best false score, worst false score
                    pbar.set_postfix({"Score": success/(N-1)})
        logger.info(f"average true similarity: {np.mean(true_scores)}")
        logger.info(f"average false similarity: {np.mean(false_scores)}")
        logger.info(f"average score: {np.mean([value[0] for value in results.values()])}")
        logger.info(f"median score: {np.median([value[0] for value in results.values()])}")
        logger.info(f"variability: {np.var([value[0] for value in results.values()])}")
    write_results(results,args)

    
if __name__ == "__main__":
    args = parse_args()
    main(args=args)