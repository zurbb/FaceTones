import argparse
import os
from time import sleep
import numpy as np
import models.eval_lib as lib
import torch
import tqdm


IMAGE_DIR = "data/test/images/"
AUDIO_DIR = "data/test/audio/"
# IMAGE_DIR = "data/yedidya_tal/images/"
# AUDIO_DIR = "data/yedidya_tal/audio/"

ROOT_DIR = "/cs/ep/120/Voice-Image-Classifier/"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_size", type=int, default=100, help="Validation size of the dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (txt)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset")
    args = parser.parse_args()
    return args

def write_results(results: dict):
    # experiment_args = {
    #     "validation_size": args.validation_size,
    #     "run_name": args.run_name,
    #     "epochs": args.epochs,
    #     "model_checkpoint": args.model_checkpoint
    # }
    
    image_dir = ROOT_DIR + IMAGE_DIR
    voice_dir = ROOT_DIR + AUDIO_DIR
    scores = [value[0] for value in results.values()]
    true_names = [value[1] for value in results.values()]
    best_false_names = [value[2] for value in results.values()]
    worst_false_names = [value[3] for value in results.values()]
    true_scores = [value[4] for value in results.values()]
    best_false_scores = [value[5] for value in results.values()]
    worst_false_scores = [value[6] for value in results.values()]

    with open(RESULT_FILE_PATH, "w") as f:
        # f.write("Experiment arguments:\n")
        # f.write("\n".join([f"{key}: {value}" for key, value in experiment_args.items()]))
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


def main(args, write_results=True):
    model =  lib.load_model_by_checkpoint(f"{args.model_checkpoint}")
    model.eval()
    print(f"loss margin: {model.loss_func.learnable_param.item()}")
    with torch.inference_mode():
        print("Loading validation data")
        batch_size = args.batch_size
        validation_data = lib.load_validation_data(limit_size=args.validation_size, batch_size=batch_size, use_dino=True)
        with tqdm.tqdm(total=np.ceil(args.validation_size), desc="Processing", bar_format="{l_bar}{bar}{r_bar}", ncols=80, colour='green') as pbar:
            results = {}
            true_scores = []
            false_scores = []
            for batch_num, images_and_voices in enumerate(validation_data):
                N = len(images_and_voices[0])
                for i in range(N):
                    pbar.set_description(f"Processing ID: {batch_num}")
                    image = images_and_voices[0][i].unsqueeze(0)
                    true_voice = images_and_voices[1][i].unsqueeze(0)
                    predict_voice = model(image)
                    success = 0
                    true_score = lib.cosine_similarity(predict_voice, true_voice).item()
                    true_scores.append(true_score)
                    best_false_score, best_false_id = 0, 0
                    worst_false_score, worst_false_id = 1, 0
                    for z in range(N):
                        if i != z:
                            false_image = images_and_voices[0][z].unsqueeze(0)
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
        print(f"average true similarity: {np.mean(true_scores)}")
        print(f"average false similarity: {np.mean(false_scores)}")
        print(f"average score: {np.mean([value[0] for value in results.values()])}")
        print(f"median score: {np.median([value[0] for value in results.values()])}")
        print(f"variability: {np.var([value[0] for value in results.values()])}")
    if write_results:
        write_results(results)
    return np.mean([value[0] for value in results.values()])

    
if __name__ == "__main__":
    args = parse_args()
    RESULT_FILE_PATH = args.result_file_path
    main(args=args)