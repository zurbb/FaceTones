import argparse
import numpy as np
import eval_lib as lib
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (txt)")

print("Parsing arguments")
args = parser.parse_args()
print("Arguments parsed")
IMAGE_DIR = "data/test/images/"
AUDIO_DIR = "data/test/audio/"
RESULT_FILE_PATH = args.result_file_path
ROOT_DIR = "/cs/ep/120/Voice-Image-Classifier/"

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
    false_names = [value[2] for value in results.values()]
    score_diffs = [value[3] for value in results.values()]
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
            f.write(f"# [True Image file]({image_dir+true_names[i]})\n")
            f.write(f"# [False Image file]({image_dir+false_names[i]})\n")
            f.write(f"# [Voice file]({voice_dir+true_names[i].replace('_0.jpg','.mp3')})\n")
            f.write(f"Score difference: {score_diffs[i]}\n")
            f.write("\n")


def main():
    model =  lib.load_model_by_checkpoint(f"{args.model_checkpoint}")
    model.eval()
    with torch.inference_mode():
        print("Loading validation data")
        validation_data = lib.load_validation_data(limit_size=args.validation_size, batch_size=32, use_dino=True)
        with tqdm.tqdm(total=np.ciel(args.validation_size/args.batch_size), desc="Processing", bar_format="{l_bar}{bar}{r_bar}", ncols=80, colour='green') as pbar:
            idx = 0
            results = {}
            for images_and_voices in validation_data:
                pbar.set_description(f"Processing ID: {idx}")
                idx += 1
                N = len(images_and_voices[0])
                for i in range(N):
                    image = images_and_voices[0][i].unsqueeze(0)
                    true_voice = images_and_voices[1][i].unsqueeze(0)
                    predict_voice = model(image)
                    success = 0
                    true_score = lib.cosine_similarity_loss(predict_voice, true_voice)
                    best_false_score, best_false_id = 0, 0
                    for z in range(N):
                        if i != z:
                            false_voice = images_and_voices[1][z].unsqueeze(0)
                            false_score = lib.cosine_similarity_loss(predict_voice, false_voice)
                            if true_score > false_score:
                                success += 1
                            if false_score > best_false_score:
                                best_false_score = false_score
                                best_false_id = z
                            
                    results[idx*args.batch_size + i] = success/(N-1), images_and_voices[2][i], images_and_voices[2][best_false_id], true_score-best_false_score # score, true image name, best false image name, diff between true and best false scores
                pbar.update(1)
                pbar.set_postfix({"Score": success/(N-1), "Image Name": images_and_voices[2][i]})
            print(f"average score: {np.mean([value[0] for value in results.values()])}")
            print(f"median score: {np.median([value[0] for value in results.values()])}")
            print(f"variability: {np.var([value[0] for value in results.values()])}")
    write_results(results)

    
if __name__ == "__main__":

    main()