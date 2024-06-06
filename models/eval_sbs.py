import argparse
import numpy as np
import eval_lib as lib
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (txt)")
parser.add_argument("--use_dino",type=bool, default=True, help="Flag to indicate whether to use DINO")


args = parser.parse_args()

IMAGE_DIR = "data/test/images/"
AUDIO_DIR = "data/test/audio/"
RESULT_FILE_PATH = args.result_file_path

def write_results(results: dict):
    # experiment_args = {
    #     "validation_size": args.validation_size,
    #     "run_name": args.run_name,
    #     "epochs": args.epochs,
    #     "model_checkpoint": args.model_checkpoint
    # }
    
    root_dir = "/cs/ep/120/Voice-Image-Classifier/"
    image_dir = root_dir + IMAGE_DIR
    voice_dir = root_dir + AUDIO_DIR
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
    model =  lib.load_model_by_checkpoint(args.model_checkpoint)
    model.eval()
    with torch.inference_mode():
        print("Loading validation data")
        validation_data = lib.load_validation_data(limit_size=args.validation_size, batch_size=args.validation_size, use_dino=args.use_dino)
        images_and_voices = next(iter(validation_data))
        
        N= len(images_and_voices[0])
        results = {} # dictionart with id and scores for each id

        with tqdm.tqdm(total=N, desc="Processing", bar_format="{l_bar}{bar}{r_bar}", ncols=80, colour='green') as pbar:
            for i in range(N):
                print("Processing ID: ", i)
                image = images_and_voices[0][i].unsqueeze(0)
                true_voice = images_and_voices[1][i].unsqueeze(0)
                predict_voice = model(image)
                success = 0
                true_score = lib.cosine_similarity_loss(predict_voice, true_voice)
                best_false_score, best_false_id = 0, 0
                for z in range(N):
                    # TODO: need some way to get the id of the voice/image
                    if i != z:
                        false_voice = images_and_voices[1][z].unsqueeze(0)
                        false_score = lib.cosine_similarity_loss(predict_voice, false_voice)
                        if true_score > false_score:
                            success += 1
                        if false_score > best_false_score:
                            best_false_score = false_score
                            best_false_id = z
                        
                results[i] = success/(N-1), images_and_voices[2][i], images_and_voices[2][best_false_id], true_score-best_false_score # score, true image name, best false image name, diff between true and best false scores
                pbar.update(1)
                print(f"ID: {i} Score: {success/(N-1)}, image name: {images_and_voices[2][i]}")
            print(f"average score: {np.mean([value[0] for value in results.values()])}")
            print(f"median score: {np.median([value[0] for value in results.values()])}")
            print(f"variability: {np.var([value[0] for value in results.values()])}")
    write_results(results)


    
if __name__ == "__main__":

    main()