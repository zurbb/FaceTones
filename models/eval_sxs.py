from data_loader import get_train_loader
import argparse
import numpy as np
import eval_lib as lib
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (txt)")
parser.add_argument("--use_dino",action="store_true", help="Flag to indicate whether to use DINO")


args = parser.parse_args()


RESULT_FILE_PATH = args.result_file_path

def write_results(results: dict):
    # experiment_args = {
    #     "validation_size": args.validation_size,
    #     "run_name": args.run_name,
    #     "epochs": args.epochs,
    #     "model_checkpoint": args.model_checkpoint
    # }
    with open(RESULT_FILE_PATH, "w") as f:
        # f.write("Experiment arguments:\n")
        # f.write("\n".join([f"{key}: {value}" for key, value in experiment_args.items()]))
        f.write("\n\nResults:\n")
        mean_score = np.mean(list(results.values()))
        f.write(f"Mean score: {mean_score}\n")
        f.write("\n\nIndividual scores:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")




            
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
                for z in range(N):
                    # TODO: need some way to get the id of the voice/image
                    if i != z:
                        false_voice = images_and_voices[1][z].unsqueeze(0)
                        false_score = lib.cosine_similarity_loss(predict_voice, false_voice)
                        if true_score > false_score:
                            success += 1
                        
                results[i] = success/(N-1) 
                pbar.update(1)
                print(f"ID: {i} Score: {success/(N-1)}, image name: {images_and_voices[2][i]}")
            print(f"average score: {np.mean(list(results.values()))}")
            print(f"median score: {np.median(list(results.values()))}")
            print(f"variability: {np.var(list(results.values()))}")
    write_results(results)


    
if __name__ == "__main__":

    main()