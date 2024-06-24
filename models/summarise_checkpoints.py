from eval_sbs import main
import argparse
import os
import re


def extract_number(s):
    # Extract number from the string
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Checkpoint file name")
    parser.add_argument("--validation_size", type=int, default=256, help="Validation size of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size of the dataset")
    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    model_name = args.model_name
    args.result_file_path = None
    checkpoints = list(sorted(os.listdir(f"trained_models/{model_name}/"), key=extract_number))
    print(f"number of checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints:
        args.model_checkpoint = f"{model_name}/{checkpoint}"
        print(f"checkpoint: {checkpoint}")
        main(args=args, write_results=False)
        print("\n\n")


if __name__ == "__main__":
    run()