from eval_sbs import main
import argparse
import os


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
    checkpoints = list(sorted(os.listdir(f"trained_models/{model_name}/")))
    print(f"number of checkpoints: {len(checkpoints)}")
    for checkpoint in checkpoints:
        args.model_checkpoint = f"{model_name}/{checkpoint}"
        print(f"checkpoint: {checkpoint}")
        print(f"average score: {main(args=args, write_results=False)}")
        print("\n\n")


if __name__ == "__main__":
    run()