from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import argparse
import eval_lib as lib


parser = argparse.ArgumentParser()
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (png)")
parser.add_argument("--use_dino",type=bool, default=True, help="Flag to indicate whether to use DINO")

args = parser.parse_args()

if __name__ == "__main__":
    train_dataloader = lib.load_validation_data(limit_size=args.validation_size, batch_size=args.validation_size, use_dino=args.use_dino)

    all_voices = next(iter(train_dataloader))[1]

    # Calculate cosine similarity matrix
    similarity_matrix = 1 - cosine_similarity(all_voices)

    # Plot the heatmap
    plt.imshow(similarity_matrix, cmap='hot_r')
    plt.colorbar()
    plt.title(f'(1 - Cosine Similarity) Heatmap.\n{all_voices.size(0)} voices.\n.')
    # Save the plot in a file
    plt.savefig('/cs/ep/120/playground/Voice-Image-Classifier/models/heatmap.png')
    plt.close()

