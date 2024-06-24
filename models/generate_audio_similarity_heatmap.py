from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import argparse
import eval_lib as lib
import torch.nn.functional as F
import numpy as np
import torch
import tqdm

# parser = argparse.ArgumentParser()
# parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
# parser.add_argument("--model_checkpoint", type=str, required=True, help="Checkpoint file name")
# parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (png)")
# parser.add_argument("--use_dino",action="store_true", help="Flag to indicate whether to use DINO")

# args = parser.parse_args()

if __name__ == "__main__":
    val_loader = lib.load_validation_data(limit_size=512, batch_size=16, use_dino=True)
    averages = []
    for images, voices,_ in tqdm.tqdm(val_loader):
        # Calculate cosine similarity matrix
        sim = cosine_similarity(voices)
        # voices = F.normalize(voices, p=2, dim=1)
        # sim = torch.tensordot(images, voices.T, dims=1) # simialrities, [n,n]
        n = sim.shape[0]

        for i in range(n):
            row = sim[i]
            # Exclude the diagonal element
            non_diag_elements = np.delete(row, i)
            # Compute the average of the non-diagonal elements
            avg = np.mean(non_diag_elements)
            averages.append(avg)
    print(np.mean(averages))
        # similarity_matrix = 1 - cosine_similarity(all_voices)

        # # Plot the heatmap
        # plt.imshow(similarity_matrix, cmap='hot_r')
        # plt.colorbar()
        # plt.title(f'(1 - Cosine Similarity) Heatmap.\n{all_voices.size(0)} voices.\n.')
        # # Save the plot in a file
        # plt.savefig('/cs/ep/120/playground/Voice-Image-Classifier/models/heatmap.png')
        # plt.close()

