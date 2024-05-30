from data_loader import get_train_loader
import os 
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

LIMIT_SIZE = 1000
images_dir = os.path.join(os.getcwd(), "data/test/images")
voices_dir = os.path.join(os.getcwd(), "data/test/audio")

train_dataloader = get_train_loader(images_dir, voices_dir, batch_size=LIMIT_SIZE, limit_size=LIMIT_SIZE)

all_voices = torch.cat([voice for _, voice in train_dataloader])

# Calculate cosine similarity matrix

similarity_matrix = 1 - cosine_similarity(all_voices)

# Plot the heatmap
plt.imshow(similarity_matrix, cmap='hot_r')
plt.colorbar()
plt.title(f'(1 - Cosine Similarity) Heatmap.\n{all_voices.size(0)} voices.\nspeechbrain/spkrec-xvect-voxceleb embedding.')

# Save the plot in a file
plt.savefig('/cs/ep/120/Voice-Image-Classifier/models/heatmap_facebook.png')
