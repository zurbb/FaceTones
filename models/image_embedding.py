import os
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import logging
import warnings


# warnings.filterwarnings('ignore', category=Warning, module='transformers')


class DinoEmbedding:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.config.return_dict = False

    def get_embedding(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # traced_model = torch.jit.trace(self.model, [inputs.pixel_values])
            # outputs = traced_model(inputs.pixel_values)
            outputs = self.model(**inputs)
        return outputs[0]


def example(images):
    embedding = DinoEmbedding()
    return embedding.get_embedding(images)

if __name__ == "__main__":
    images_filenames = os.listdir("data/test/images")[:3]
    images = [Image.open(f"data/test/images/{filename}") for filename in images_filenames]
    embeddings = example(images)
    print(len(embeddings))
    print(embeddings[0].shape)
