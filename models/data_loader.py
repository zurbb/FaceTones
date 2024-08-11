from models.image_embedding import DinoEmbedding
import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.voice_to_vec import VoiceToVec
from torchvision.transforms.functional import pil_to_tensor


import coloredlogs, logging
import time

logger = logging.getLogger()
coloredlogs.install()

IMAGE_SUFFIX = "_0.jpg"
VOICE_SUFFIX = ".mp3"

class ImagesVoicesDataset(Dataset):
    """
    A custom dataset class for handling image and voice data pairs.

    This dataset class is designed to load and pair images with corresponding voice clips from a specified directory structure.
    It supports transformations for images and can be used with PyTorch DataLoader for efficient data loading in training or
    inference pipelines.
    Note: It may be more efficient to load the models a preform the encodings in the training instead, to utilize the GPU.
    """


    def __init__(self, images_dir, voices_dir, transform=None, voice_transformer=None, limit_size=None, dino_embedding=False):
        self.images_dir = images_dir
        self.voices_dir = voices_dir
        self.transform = transform
        self.voice_transform = voice_transformer.get_signals
        self.voice_embedder = voice_transformer.get_embedding
        image_files = os.listdir(images_dir)
        voice_files = os.listdir(voices_dir)
        image_ids = [img.split(IMAGE_SUFFIX)[0] for img in image_files]
        voice_ids = [voice.split(VOICE_SUFFIX)[0] for voice in voice_files]
        ids = list(set(image_ids).intersection(set(voice_ids)))  # Get the common ids
        if limit_size:
            ids = ids[:limit_size]
        self.images = [f"{id}{IMAGE_SUFFIX}" for id in ids]
        self.voices = [f"{id}{VOICE_SUFFIX}" for id in ids]
        self.images_and_voices_file_names = list(zip(self.images, self.voices))
        self.dino_embedding = dino_embedding
        if self.dino_embedding:
            self.dino = DinoEmbedding()
        else:
            self.dino = None


    def __len__(self):
        return len(self.images_and_voices_file_names)

    def __getitem__(self, idx):
        """
        Loads and returns an image,voice clip, image_name triplet from the dataset.
        """
        img_name, voice_name = self.images_and_voices_file_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        voice_path = os.path.join(self.voices_dir, voice_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        voice_embedding = self.voice_transform(voice_path)
        voice_embedding = self.voice_embedder(voice_embedding)
        return image , voice_embedding, img_name
    

    # Custom collate function
    def custom_collate_fn(self, batch):
        """
        stacks the images and voices in the batch into a single tensor.
        encoding the images using the DINO model if specified.
        """
        images = torch.stack([item[0] for item in batch])
        voices = torch.stack([item[1] for item in batch])
        if self.dino_embedding:
            images = self.dino.get_embedding(images)
            images = images.unsqueeze(1)
        image_names = [item[2] for item in batch]
        return images, voices, image_names

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def get_train_loader(images_dir, voices_dir, batch_size=4, shuffle=True, num_workers=11, limit_size=None, dino=True):
    """
    Create and return a DataLoader for training data.

    Args:
        images_dir (str): Directory path containing the images.
        voices_dir (str): Directory path containing the voice files.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 11.
        limit_size (int, optional): Maximum number of samples to load. Defaults to None (no limit).
        dino (bool, optional): Whether to use DINO embedding. Defaults to True. 

    Returns:
        torch.utils.data.DataLoader: DataLoader object for training data.
    """
    logging.debug("Creating loader")
    voice_embedder = VoiceToVec()
    logger.debug("Voice embedder created")
    logger.debug("Creating DatasetLoader")
    train_dataset = ImagesVoicesDataset(images_dir, voices_dir, transform=transform, voice_transformer=voice_embedder, limit_size=limit_size, dino_embedding=dino)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=train_dataset.custom_collate_fn)
    logger.debug("Train loader created")
    return train_loader