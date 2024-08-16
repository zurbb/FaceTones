import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))

from models.eval_lib import load_model_by_checkpoint
import torch
from enum import Enum
from dataclasses import dataclass
import random
from torchvision import transforms
from PIL import Image
from models.voice_to_vec import VoiceToVec
from models.image_embedding import DinoEmbedding

CHECKPOINT = "only_linear/checkpoint_8.pth"

root = '/cs/ep/120/Voice-Image-Classifier'

class Gender(Enum):
    """
    Enum for representing gender categories.
    """
    MALE = "male"
    FEMALE = "female"

@dataclass
class DataItem:
    """
    Data class for representing a single data item in the dataset.
    
    Attributes:
        gender (Gender): The gender of the person in the voice and image data.
        image_path (str): File path to the image data.
        voice_path (str): File path to the voice data.
    
    Include hash and less than methods for comparing DataItem objects and using them in sets.
    """
    gender: Gender
    image_path: str
    voice_path: str
    def __hash__(self):
        return hash((self.gender, self.image_path, self.voice_path))
    def __lt__(self, other):
        return self.gender.value < other.gender.value or \
               self.image_path < other.image_path or \
               self.voice_path < other.voice_path

class GuiBackend:
    """
    Backend class for the GUI application, responsible for managing the data loading,
    preprocessing, and model inference processes.

    This class initializes the paths for male and female voice and image data,
    loads a pre-trained model from a specified checkpoint, and prepares the data items for processing. 

    Attributes:
        male_path (str): Path to the file containing male data items.
        female_path (str): Path to the file containing female data items.
        model: The loaded model from the specified checkpoint for inference.
        male_items (list[DataItem]): List of male data items prepared for processing.
        female_items (list[DataItem]): List of female data items prepared for processing.
        voice_embedder (VoiceToVec): Instance of the VoiceToVec class for converting voice data to vector representations.
        dino (DinoEmbedding): Instance of the DinoEmbedding class for obtaining image embeddings.
        seen (set): A set to keep track of processed data items to avoid duplicates.
    """
    data_loader = None

    def __init__(self):
        self.male_path =os.path.join(ROOT_DIR, 'gui/male.txt')
        self.female_path = os.path.join(ROOT_DIR, 'gui/females.txt')
        # self.model =  load_model_by_checkpoint(CHECKPOINT, hard_checkpoint=True)
        self.male_items = self.make_data_items_list(self.male_path,Gender.MALE)
        self.female_items = self.make_data_items_list(self.female_path,Gender.FEMALE)
        self.voice_embedder = VoiceToVec()
        self.dino = DinoEmbedding()
        self.seen = set()

    def make_data_items_list(self,data_source_path:str, gender:Gender) -> list[DataItem]:
        """
        Reads the data source file and creates a list of DataItem objects for the specified IDs.

        Parameters:
        - data_source_path (str): The path to the file containing the data source IDs
        
        Returns:
        - data_items (list[DataItem]): A list of DataItem objects created from the data source IDs.
        """

        data_items = []
        with open(data_source_path, 'r') as f:
            data_sources = [line.strip() for line in f.readlines()]
        for youtube_id in data_sources:
            image_path = os.path.join(ROOT_DIR, 'data/evaluation/images', youtube_id+ "_0.jpg")
            voice_path = os.path.join(ROOT_DIR,'data/evaluation/audio', youtube_id +".mp3")
            data_item = DataItem(gender=gender, image_path=image_path, voice_path=voice_path)
            data_items.append(data_item)
        return data_items
    

    
    def get_image(self, image_path):
        """
        Retrieves the image from the given image path and returns its embedding.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - embedding (torch.Tensor): The embedding of the image.
        """
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path)
        image = transform(image)
        image = self.dino.get_embedding(image).unsqueeze(0)
        return image
    
    
    def get_voice(self, voice_path):
        """
        Retrieves the voice signal from the given voice path and returns its embedding.

        Parameters:
        - voice_path (str): The path to the voice file.

        Returns:
        - embedding (torch.Tensor): The embedding of the voice signal.
        """
        signal = self.voice_embedder.get_signals(voice_path)
        return self.voice_embedder.get_embedding(signal)
        
    def get_two_random_items(self, dificulty_level):
        """
        Retrieves two random items from the dataset that have not yet been seen.
        The items are selected based on the specified difficulty level, which chance of same-gender items.
        Args:
            dificulty_level (int): The difficulty level which influences the selection of items (1 to 5)

        Returns:
            tuple[DataItem, DataItem]: A tuple containing two DataItem objects representing the selected
            items for the game.
        
        """
        male_items, female_items = self.get_available_items()
        choices = [False] * (5-dificulty_level) + [True] * (dificulty_level-1)
        same_gender = random.choice(choices)
        if same_gender:
            item_list = random.choice([female_items, male_items])
            item1, item2 = random.sample(sorted(item_list), 2)
        else:
            item_list_1 = random.choice([female_items, male_items])
            item_list_2 = female_items if item_list_1 == male_items else male_items
            item1 = random.choice(sorted(item_list_1))
            item2 = random.choice(sorted(item_list_2))
        self.seen.add(item1)
        self.seen.add(item2)
        return item1, item2

    def get_available_items(self):
        """
        Retrieves sets of male and female items that have not yet been seen.
        Returns:
            tuple[set, set]: A tuple containing two sets, the first of unseen male items and the second of unseen female items.
        """
        if len(set(self.female_items) - self.seen) <= 1 or \
        len(set(self.male_items) - self.seen) <= 1:
            self.seen = set()
        female_items = set(self.female_items) - self.seen
        male_items = set(self.male_items) - self.seen
        return male_items, female_items
        
    def getImagesAndVoice(self, dificulty_level):
        """
        Generates pairs of images and a voice clip for the game, based on the specified difficulty level.
        Calculates the cosine similarity between the voice clip and each of the images, to get the model "decision".
        Args:
            difficulty_level (int): The difficulty level which influences the selection of items (1 to 5)

        Yields:
            tuple[str, str, str, torch.Tensor, torch.Tensor]: A tuple containing the file paths to the true image,
            false image, and voice clip, followed by the cosine similarity scores between the voice clip and each of the images.
        """
        # self.model.eval()
        with torch.inference_mode():
            while True:
                item1, item2 = self.get_two_random_items(dificulty_level)
                true_image_file_path = item1.image_path
                false_image_file_path = item2.image_path
                true_voice_file_path = item1.voice_path
                # true_image = self.get_image(true_image_file_path).clone()
                # false_image = self.get_image(false_image_file_path).clone()
                # true_voice = self.get_voice(true_voice_file_path).clone()
                # true_image_prediction = self.model(true_image)
                # false_image_prediction = self.model(false_image)
                true_image_prediction = torch.load(f"{ROOT_DIR}/gui/embeddings/images/{true_image_file_path.split('/')[-1].split('.')[0]}.pth")
                false_image_prediction = torch.load(f"{ROOT_DIR}/gui/embeddings/images/{false_image_file_path.split('/')[-1].split('.')[0]}.pth")
                true_voice = torch.load(f"{ROOT_DIR}/gui/embeddings/audio/{true_voice_file_path.split('/')[-1].split('.')[0]}.pth")
                true_smilarity = torch.nn.functional.cosine_similarity(true_image_prediction, true_voice)
                false_similarity = torch.nn.functional.cosine_similarity(false_image_prediction, true_voice)
                yield true_image_file_path, false_image_file_path, true_voice_file_path, true_smilarity, false_similarity

    
if __name__ == "__main__":
    from tqdm import tqdm

    def check_saved_embeddings(items_list):
        for item in tqdm(items_list, desc="Saving embeddings"):
            image = gui.get_image(item.image_path)
            image_prediction = gui.model(image)
            print(f"image prediction shape: {image_prediction.shape}")
            voice_prediction = gui.get_voice(item.voice_path)
            # compare embeddings to gui/embeddings
            saved_image = torch.load(f"{ROOT_DIR}/gui/embeddings/images/{item.image_path.split('/')[-1].split('.')[0]}.pth")
            print(f"saved image shape: {saved_image.shape}")
            print(f"Checking {item.image_path}")
            # print(saved_image)
            saved_voice = torch.load(f"{ROOT_DIR}/gui/embeddings/audio/{item.voice_path.split('/')[-1].split('.')[0]}.pth")
            print("checking voice")
            # print(saved_voice)
            print(image_prediction.detach().numpy()[0, :10])
            print(saved_image.detach().numpy()[0, :10])
            assert torch.allclose(image_prediction, saved_image, rtol=0.1, atol=0.1), f"Image embeddings do not match for {item.image_path}"
            assert torch.allclose(voice_prediction, saved_voice), f"Voice embeddings do not match for {item.voice_path}"
    
    gui = GuiBackend()
    print("checking male embeddings")
    check_saved_embeddings(gui.male_items)
    print("checking female embeddings")
    check_saved_embeddings(gui.female_items)
    print("Embeddings saved successfully.")
        
