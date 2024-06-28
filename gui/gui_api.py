import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))

from models.eval_lib import load_model_by_checkpoint, load_validation_data
import torch
from enum import Enum
from dataclasses import dataclass
import random
from torchvision import transforms
from PIL import Image
from models.voice_to_vec import VoiceToVec


CHECKPOINT = "2206_postive_punish/checkpoint_4.pth"

root = '/cs/ep/120/Voice-Image-Classifier'

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"

@dataclass
class DataItem:
    gender: Gender
    image_path: str
    voice_path: str

class GuiBackend:
    data_loader = None

    def __init__(self):
        self.male_items =os.path.join(ROOT_DIR, 'gui/females.txt')
        self.female_items = os.path.join(ROOT_DIR, 'gui/females.txt')
        self.model =  load_model_by_checkpoint(CHECKPOINT, hard_checkpoint=True)
        self.male_items = self.make_data_items_list(self.female_items,Gender.MALE)
        self.female_items = self.make_data_items_list(self.male_items,Gender.FEMALE)
        self.voice_embedder = VoiceToVec()

    def make_data_items_list(self,data_source_path:str, gender:Gender) -> list[DataItem]:
        data_items = []
        with open(data_source_path, 'r') as f:
            data_sources = f.readlines()
        for item in data_sources:
            image_path = os.path.join(root, 'data/evaluation/images',item[1] + "_0.jpg")
            voice_path = os.path.join(root,'data/evaluation/audio', item[2] + ".mp3")
            data_item = DataItem(gender=gender, image_path=image_path, voice_path=voice_path)
            data_items.append(data_item)
        return data_items
    
    def get_two_random_items(self, item_list_1: list[DataItem], item_list_2: list[DataItem]):
        item1 = random.choice(item_list_1)
        item2 = random.choice(item_list_2)
        while item1 == item2:
            item2 = random.choice(item_list_2)
        return item1, item2
    
    def get_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        image = Image.open(image_path)
        return transform(image)
    
    
    def get_voice(self, voice_path):
        signal =self.voice_embedder.get_signals(voice_path)
        return self.voice_embedder.get_embedding(signal)
        
        
    def getImagesAndVoice(self):
        self.model.eval()
        with torch.inference_mode():
            while True:
                data_source_1 = random.choice([self.female_items,self.male_items])
                data_source_2 = random.choice([self.female_items,self.male_items])
                item1, item2 = self.get_two_random_items(data_source_1,data_source_2)
                true_image_file_path = item1.image_path
                false_image_file_path = item2.image_path
                true_voice_file_path = item1.voice_path
                true_image = self.get_image(true_image_file_path).clone()
                false_image = self.get_image(false_image_file_path).clone()
                true_voice = self.get_voice(true_voice_file_path)
                true_image_prediction = self.model(true_image)
                false_image_prediction = self.model(false_image)
                true_smilarity = torch.nn.functional.cosine_similarity(true_image_prediction, true_voice)
                false_similarity = torch.nn.functional.cosine_similarity(false_image_prediction, true_voice)
                yield true_image_file_path, false_image_file_path, true_voice_file_path, true_smilarity, false_similarity


            
if __name__ == "__main__":
    gui_backend = GuiBackend()
    for true_image, false_image, true_voice, true_similarity, false_similarity in gui_backend.getImagesAndVoice():
        print(f"true_image: {true_image}")
        print(f"false_image: {false_image}")
        print(f"true_voice: {true_voice}")
        print(f"true_similarity: {true_similarity}")
        print(f"false_similarity: {false_similarity}")
        break