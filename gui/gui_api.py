import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))

from models.eval_lib import load_model_by_checkpoint, load_validation_data
import torch

CHECKPOINT = "2206_postive_punish/checkpoint_4.pth"


class GuiBackend:
    data_loader = None

    def __init__(self):
        self.data_loader = load_validation_data(limit_size=1000, batch_size=2, use_dino=True)
        self.model =  load_model_by_checkpoint(CHECKPOINT, hard_checkpoint=True)

    def getImagesAndVoice(self):
        self.model.eval()
        with torch.inference_mode():
            for images_and_voices in self.data_loader:
                true_image_file_path = f"{ROOT_DIR}/data/evaluation/images/{images_and_voices[2][0]}"
                false_image_file_path = f"{ROOT_DIR}/data/evaluation/images/{images_and_voices[2][1]}"
                true_voice_file_path = f"{ROOT_DIR}/data/evaluation/audio/{images_and_voices[2][0].replace('_0.jpg', '.mp3')}"
                true_image = images_and_voices[0][0].unsqueeze(0)
                false_image = images_and_voices[0][1].unsqueeze(0)
                true_voice = images_and_voices[1][0].unsqueeze(0)
                true_image_prediction = self.model(true_image)
                false_image_prediction = self.model(false_image)
                true_smilarity = torch.nn.functional.cosine_similarity(true_image_prediction, true_voice)
                false_similarity = torch.nn.functional.cosine_similarity(false_image_prediction, true_voice)
                print("called getImagesAndVoice")
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