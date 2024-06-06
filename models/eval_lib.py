from training import ImageVoiceClassifier
import torch
from training import ROOT_DIR
import os
from data_loader import get_train_loader
import coloredlogs, logging

logger = logging.getLogger()
coloredlogs.install()


def load_model_by_checkpoint(checkpoint_name:str)->ImageVoiceClassifier:
<<<<<<< HEAD
    logger.info(f"geting model {checkpoint_name}")
    model = ImageVoiceClassifier()
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR,checkpoint_name)))
    logger.info(f"loaded model")
=======
    model = ImageVoiceClassifier(dino=True)
    checkpoint = torch.load(os.path.join(ROOT_DIR,'trained_models',checkpoint_name), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
>>>>>>> 3f83e566866bbbdf9a25d0e76b642e9030a7e976
    return model

def load_validation_data(limit_size:int, batch_size:int, use_dino:bool)->torch.utils.data.DataLoader:
    test_images_dir = os.path.join(ROOT_DIR, "data/test/images")
    test_voices_dir = os.path.join(ROOT_DIR, "data/test/audio")
    validation_data = get_train_loader(images_dir=test_images_dir, voices_dir=test_voices_dir, batch_size=batch_size, limit_size=limit_size, dino=use_dino)
    return validation_data

def cosine_similarity_loss(predicted, true):
    return torch.nn.functional.cosine_similarity(predicted, true)
