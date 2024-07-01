from models.model_config_lib import ImageToVoice
from models.checkpoint_model import CheckPointImageToVoice
import torch
from models.training import ROOT_DIR, device
import os
from models.data_loader import get_train_loader
import coloredlogs, logging

logger = logging.getLogger()
coloredlogs.install()


def load_model_by_checkpoint(checkpoint_name:str, hard_checkpoint=False)->ImageToVoice:
    logger.info(f"geting model {checkpoint_name}")
    if hard_checkpoint:
        model = CheckPointImageToVoice()
    else:
        model = ImageToVoice()
    checkpoint = torch.load(os.path.join(ROOT_DIR,'trained_models',checkpoint_name), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"loaded model")
    return model




def load_validation_data(limit_size:int, batch_size:int, use_dino:bool, num_workers:int =2)->torch.utils.data.DataLoader:
    #TODO change to the correct path. talk with @yedidya
    test_images_dir = os.path.join(ROOT_DIR, "data/test/images")
    test_voices_dir = os.path.join(ROOT_DIR, "data/test/audio")
    # test_images_dir = os.path.join(ROOT_DIR, "data/yedidya_tal/images")
    # test_voices_dir = os.path.join(ROOT_DIR, "data/yedidya_tal/audio")
    validation_data = get_train_loader(num_workers=num_workers,images_dir=test_images_dir, shuffle=False, voices_dir=test_voices_dir, batch_size=batch_size, limit_size=limit_size, dino=use_dino)
    return validation_data

def cosine_similarity(predicted, true):
    return torch.nn.functional.cosine_similarity(predicted, true)
