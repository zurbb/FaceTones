import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import sys
ROOT_DIR = os.getcwd()
sys.path.append(os.path.abspath(ROOT_DIR))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.data_loader import get_train_loader
from models import eval_sbs
import coloredlogs, logging
import argparse
from models.model_config_lib import ImageToVoice
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

LOG_VALIDATION_STEP = 500
LOG_TRAIN_STEP = 100
TENSOR_BOARD_LOG_STEP = 250
LEARNING_RATE = 0.0001

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_size", type=int, default=2048, help="Limit size of the dataset")
    parser.add_argument("--validation_size", type=int, default=16, help="Validation size of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--num_workers", type=int, default=11, help="Number of workers for the data loader")
    parser.add_argument("--description", type=str, required=True, help="description of exp")

    args = parser.parse_args()
    return args

# ROOT_DIR = '/cs/ep/120/Voice-Image-Classifier/' # for use by Zur and Yedidya, becuase data is in
#  /cs/ep/120/Voice-Image-Classifier/data but repo is cloned elsewhere


for logger_name in logging.Logger.manager.loggerDict:
    logger2 = logging.getLogger(logger_name)
    logger2.propagate = False

logger = logging.getLogger()

coloredlogs.install(logger=logger)


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def similarity_average(predicted, voices) -> tuple[np.float64, np.float64]:
    """
    Calculate the average similarity between predicted and voices.

    Args:
        predicted (array-like): The predicted values.
        voices (array-like): The voices values.

    Returns:
        tuple: A tuple containing the average positive similarity and average negative similarity.
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(voices, torch.Tensor):
        voices = voices.detach().cpu().numpy()
    try: 
        sim_matrix = cosine_similarity(predicted, voices)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {e}")
        return 0.0, 0.0
    n = sim_matrix.shape[0]
    positive = []
    negative = []
    for i in range(n):
        row = sim_matrix[i]
        positive.append(row[i])
        non_diag_elements = np.delete(row, i)
        negative.append(np.mean(non_diag_elements))
    return np.mean(positive), np.mean(negative)

def eval_epoch(model, validation_loader, epoch, size, Batch_number):
    """
    Evaluate the model on the validation dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        validation_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        epoch (int): The current epoch number.
        size (int): The total number of samples in the validation dataset.
        Batch_number (int): The total number of batches in the validation dataset.

    Returns:
        None

    """
    with torch.no_grad():
        val_loss = 0
        num_batches = 0
        postive = []
        negative = []
        for images, voices, _ in validation_loader:
            try:
                outputs = model(images)
                p,n = similarity_average(outputs, voices)
                if p!=0.0 and n!=0.0:
                    postive.append(p)
                    negative.append(n)
                val_loss += model.loss(outputs, voices)
            except Exception as e:
                logger.error(f"Error in validation batch {num_batches+1}: {e}")
            num_batches += 1
        average_p = np.mean(postive) if len(postive) > 0 else 0.0
        average_n = np.mean(negative) if len(negative) > 0 else 0.0
        logger.info(f"margin:{model.loss_func.learnable_param}")
        loss = val_loss.item()/num_batches
        log_and_add_scalar('validation', loss, model, epoch, size, Batch_number, average_p, average_n)

        

def log_and_add_scalar(tag,loss,model,epoch,size,Batch_number,average_p, average_n):
    """
    Log the loss and similarity values and add them to the tensorboard.
    """
    if tag not in ['train', 'validation']:
        raise ValueError("tag should be either 'train' or 'validation'")
    if tag=='validation' or Batch_number%LOG_TRAIN_STEP==0:
        logger.info(f"{tag} loss: {loss}")
        logger.info(f"{tag} positive similarity: {average_p}")
        logger.info(f"{tag} negative similarity: {average_n}")
        logger.info(f"{tag} postive_mean_loss : {model.loss_func.positive_mean_loss}")
        logger.info(f"{tag} entropy loss: {model.loss_func.entropy_loss}")
    step =  epoch * size + Batch_number
    WRITER.add_scalar(f'postive_mean_loss/{tag}', model.loss_func.positive_mean_loss, step)
    WRITER.add_scalar(f'entropy_loss/{tag}', model.loss_func.entropy_loss, step)
    if average_p!=0.0 and average_n!=0.0:
        WRITER.add_scalar(f'postive_similarity/{tag}',average_p,step )
        WRITER.add_scalar(f'negative_similarity/{tag}',average_n, step)
    else: 
        logger.warning("Probably an error in similarity calculation.")
    WRITER.add_scalar(f'Loss/{tag}',loss, step)
    
def train(train_data_loader, validation_loader, model, optimizer, num_epochs):
    """
    Train the model on the training dataset, and log the results as we go.
    Check the model on the validation dataset every N batches and log the results.
    Save the model checkpoint every epoch.
    
    Args:
        train_data_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        validation_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        num_epochs (int): The number of epochs to train the model.
        
    Returns:
        None
    """
    size = len(train_data_loader)
    logger.info(f"Training on {size} batches")
    # Training loop
    for epoch in range(num_epochs):
        for Batch_number, (images, voices, _) in enumerate(train_data_loader):
            try:
                # Forward pass
                outputs = model(images)
                loss = model.loss(outputs, voices)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if Batch_number%TENSOR_BOARD_LOG_STEP==0:
                    p,n = similarity_average(outputs, voices)
                    log_and_add_scalar('train', loss, model, epoch, size, Batch_number, p, n)
                    logger.info(f"done with batch {Batch_number}")
                if Batch_number%LOG_VALIDATION_STEP==0:
                    # Validate the model on the validation set
                    eval_epoch(model, validation_loader, epoch, size, Batch_number)
        
            except Exception as e:
                logger.error(f"Error in batch {Batch_number+1}: {e}", exc_info=True)

        current = (Batch_number + 1) * len(images)
        logger.info(f"Epoch: {epoch+1} done. [{current:>5d}/{len(images) * size:>5d}]")    

        save_checkpoint(model, optimizer, epoch, loss, os.path.join(ROOT_DIR, 'trained_models', RUN_NAME,f'checkpoint_{epoch}.pth'))



def main():
    logger.info(args.description)
    if not os.path.exists(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME)):
        os.mkdir(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME))
    # Create an instance of your network
    model = ImageToVoice().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for param in model.parameters():
        logger.info(param.size())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f"Model created:\n{model}")
    images_dir = os.path.join(ROOT_DIR, "data/train/images")
    voices_dir = os.path.join(ROOT_DIR, "data/train/audio")
    test_images_dir = os.path.join(ROOT_DIR, "data/test/images")
    test_voices_dir = os.path.join(ROOT_DIR, "data/test/audio")
    logger.info("Creating train data loader")
    train_dataloader = get_train_loader(images_dir, voices_dir, batch_size=BATCH_SIZE, limit_size=LIMIT_SIZE, num_workers=NUM_WORKERS)
    logger.info("Creating test data loader")
    validation_dataloader = get_train_loader(test_images_dir, test_voices_dir, batch_size=BATCH_SIZE, limit_size=VALIDATION_SIZE, num_workers=NUM_WORKERS)
    logger.info("Starting training")
    train(train_dataloader, validation_dataloader, model, optimizer, num_epochs=EPOCHS)
    logger.info("Finished training")


if __name__ == '__main__':
    args = parse_args()
    LIMIT_SIZE = args.limit_size
    VALIDATION_SIZE = args.validation_size
    BATCH_SIZE = args.batch_size
    RUN_NAME = args.run_name
    EPOCHS = args.epochs
    WRITER = SummaryWriter(f'runs/{RUN_NAME}')
    NUM_WORKERS = args.num_workers


    torch.multiprocessing.set_start_method('spawn', force=True)
    main()

