import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_train_loader
import coloredlogs, logging
import argparse
from model_config_lib import ImageToVoice
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_size", type=int, default=2048, help="Limit size of the dataset")
    parser.add_argument("--validation_size", type=int, default=16, help="Validation size of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--num_workers", type=int, default=11, help="Number of workers for the data loader")
    args = parser.parse_args()
    return args

ROOT_DIR = '/cs/ep/120/Voice-Image-Classifier/'


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

    sim_matrix = cosine_similarity(predicted, voices)
    n = sim_matrix.shape[0]
    positive = []
    negative = []
    for i in range(n):
        row = sim_matrix[i]
        positive.append(row[i])
        non_diag_elements = np.delete(row, i)
        negative.append(np.mean(non_diag_elements))
    return np.mean(positive), np.mean(negative)

def eval_epoch(model, validation_loader,epoch, size, Batch_number):
    with torch.no_grad():
        val_loss = 0
        num_batches = 0
        postive = []
        negative = []
        for images, voices, _ in validation_loader:
            try:
                outputs = model(images)
                p,n = similarity_average(outputs, voices)
                postive.append(p)
                negative.append(n)
                val_loss += model.loss(outputs, voices)
            except Exception as e:
                logger.error(f"Error in validation batch {num_batches+1}: {e}")
            num_batches += 1
        average_p = np.mean(postive)
        average_n = np.mean(negative)

        logger.info(f"Validation Error: {val_loss.item()/num_batches:>7f}")
        logger.info(f"margin {model.loss_func.learnable_param}")
        logger.info(f"Average positive similarity: {average_p}")
        logger.info(f"Average negative similarity: {average_n}")
        WRITER.add_scalar('postive_similarity', average_p, epoch * size + Batch_number)
        WRITER.add_scalar('negative_similarity', average_n, epoch * size + Batch_number)
        WRITER.add_scalar('Loss/validation', val_loss.item()/num_batches, epoch * size + Batch_number)


    
def train(train_data_loader, validation_loader, model, optimizer, num_epochs):
    size = len(train_data_loader.dataset)
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

                if Batch_number%25==0: 
                    WRITER.add_scalar('Loss/train',loss, epoch * size + Batch_number)

                if Batch_number%100==0:
                    logger.info(f"batch: {Batch_number+1} done.")
                    logger.info(f"loss: {loss:>7f}")
                if Batch_number%500==0:
                    # Validate the model on the validation set
                    eval_epoch(model, validation_loader, epoch, size, Batch_number)
        
            except Exception as e:
                logger.error(f"Error in batch {Batch_number+1}: {e}")


    
          
        current = (Batch_number + 1) * len(images)
        logger.info(f"Epoch: {epoch+1} done. [{current:>5d}/{size:>5d}]")    

        save_checkpoint(model, optimizer, epoch, loss, os.path.join(ROOT_DIR, 'trained_models', RUN_NAME,f'checkpoint_{epoch}.pth'))


def main():
    if not os.path.exists(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME)):
        os.mkdir(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME))
    # Create an instance of your network
    model = ImageToVoice().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
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

