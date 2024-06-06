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



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_size", type=int, default=5000, help="Limit size of the dataset")
    parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
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
                if Batch_number%100==0:
                    WRITER.add_scalar('Loss/train', loss.item(), epoch * size + Batch_number)
                    logger.info(f"batch: {Batch_number+1} done.")
                    logger.info(f"loss: {loss:>7f}")
            except Exception as e:
                logger.error(f"Error in batch {Batch_number+1}: {e}")

        current = (Batch_number + 1) * len(images)
        logger.info(f"Epoch: {epoch+1} done. [{current:>5d}/{size:>5d}]")    


        # Validate the model on the validation set
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for images, voices, _ in validation_loader:
                try:
                    outputs = model(images)
                    val_loss += model.loss(outputs, voices)
                except Exception as e:
                    logger.error(f"Error in validation batch {num_batches+1}: {e}")
                num_batches += 1
            logger.info(f"Validation Error: {val_loss.item()/num_batches:>7f}")
            WRITER.add_scalar('Loss/validation', val_loss.item()/num_batches, epoch * size + Batch_number)
        save_checkpoint(model, optimizer, epoch, loss, os.path.join(ROOT_DIR, 'trained_models', RUN_NAME,f'checkpoint_{epoch}.pth'))


def main():
    if not os.path.exists(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME)):
        os.mkdir(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME))
    # Create an instance of your network
    model = ImageToVoice().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    train_dataloader = get_train_loader(images_dir, voices_dir, batch_size=BATCH_SIZE, limit_size=LIMIT_SIZE)
    logger.info("Creating test data loader")
    validation_dataloader = get_train_loader(test_images_dir, test_voices_dir, batch_size=BATCH_SIZE, limit_size=VALIDATION_SIZE)
    logger.info("Starting training")
    train(train_dataloader, validation_dataloader, model, optimizer, num_epochs=EPOCHS)


if __name__ == '__main__':
    args = parse_args()
    LIMIT_SIZE = args.limit_size
    VALIDATION_SIZE = args.validation_size
    BATCH_SIZE = args.batch_size
    RUN_NAME = args.run_name
    EPOCHS = args.epochs
    WRITER = SummaryWriter(F'runs/{RUN_NAME}')

    torch.multiprocessing.set_start_method('spawn', force=True)
    main()

