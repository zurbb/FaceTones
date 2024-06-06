import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache')
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_train_loader
import coloredlogs, logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit_size", type=int, default=5000, help="Limit size of the dataset")
parser.add_argument("--validation_size", type=int, default=128, help="Validation size of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the dataset")
parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")
args = parser.parse_args()

LIMIT_SIZE = args.limit_size
VALIDATION_SIZE = args.validation_size
BATCH_SIZE = args.batch_size
RUN_NAME = args.run_name
EPOCHS = args.epochs
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

# Define your neural network architecture
class ImageVoiceClassifier(nn.Module):
    def __init__(self, dino=False):
        super().__init__()
        self.dropout = nn.Dropout(0.1)  # Dropout layer
        self.convolutional_layers = nn.Sequential(
            # input 3,128,128
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),  # output 12,64,64
            nn.ReLU(),
            self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # output 3,32,32 
            nn.Conv2d(12, 4, kernel_size=3, stride=1, padding=1), # output 4,32,32
            nn.ReLU(),  # output 1,32,32
            self.dropout,
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),  # output 1,32,32
            nn.ReLU(),  # output 1,32,32
            self.dropout,
            nn.Flatten(),  # output 1,1024
        )
        self.dino_convolution_layers = nn.Sequential(
            #input 1,257,768
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # output 8,129,384
            nn.ReLU(),
            #self.dropout,
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # output 12,65,192
            nn.Conv2d(8, 2, kernel_size=3, stride=2, padding=1),  # output 2,33,96
            nn.ReLU(), 
            #self.dropout,
            nn.Conv2d(2, 1, kernel_size=3, stride=2, padding=1),  # output 1,17,48
            nn.ReLU(), 
            self.dropout,
            nn.Flatten(),  # output 1,816
        )
        embed_dim = 1024 if not dino else 816
        self.multihead = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8) 
        self.final_layer = nn.Linear(1024, 512)  # output 1,768
        self.dino_final_layer = nn.Linear(816, 512)  # output 1,768
        if dino:
            self.convolutional_layers = self.dino_convolution_layers
            self.final_layer = self.dino_final_layer
        
        
    def forward(self, x):
        logits = self.convolutional_layers(x.to(device))
        attn_output, _ = self.multihead(logits.to(device), logits.to(device), logits.to(device), need_weights=False)
        attn_output = self.dropout(attn_output.to(device))  # Apply dropout
        logits = self.final_layer(attn_output.to(device))
        return logits


LOSS = nn.CosineEmbeddingLoss()

# Define your loss function
def cosine_similarity_loss(outputs, voices):
    # TODO: maybe get size from constants and take labels out of the function
    labels = torch.ones(outputs.size(0)).to(outputs.device)
    loss = LOSS(outputs, voices.to(outputs.device), labels)
    return loss

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
# Load and preprocess your "imagesVoices" dataset
# Split it into training and validation sets
def train(train_data_loader, validation_loader, model, loss_fn, optimizer, num_epochs):
    size = len(train_data_loader.dataset)
    # Training loop
    for epoch in range(num_epochs):
        for Batch_number, (images, voices) in enumerate(train_data_loader):
            try:
                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, voices)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if Batch_number%10==0:
                    logger.info(f"batch: {Batch_number+1} done.")
                    logger.info(f"loss: {loss:>7f}")
            except Exception as e:
                logger.error(f"Error in batch {Batch_number+1}: {e}")

        logger.info(f"Epoch: {epoch+1} done.")    
        loss, current = torch.mean(loss), (Batch_number + 1) * len(images)
        logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # Validate your model on the validation set
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for images, voices in validation_loader:
                try:
                    outputs = model(images)
                    val_loss += loss_fn(outputs, voices)
                except Exception as e:
                    logger.error(f"Error in validation batch {num_batches+1}: {e}")
                num_batches += 1
            logger.info(f"Validation Error: {val_loss.item()/num_batches:>7f}")
        save_checkpoint(model, optimizer, epoch, loss, os.path.join(ROOT_DIR, 'trained_models', RUN_NAME,f'checkpoint_{epoch}.pth'))


def main():
    if not os.path.exists(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME)):
        os.mkdir(os.path.join(ROOT_DIR, 'trained_models', RUN_NAME))
    # Create an instance of your network
    model = ImageVoiceClassifier(dino=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for param in model.parameters():
        logger.info(param.size())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    logger.info(f"Model created:\n{model}")
    # Load your dataset
    images_dir = os.path.join(ROOT_DIR, "data/train/images")
    voices_dir = os.path.join(ROOT_DIR, "data/train/audio")
    test_images_dir = os.path.join(ROOT_DIR, "data/test/images")
    test_voices_dir = os.path.join(ROOT_DIR, "data/test/audio")
    logger.info("Creating train data loader")
    train_dataloader = get_train_loader(images_dir, voices_dir, batch_size=BATCH_SIZE, limit_size=LIMIT_SIZE, dino=True)
    logger.info("Creating test data loader")
    validation_dataloader = get_train_loader(test_images_dir, test_voices_dir, batch_size=BATCH_SIZE, limit_size=VALIDATION_SIZE, dino=True)
    logger.info("Starting training")
    train(train_dataloader, validation_dataloader, model, cosine_similarity_loss, optimizer, num_epochs=EPOCHS)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()

