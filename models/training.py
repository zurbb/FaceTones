import os


os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache')
import torch

import torch.nn as nn
import torch.optim as optim
from data_loader import get_train_loader


import coloredlogs, logging


for logger_name in logging.Logger.manager.loggerDict:
    logger2 = logging.getLogger(logger_name)
    logger2.propagate = False

logger = logging.getLogger()

coloredlogs.install(level='NOTSET', logger=logger)



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
    def __init__(self):
        super().__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),  # output 3,64,64
            nn.ReLU(),  # output 3,64,64
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # output 3,32,32
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),  # output 1,32,32
            nn.ReLU(),  # output 1,32,32
            nn.Flatten(),  # output 1,32*32
            nn.Linear(1024, 512)  # output 1,512
        )
        
    def forward(self, x):
        logits = self.convolutional_layers(x)
        return logits



# Define your loss function
def cosine_similarity_loss(outputs, voices):
    cosine_similarity = nn.CosineSimilarity(dim=1)(outputs, voices)
    loss = 1 - cosine_similarity.mean()  # subtract from 1 to make it a minimization problem
    return loss


# Load and preprocess your "imagesVoices" dataset
# Split it into training and validation sets
def train(train_data_loader, validation_loader, model, loss_fn, optimizer, num_epochs=1):
    size = len(train_data_loader.dataset)
    # Training loop
    # for epoch in range(num_epochs):
    for Batch_number, (images, voices) in enumerate(train_data_loader):
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, voices)

        if Batch_number % 20 == 0:
            loss, current = torch.mean(loss), (Batch_number + 1) * len(images)
            logger.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # Validate your model on the validation set
            with torch.no_grad():
                val_loss = 0
                num_batches = 0
                for images, voices in validation_loader:
                    outputs = model(images)
                    val_loss += loss_fn(outputs, voices)
                    num_batches += 1
                logger.debug(f"Validation Error: {val_loss.item()/num_batches:>7f}")
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"batch: {Batch_number} done.") if Batch_number%10==0 else None


if __name__ == '__main__':

    LIMIT_SIZE = 2048
    VALIDATION_SIZE = 1024
    # Create an instance of your network
    model = ImageVoiceClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.debug(f"Model created:\n{model}")
    # Load your dataset
    images_dir = os.path.join(os.getcwd(), "data/test/images")
    voices_dir = os.path.join(os.getcwd(), "data/test/audio")
    test_images_dir = os.path.join(os.getcwd(), "data/test/images")
    test_voices_dir = os.path.join(os.getcwd(), "data/test/audio")
    train_dataloader = get_train_loader(images_dir, voices_dir, batch_size=16, limit_size=LIMIT_SIZE)
    validation_dataloader = get_train_loader(images_dir, voices_dir, batch_size=16, limit_size=VALIDATION_SIZE)
    train(train_dataloader, validation_dataloader, model, cosine_similarity_loss, optimizer, num_epochs=1)

    # Save your trained model
    torch.save(model.state_dict(), 'image_voice_classifier.pth')