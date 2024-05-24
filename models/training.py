import torch

import torch.nn as nn
import torch.optim as optim
from data_loader import get_train_loader



import coloredlogs, logging
import time

logger = logging.getLogger()
coloredlogs.install()

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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*128*128, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
    def forward(self, x):
        logger.debug(f"Making forward with input shape: {x.shape}")
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



# Define your loss function
def cosine_similarity_loss(outputs, voices):
    logger.debug(f"Making cosine_similarity_loss with outputs shape: {outputs.shape} and voices shape: {voices.shape}")
    cosine_similarity = nn.CosineSimilarity(dim=1)(outputs, voices)
    loss = 1 - cosine_similarity.mean()  # subtract from 1 to make it a minimization problem
    logger.debug(f"Loss: {loss}")
    return loss


# Load and preprocess your "imagesVoices" dataset
# Split it into training and validation sets
def train(train_data_loader, model, loss_fn, optimizer, num_epochs=1):
    size = len(train_data_loader.dataset)
    # Training loop
    # for epoch in range(num_epochs):
    for Batch_number, (images, voices) in enumerate(train_data_loader):
        logger.debug(f"Batch number: {Batch_number}")
        logger.debug(f"Images shape: {images.shape}")
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, voices)

        if Batch_number % 1 == 0:
            loss, current = torch.mean(loss), (Batch_number + 1) * len(images)
            logger.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # Backward and optimize
        logger.debug("Backward and optimize")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug("Backward and optimize done")
        if Batch_number == 10:
             break
    


        # # Validate your model on the validation set
        # with torch.no_grad():
        #     for images, voices in validation_loader:
        #         outputs = model(images)
        #         val_loss = loss_fn(outputs, voices)
        #         logger.debug(f"Validation Error: {val_loss.item():>7f}")


        # Print training and validation metrics


if __name__ == '__main__':
    # Your code here

    # Create an instance of your network
    model = ImageVoiceClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.debug(f"Model created:\n{model}")

    # Load your dataset
    images_dir = "data/train/images"
    voices_dir = "data/train/audio"
    train_dataset = get_train_loader(images_dir, voices_dir, batch_size=2)
    train(train_dataset, model, cosine_similarity_loss, optimizer, num_epochs=1)

    # Save your trained model
    torch.save(model.state_dict(), 'image_voice_classifier.pth')