import torch
from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

class ImagesDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # image = read_image(img_path)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = self._img_file_name_to_label(img_name)
        return image #, label
    
    def _img_file_name_to_label(self, img_name):
        return "".join(img_name.split('.')[0].split('_')[:-1])
# Custom collate function
def custom_collate_fn(batch):
    return torch.stack(batch)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Specify the directory containing the images
    images_dir = "data/train/images"
    
    # Create an instance of the ImagesDataset class
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    train_dataset = ImagesDataset(images_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    # Iterate over DataLoader
    for batch in train_loader:
        print("Batch Tensor Shape:", batch.size()) 
        break
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    second_batch = next(train_iter)
    sample_images = second_batch
    print("Sample images shape:", sample_images.shape)
    # print("Sample labels:", sample_labels)
    img = sample_images[1].squeeze().permute(1, 2, 0)
    plt.imshow(img)
    plt.savefig("sample_image.png")
    plt.show()
    label = "image 1"
    print("Label for the image:", label)