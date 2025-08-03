import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

# Set seed for reproducibility
torch.manual_seed(0)

# Function to show image
def show_data(data_sample, shape=(28, 28)):
    # Extract image and label
    image, label = data_sample
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.numpy().squeeze()  # Handle tensor
    else:
        image = np.array(image)  # Handle PIL image if no transform

    plt.imshow(image.reshape(shape), cmap='gray')
    plt.title(f'y = {label}')
    plt.show()

# Define paths
directory = ""  # Local directory; leave empty if using direct URL
csv_file = 'index.csv'
csv_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/labs/Week1/data/index.csv"

# Read CSV from URL
data_name = pd.read_csv(csv_path)
print("CSV loaded successfully. First few entries:")
print(data_name.head())

# Custom Dataset Class – renamed to avoid conflict
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)  # Load directly from URL or local path
        self.len = self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Image file path: since CSV contains full URLs or relative paths
        img_name = os.path.join(self.data_dir, self.data_name.iloc[idx, 1])
        #loading img from the path in a way pytorch can process
        image = Image.open(img_name)

        y = self.data_name.iloc[idx, 0]  # label

        if self.transform:
            image = self.transform(image)

        return image, y


# === Test 1: Cropped + Tensor transform ===
croptensor_transform = transforms.Compose([
    transforms.CenterCrop(20),
    transforms.ToTensor()
])

dataset_crop = CustomImageDataset(csv_file=csv_path, data_dir=directory, transform=croptensor_transform)
print("The shape of the first element tensor (after CenterCrop(20) + ToTensor): ", dataset_crop[0][0].shape)

# === Test 2: Flip + Tensor transform ===
fliptensor_transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=1),  # Always flip
    transforms.ToTensor()
])

dataset_flip = CustomImageDataset(csv_file=csv_path, data_dir=directory, transform=fliptensor_transform)

# Show the second sample
show_data(dataset_flip[1])