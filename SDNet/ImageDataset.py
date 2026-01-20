import torch
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, dataset, augment=False): # [(path/to/image, label), ...]
        self.images = [item[0] for item in dataset]
        self.labels = [item[1] for item in dataset]
        self.augment = augment
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv.imread(self.images[idx], cv.IMREAD_GRAYSCALE)
        label = self.labels[idx]

        # Apply transformations
        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.base_transform(image)

        label = torch.tensor([label], dtype=torch.float32)
        
        return image, label
    