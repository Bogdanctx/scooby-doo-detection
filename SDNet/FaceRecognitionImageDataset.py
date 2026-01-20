from ImageDataset import ImageDataset
from torchvision import transforms
from Parameters import Parameters

class FaceRecognitionImageDataset(ImageDataset):
    def __init__(self, dataset, augment=False): # [(path/to/image, label), ...]
        super().__init__(dataset=dataset, augment=augment)

        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=15, 
                translate=(0.1, 0.1), 
                scale=(0.85, 1.15)
            ),
            transforms.ColorJitter(brightness=0.3, contrast=0.3), 
            transforms.Resize((Parameters.WINDOW_SIZE, Parameters.WINDOW_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        