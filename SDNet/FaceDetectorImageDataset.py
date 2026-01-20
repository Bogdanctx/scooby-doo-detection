from ImageDataset import ImageDataset
from torchvision import transforms

class FaceDetectorImageDataset(ImageDataset):
    def __init__(self, dataset, augment=False): # [(path/to/image, label), ...]
        super().__init__(dataset=dataset, augment=augment)

        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomRotation(10),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ])
            ], p=0.4),
            transforms.ToTensor(),
        ])
        