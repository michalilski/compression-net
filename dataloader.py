from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from config import batch_size, data_path, train_data_part
from transforms import ImageTransform
import os


class ImageDataLoader:
    def __init__(self):
        img_transform = ImageTransform()
        self.train_set = datasets.ImageFolder(
            root=f"{data_path}/train", transform=img_transform.transform
        )
        self.test_set = datasets.ImageFolder(
            root=f"{data_path}/train", transform=img_transform.transform
        )

    def train_loader(self):
        return DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_loader(self, test_batch=batch_size):
        return DataLoader(
            self.test_set,
            batch_size=test_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
