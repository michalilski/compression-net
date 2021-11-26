from torch.utils.data import DataLoader
from torchvision import datasets

from settings import BATCH_SIZE, DATA_PATH
from transforms import ImageTransform


class ImageDataLoader:
    def __init__(self):
        img_transform = ImageTransform()
        self.train_set = datasets.ImageFolder(
            root=f"{DATA_PATH}/train", transform=img_transform.transform
        )
        self.test_set = datasets.ImageFolder(
            root=f"{DATA_PATH}/test", transform=img_transform.transform
        )

    def train_loader(self):
        return DataLoader(
            self.train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_loader(self, test_batch=BATCH_SIZE):
        return DataLoader(
            self.test_set,
            batch_size=test_batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
