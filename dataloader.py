from torch.utils.data import DataLoader
from torchvision import datasets

from settings import BATCH_SIZE, DATA_PATH
from transforms import ImageTransform


class ImageDataLoader:
    """
    Data loader for images from data set.
    """
    def __init__(self):
        img_transform = ImageTransform()
        self.train_set = datasets.ImageFolder(
            root=f"{DATA_PATH}/train", transform=img_transform.transform
        )
        self.test_set = datasets.ImageFolder(
            root=f"{DATA_PATH}/test", transform=img_transform.transform
        )

    def train_loader(self):
        """
        Data loader for training part of data set

        :return: training set loader
        """
        return DataLoader(
            self.train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_loader(self, test_batch=BATCH_SIZE):
        """
        Data loader for test part of data set

        :param test_batch: batch size for test data loader
        :return: test set loader
        """
        return DataLoader(
            self.test_set,
            batch_size=test_batch,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
