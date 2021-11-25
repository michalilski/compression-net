from torch.utils import data
from torch.utils.data import dataloader
from dataloader import ImageDataLoader
from utils.dataset_scanner import DatasetScanner

def run():
    image_loader = ImageDataLoader().test_loader()
    DatasetScanner(image_loader).scan_dataset()

if __name__ == "__main__":
    run()