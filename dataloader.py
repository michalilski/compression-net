from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from config import batch_size, data_path, train_data_part
from transforms import ImageTransform


class ImageDataLoader():
    def __init__(self):
        img_transform = ImageTransform()
        dataset = datasets.ImageFolder(root=data_path, transform=img_transform.transform)
        div = int(len(dataset)*train_data_part)
        train_set, test_set = random_split(dataset, [div, len(dataset)-div])
        self._train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self._test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    @property
    def train_loader(self):
        return self._train_dataloader
    
    @property
    def test_loader(self):
        return self._test_dataloader
        
