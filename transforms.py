from config import mean, std, image_size
from torchvision import transforms
import torch

class ImageTransform():
    def __init__(self):
        self._img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    @property
    def transform(self):
        return self._img_transform
    
    def denormalize(self, tensor):
        tensor_mean = torch.as_tensor(mean)
        tensor_std = torch.as_tensor(std)
        std_inv = 1 / (tensor_std + 1e-7)
        mean_inv = -tensor_mean * std_inv
        return transforms.Normalize(mean_inv, std_inv)(tensor)