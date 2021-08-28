import torch
import numpy as np
from torchvision import transforms

from config import image_size, mean, std



class ImageTransform():
    def __init__(self):
        self._img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self._img_denormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    @property
    def transform(self):
        return self._img_transform
    
    @property
    def denormalize(self):
        return self._img_denormalize
    
    # def denormalize(self, tensor):
        # tensor_mean = torch.as_tensor(mean)
        # tensor_std = torch.as_tensor(std)
        # std_inv = 1 / (tensor_std + 1e-7)
        # mean_inv = -tensor_mean * std_inv
        # return transforms.Normalize(mean_inv, std_inv)(tensor)