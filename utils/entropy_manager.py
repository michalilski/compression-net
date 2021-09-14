from transforms import ImageTransform
from skimage.measure import shannon_entropy
from torch import Tensor
from typing import Dict, List
from dataloader import ImageDataLoader
from tqdm import tqdm

class EntropyManager:
    def __init__(self) -> None:
        self.image_transform = ImageTransform()

    def calculate_image_entropy(self, image: Tensor) -> Dict[str, float]:
        color_entropy = {
            channel : shannon_entropy(value) 
            for channel, value in zip(("r","g","b"), image)
        }
        color_entropy.update(
            self._calculate_image_grayscale_entropy(image)
        )
        return color_entropy
    
    def plot_entropy_distribution(self, image_loader: ImageDataLoader):
        entropies = self._calculate_dataset_entropy(image_loader)
        #TODO plot

    def _calculate_image_grayscale_entropy(self, image: Tensor) -> Dict[str, float]:
        return {"grayscale": shannon_entropy(self.image_transform.grayscale(image))}
    
    def _calculate_dataset_entropy(self, image_loader: ImageDataLoader) -> List[Dict]:
        print("Calculating dataset entropies:")
        entropies = []
        for i, batch in enumerate(tqdm(image_loader)):
            entropies.extend([
                self.calculate_image_entropy(image)
                for image in batch[0]
            ])
        return entropies