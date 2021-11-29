import logging
import sys
from typing import Dict, List

from skimage.measure import shannon_entropy
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from transforms import ImageTransform

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class EntropyManager:
    """
    Manager for entropy calculation.
    """
    def __init__(self) -> None:
        self.image_transform = ImageTransform()

    def calculate_image_entropy(self, image: Tensor) -> Dict[str, float]:
        """
        Function calculating entropy for one image.

        :param image: Tensor representing image
        :return: dictionary with calculated entropy for red, green, blue
        channels and grayscale
        """
        color_entropy = {
            channel: shannon_entropy(value.cpu().detach().numpy())
            for channel, value in zip(("r", "g", "b"), image)
        }
        color_entropy.update(self._calculate_image_grayscale_entropy(image))
        return color_entropy

    def _calculate_image_grayscale_entropy(self, image: Tensor) -> Dict[str, float]:
        return {"grayscale": shannon_entropy(self.image_transform.grayscale(image))}

    def calculate_dataset_entropy(self, image_loader: DataLoader) -> List[Dict]:
        """
        Function calculating entropy for given dataset.

        :param image_loader: DataLoader for given data set
        :return: list of dictionaries containing entropy values for each image
        from data set
        """
        logger.info("Calculating dataset entropy:")
        entropy = []
        for _, batch in enumerate(tqdm(image_loader)):
            entropy.extend([self.calculate_image_entropy(image) for image in batch[0]])
        return entropy

    def entropy_to_text(self, data: dict):
        """
        Function returning entropy values as string.
        """
        return """
        Entropy
        Red: {r}
        Green: {g}
        Blue: {b}
        Grayscale: {grayscale}
        """.format_map(
            data
        )
