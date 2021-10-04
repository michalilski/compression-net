from torchvision import transforms

from config import image_size, mean, std


class ImageTransform:
    def __init__(self):
        self._img_transform = transforms.Compose(
            [
                transforms.CenterCrop((300, 400)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self._img_denormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        self._to_grayscale = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
            ]
        )

    @property
    def transform(self):
        return self._img_transform

    @property
    def denormalize(self):
        return self._img_denormalize

    @property
    def grayscale(self):
        return self._to_grayscale
