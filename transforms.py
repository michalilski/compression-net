from torchvision import transforms


class ImageTransform:
    """
    Class providing image transforms - initial
    crop and normalization, normalization inversion
    and gray scale conversion.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self):
        self._img_transform = transforms.Compose(
            [
                transforms.CenterCrop((300, 400)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        self._img_denormalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / value for value in self.std]
                ),
                transforms.Normalize(
                    mean=[-value for value in self.mean], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        self._to_grayscale = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),]
        )

    @property
    def transform(self):
        """
        Initial transform for loaded image.
        """
        return self._img_transform

    @property
    def denormalize(self):
        """
        Normalization inversion.
        """
        return self._img_denormalize

    @property
    def grayscale(self):
        """
        RGB to grayscale conversion.
        """
        return self._to_grayscale
