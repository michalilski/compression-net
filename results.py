import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import adjust_contrast, adjust_saturation

from dataloader import ImageDataLoader
from model import Encoder, Generator
from settings import (
    ENTROPY_SCAN_CHANNELS,
    ENTROPY_SCAN_FILE,
    GENERATED_IMAGES_PATH,
    METRICS_FILE,
    MODEL_PATH,
    device,
)
from transforms import ImageTransform
from utils.dataset_scanner import DatasetScanner
from utils.metrics_manager import Metrics, MetricsManager

TESTED_FILE_PATH = ""


def train_set_entropy():
    """
    Function presenting entropy for red, green, blue channels and gray
    scale. Function presents entropy from training part of data set.
    """
    if not os.path.exists(ENTROPY_SCAN_FILE):
        DatasetScanner(ImageDataLoader().train_loader()).scan_dataset()

    with open(ENTROPY_SCAN_FILE, "r") as file:
        results = file.read()
    data = json.loads(results)

    listed_entropy = {
        key: [entropy[key] for entropy in data["entropy"]]
        for key in ENTROPY_SCAN_CHANNELS
    }

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("Training set entropy distribution")

    ax[0, 0].hist(listed_entropy["r"], bins=100)
    ax[0, 0].set_title("Red channel")

    ax[0, 1].hist(listed_entropy["g"], bins=100)
    ax[0, 1].set_title("Green channel")

    ax[1, 0].hist(listed_entropy["b"], bins=100)
    ax[1, 0].set_title("Blue channel")

    ax[1, 1].hist(listed_entropy["grayscale"], bins=100)
    ax[1, 1].set_title("Grayscale")

    plt.tight_layout()
    plt.show()


def show_metrics():
    """
    Function presenting MSE, Perceptual loss, SSIM, MS-SSIM
    and PSNR metrics calculated on test part of data set.
    """
    if not os.path.exists(METRICS_FILE):
        MetricsManager(ImageDataLoader().test_loader(test_batch=1)).run()

    with open(METRICS_FILE, "r") as file:
        results = file.read()

    results = json.loads(results)
    results = [Metrics(**result) for result in results]

    sorted_results = {
        key: sorted(results, key=lambda x: x.entropy[key])
        for key in ENTROPY_SCAN_CHANNELS
    }
    metrics = ("ssim_value", "ms_ssim_value", "psnr")

    fig, ax = plt.subplots(nrows=len(metrics), ncols=4)
    fig.suptitle("Test set metrics")
    for col_idx, key in enumerate(ENTROPY_SCAN_CHANNELS):
        for row_idx, metric_value in enumerate(metrics):
            ax[row_idx, col_idx].plot(
                [x.entropy[key] for x in sorted_results[key]],
                [getattr(x, metric_value) for x in sorted_results[key]],
            )
            ax[row_idx, col_idx].set_title(f"Channel {key.upper()}: {metric_value}")

    plt.tight_layout()
    plt.show()


def present_visual_effect(source_image, normalized_image, image_transform):
    """
    Function presenting visual effects of compression. Original and
    generated images are presented.

    :param source_image: tensor representation of real image
    :param normalized_image: normalized version of real image
    :param image_transform: transforms manager for data set
    """
    encoder = Encoder().to(device)
    generator = Generator().to(device)

    encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, "encoder.pth")))
    generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, "generator.pth")))

    normalized_image = torch.tensor(np.expand_dims(normalized_image, axis=0))
    normalized_image = normalized_image.to(device)

    encoder.eval()
    generator.eval()
    encoded = encoder(normalized_image)
    output = generator(encoded)
    image_tensor = output[0].cpu()
    output = image_tensor.detach()

    generated_image = image_transform.denormalize(output)
    generated_image = adjust_saturation(generated_image, 1.1)
    generated_image = adjust_contrast(generated_image, 1.2)

    filename = f"generated-{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    orig_filename = f"original-{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    if not os.path.exists(GENERATED_IMAGES_PATH):
        os.mkdir(GENERATED_IMAGES_PATH)

    im = ToPILImage()(generated_image)
    im.save(os.path.join(GENERATED_IMAGES_PATH, filename), "PNG")

    im = ToPILImage()(source_image.permute(2, 0, 1))
    im.save(os.path.join(GENERATED_IMAGES_PATH, orig_filename), "PNG")

    _, grid = plt.subplots(1, 2)
    grid[0].imshow(source_image)
    grid[0].set_title("Real image")

    grid[1].imshow(generated_image.permute(1, 2, 0))
    grid[1].set_title("Generated image")

    plt.show()


def visual_test():
    """
    Function presenting visual effects for specified image or
    random image from test set.
    """
    image_transform = ImageTransform()
    if TESTED_FILE_PATH:
        if not os.path.exists(TESTED_FILE_PATH):
            raise FileNotFoundError(TESTED_FILE_PATH)
        image = Image.open(TESTED_FILE_PATH)
        normalized_image = image_transform.transform(image)
        image = image_transform.denormalize(normalized_image).cpu().permute(1, 2, 0)
        return present_visual_effect(image, normalized_image, image_transform)

    test_loader = ImageDataLoader().test_loader()
    iterator = iter(test_loader)
    raw_batch = iterator.next()
    normalized_image = raw_batch[0][0]
    image = image_transform.denormalize(normalized_image).cpu().permute(1, 2, 0)
    return present_visual_effect(image, normalized_image, image_transform)


actions = {
    "train-set-entropy": train_set_entropy,
    "test-set-metrics": show_metrics,
    "visual-test": visual_test,
}


def main():
    """
    Function parsing arguments for current program and calling chosen
    command - train-set-entropy, test-set-metrics or visual-test.
    """
    global TESTED_FILE_PATH
    parser = argparse.ArgumentParser(description="Compression net results.")
    parser.add_argument(
        "command", choices=actions.keys(), help="Possible commands to run."
    )
    parser.add_argument(
        "path", nargs="?", action="store", type=str, help="Image path for visual test."
    )
    args = parser.parse_args()
    cmd = actions[args.command]
    TESTED_FILE_PATH = args.path
    cmd()


if __name__ == "__main__":
    main()
