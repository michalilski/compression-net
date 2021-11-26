import argparse
import json
import os

import matplotlib.pyplot as plt

from dataloader import ImageDataLoader
from settings import ENTROPY_SCAN_CHANNELS, ENTROPY_SCAN_FILE, METRICS_FILE
from utils.dataset_scanner import DatasetScanner
from utils.metrics_manager import Metrics, MetricsManager


def train_set_entropy():
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
    metrics = ("mse_loss", "perceptual_loss", "ssim_value")

    fig, ax = plt.subplots(nrows=4, ncols=3)
    fig.suptitle("Test set metrics")
    for row_idx, key in enumerate(ENTROPY_SCAN_CHANNELS):
        for col_idx, metric_value in enumerate(metrics):
            ax[row_idx, col_idx].plot(
                [x.entropy[key] for x in sorted_results[key]],
                [getattr(x, metric_value) for x in sorted_results[key]],
            )
            ax[row_idx, col_idx].set_title(f"Channel {key.upper()}: {metric_value}")

    plt.tight_layout()
    plt.show()


actions = {
    "train-set-entropy": train_set_entropy,
    "test-set-metrics": show_metrics,
}


def main():
    parser = argparse.ArgumentParser(description="Compression net results.")
    parser.add_argument("command", choices=actions.keys())
    args = parser.parse_args()
    cmd = actions[args.command]
    cmd()


if __name__ == "__main__":
    main()
