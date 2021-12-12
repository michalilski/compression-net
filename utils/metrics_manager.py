import json
import os
from dataclasses import asdict, dataclass

import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import perceptual_loss
from model import Encoder, Generator
from settings import METRICS_FILE, MODEL_PATH, device
from utils.entropy_manager import EntropyManager
from pytorch_msssim import ms_ssim


@dataclass
class Metrics:
    entropy: dict
    mse_loss: float
    perceptual_loss: float
    ssim_value: float
    ms_ssim_value: float
    psnr: float


class MetricsManager:
    """
    Manager for metrics calculation.
    """
    mse = MSELoss()
    entropy_manager = EntropyManager()
    log_size = 50

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def results_to_file(self, metrics: list):
        """
        Function saving calculated metrics results to METRIC_FILE directory.
        """
        json_metrics = [asdict(metric) for metric in metrics]
        if not os.path.exists(os.path.dirname(METRICS_FILE)):
            os.makedirs(os.path.dirname(METRICS_FILE))
        with open(METRICS_FILE, "w+") as file:
            file.write(json.dumps(str(json_metrics)))

    def run(self):
        """
        Function running metrics calculation process.

        MSE, Perceptual Loss and SSIM are calculated on data
        from given dataloader.
        """
        encoder = Encoder().to(device)
        generator = Generator().to(device)
        encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, "encoder.pth")))
        generator.load_state_dict(torch.load(os.path.join(MODEL_PATH, "generator.pth")))
        tested_entries = []

        for i, data in enumerate(tqdm(self.dataloader)):

            source_image = data[0].to(device)
            encoded = encoder(source_image)
            generated = generator(encoded)

            source_entropy = self.entropy_manager.calculate_image_entropy(
                source_image[0].cpu()
            )
            mse_loss = self.mse(generated, source_image)
            vgg_loss = perceptual_loss(generated, source_image)

            ssim_value = ssim(
                generated[0].permute(1, 2, 0).cpu().detach().numpy(),
                source_image[0].permute(1, 2, 0).cpu().detach().numpy(),
                data_range=float(generated[0].max() - generated[0].min()),
                multichannel=True,
            )

            ms_ssim_value = ms_ssim(
                generated,
                source_image,
                data_range=float(generated[0].max() - generated[0].min())
            ).item()

            psnr_value = psnr(
                generated[0].permute(1, 2, 0).cpu().detach().numpy(),
                source_image[0].permute(1, 2, 0).cpu().detach().numpy(),
                data_range=float(generated[0].max() - generated[0].min())
            )

            tested_entries.append(
                Metrics(
                    entropy=source_entropy,
                    mse_loss=float(mse_loss),
                    perceptual_loss=float(vgg_loss),
                    ssim_value=ssim_value,
                    ms_ssim_value=ms_ssim_value,
                    psnr=psnr_value,
                )
            )

        self.results_to_file(tested_entries)
        return tested_entries
