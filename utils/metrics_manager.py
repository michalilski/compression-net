from torch.utils.data import DataLoader
from dataclasses import dataclass
from model import Encoder, Generator
from config import device, model_path
from utils.entropy_manager import EntropyManager
import torch
from torch.nn import MSELoss
from loss import perceptual_loss
from skimage.metrics import structural_similarity as ssim


@dataclass
class TestEntry:
    entropy: dict
    mse_loss: float
    perceptual_loss: float
    ssim_value: float


class MetricsManager:
    mse = MSELoss()
    entropy_manager = EntropyManager()
    log_size = 50

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader

    def run(self):
        encoder = Encoder().to(device)
        generator = Generator().to(device)
        encoder.load_state_dict(torch.load(model_path + "encoder.pth"))
        generator.load_state_dict(torch.load(model_path + "generator.pth"))
        tested_entries = []

        for data in self.dataloader:

            #TODO remove + tqdm
            # if len(tested_entries) == 100:
            #     return tested_entries

            source_image = data[0].to(device)
            encoded = encoder(source_image)
            generated = generator(encoded)
            if len(tested_entries) % self.log_size == 0:
                print(f"Processed {len(tested_entries)}")

            source_entropy = self.entropy_manager.calculate_image_entropy(source_image[0].cpu())
            mse_loss = self.mse(generated, source_image)

            #TODO refactor in loss.py
            gen_features = perceptual_loss(generated)
            raw_img_features = perceptual_loss(source_image)
            with torch.no_grad():
                img_features = raw_img_features.detach()
            vgg_loss = self.mse(gen_features, img_features)

            ssim_loss = ssim(
                generated[0].permute(1,2,0).cpu().detach().numpy(),
                source_image[0].permute(1,2,0).cpu().detach().numpy(),
                data_range=float(generated[0].max() - generated[0].min()),
                multichannel=True,
            )

            tested_entries.append(
                TestEntry(
                    entropy=source_entropy,
                    mse_loss=float(mse_loss),
                    perceptual_loss=float(vgg_loss),
                    ssim_value=ssim_loss
                )
            )
        return tested_entries


