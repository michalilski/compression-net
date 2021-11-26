import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import ImageDataLoader
from model import Discriminator, Encoder, Generator
from settings import TENSORBOARD_LOGS, device


class TensorboardManager:
    def __init__(self):
        if not os.path.exists(TENSORBOARD_LOGS):
            os.makedirs(TENSORBOARD_LOGS)

        self.writer = SummaryWriter(TENSORBOARD_LOGS)

    def present_models(
        self,
        encoder: Encoder,
        generator: Generator,
        discriminator: Discriminator,
    ):
        with torch.no_grad():
            test_loader = ImageDataLoader().test_loader()
            test_batch = next(iter(test_loader))[0].to(device)
            encoded = encoder(test_batch)
            decoded = generator(encoded)
            discriminator_input = {"encoded": encoded, "img": decoded}
        self.writer.add_graph(encoder, test_batch)
        self.writer.add_graph(generator, encoded)
        self.writer.add_graph(discriminator, discriminator_input)
