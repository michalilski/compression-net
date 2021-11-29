import os

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import ImageDataLoader
from model import Discriminator, Encoder, Generator
from settings import TENSORBOARD_LOGS, device


class TensorboardManager:
    """
    Manager for TensorBoard results
    """
    def __init__(self):
        if not os.path.exists(TENSORBOARD_LOGS):
            os.makedirs(TENSORBOARD_LOGS)

        self.writer = SummaryWriter(TENSORBOARD_LOGS)

    def present_models(
        self,
        encoder: Encoder,
        generator: Generator,
        to_present: str = "encoder",
    ):
        """
        Function presenting models architectures in tensorboard

        :param encoder: encoder model
        :param generator: generator model
        :param to_present: define which model to present
        """
        with torch.no_grad():
            test_loader = ImageDataLoader().test_loader()
            test_batch = next(iter(test_loader))[0].to(device)
            encoded = encoder(test_batch)
        if to_present == "encoder":
            self.writer.add_graph(encoder, test_batch)
        else:
            self.writer.add_graph(generator, encoded)
