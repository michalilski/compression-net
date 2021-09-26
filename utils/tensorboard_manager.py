from torch.utils.tensorboard import SummaryWriter
from config import tensorboard_runs, device
from model import Encoder, Discriminator, Generator
from dataloader import ImageDataLoader
import torch

class TensorboardManager:
    writer = SummaryWriter(tensorboard_runs)

    def present_models(self, 
        encoder: Encoder, 
        generator: Generator, 
        discriminator: Discriminator,
    ):
        with torch.no_grad():
            test_loader = ImageDataLoader().test_loader
            test_batch = next(iter(test_loader))[0].to(device)
            encoded = encoder(test_batch)
            decoded = generator(encoded)
            discriminator_input = {
                "encoded": encoded,
                "img": decoded
            }
        self.writer.add_graph(encoder, test_batch)
        self.writer.add_graph(generator, encoded)
        self.writer.add_graph(discriminator, discriminator_input)