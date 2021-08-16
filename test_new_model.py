import matplotlib.pyplot as plt
import torch

from config import model_path
from dataloader import ImageDataLoader
from model import Discriminator, Encoder, Generator
from transforms import ImageTransform


def main():
    device = 'cuda'

    dataloader = ImageDataLoader().test_loader
    iterator = iter(dataloader)
    raw_batch = iterator.next()
    data = raw_batch[0]
    data = data.to(device)

    encoder = Encoder().to(device)
    generator = Generator().to(device)
    encoded = encoder(data)
    output = generator(encoded)
    discriminator = Discriminator().to(device)

    input_value = {
        "encoded" : encoded,
        "img" : data,
    }

    discriminator(input_value).view(-1)

if __name__ == '__main__':
    main()