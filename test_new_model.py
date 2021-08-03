from dataloader import ImageDataLoader
from transforms import ImageTransform
import matplotlib.pyplot as plt
from model3 import Generator, Encoder, Discriminator
import torch
from config import model_path


def main():
    device = 'cuda'

    dataloader = ImageDataLoader().test_loader
    iterator = iter(dataloader)
    raw_batch = iterator.next()
    data = raw_batch[0]
    data = data.to(device)
    print(data.shape)

    encoder = Encoder().to(device)
    generator = Generator().to(device)
    encoded = encoder(data)
    print(encoded.shape)
    print()
    output = generator(encoded)
    print(output.shape)
    # print()
    # generated = generator(encoded)
    # print()
    # discriminator = Discriminator().to(device)

    # input_value = {
    #     "encoded" : encoded,
    #     "img" : data,
    # }

    # discriminator(input_value).view(-1)

if __name__ == '__main__':
    main()