import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from model import Encoder, Generator, Discriminator
import logging
from config import lr, batch_size, num_epochs, beta1, gi
from dataloader import ImageDataLoader
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Training running on {device}')
encoder = Encoder().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

real_label = 1.0
fake_label = 0.0


encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

dataloader = ImageDataLoader().train_loader

epochs = 5

data_iter = enumerate(tqdm(dataloader))

encoder.train()
generator.train()
discriminator.train()


for epoch in range(epochs):
    for i, data in data_iter:
        
        images = data[0].to(device)
        encoded = encoder(data)
        generated = generator(encoded)
        
        #discriminator training
        discriminator.zero_grad()

        real_input = {
            'img' : images,
            'encoded' : encoded,
        }

        r_label = torch.FloatTensor(np.random.uniform(low=0.855, high=0.999, size=(images.size(0)))).to(device)
        r_discriminator_output = discriminator(real_input).view(-1)
        r_discriminator_error = criterion(r_discriminator_output, r_label)
        r_discriminator_error.backward(retain_graph=True)
        Dx = r_discriminator_output.mean().item()

        fake_input = {
            'img' : generated,
            'encoded' : encoded,
        }

        f_label = torch.FloatTensor(np.random.uniform(low=0.005, high=0.155, size=(images.size(0)))).to(device)
        f_discriminator_output = discriminator(fake_input).view(-1)
        f_discriminator_error = criterion(f_discriminator_output, f_label)
        f_discriminator_error.backward(retain_graph=True)
        DGz = f_discriminator_output.mean().item()

        discriminator_error = r_discriminator_error + f_discriminator_error
        discriminator_optimizer.step()

        #generator training
        generator.zero_grad()
        generator_error = criterion(f_discriminator_output. real_label) + 2*nn.L1Loss(images, generated)
        generated_error.backward(retain_graph=True)
        generator_loss = f_discriminator_output.mean().item()
        generator_optimizer.step()

        #encoder training
        encoder.zero_grad()
        encoder_error = criterion(f_discriminator_output. real_label) + 2*nn.L1Loss(images, generated)
        encoder_error.backward(retain_graph=True)
        encoder_loss = f_discriminator_output.mean().item()
        encoder_optimizer.step()

        
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_E: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     discriminator_error.item(), generated_error.item(), encoder_error.item(), Dx, DGz))


        torch.cuda.empty_cache()