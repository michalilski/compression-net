import os
from os import makedirs, path

import numpy as np
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm

from dataloader import ImageDataLoader
from loss import discriminator_loss, encoder_loss, generator_loss
from model import Discriminator, Encoder, Generator
from settings import EPOCHS, MODEL_PATH, beta1, beta2, device, lr
from utils.tensorboard_manager import TensorboardManager

# hardware config
print(f"Training running on {device}")
encoder = Encoder().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# model loading
if (
    path.isfile(MODEL_PATH + "encoder.pth")
    and path.isfile(MODEL_PATH + "generator.pth")
    and path.isfile(MODEL_PATH + "discriminator.pth")
):
    print("Loading models...")
    encoder.load_state_dict(torch.load(MODEL_PATH + "encoder.pth"))
    generator.load_state_dict(torch.load(MODEL_PATH + "generator.pth"))
    discriminator.load_state_dict(torch.load(MODEL_PATH + "discriminator.pth"))

encoder.train()
generator.train()
discriminator.train()


# optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
discriminator_optimizer = optim.Adam(
    discriminator.parameters(), lr=lr, betas=(beta1, beta2)
)


# training images loader
dataloader = ImageDataLoader().train_loader()

# tensorboard
tensorboard_manager = TensorboardManager()
tensorboard_manager.present_models(encoder, generator, discriminator)

# training
for epoch in range(EPOCHS):
    for i, data in enumerate(tqdm(dataloader)):
        images = data[0].to(device)
        encoded = encoder(images)
        generated = generator(encoded)

        real_label = torch.FloatTensor(np.full((images.size(0)), 1.0, dtype=float)).to(
            device
        )
        fake_label = torch.FloatTensor(np.full((images.size(0)), 0.0, dtype=float)).to(
            device
        )

        # discriminator feedforward
        discriminator.zero_grad()

        real_input = {
            "img": images,
            "encoded": encoded,
        }

        r_label = torch.FloatTensor(
            np.random.uniform(low=0.855, high=0.999, size=(images.size(0)))
        ).to(device)
        r_discriminator_output = discriminator(real_input).view(-1)
        r_discriminator_error = discriminator_loss(r_discriminator_output, r_label)
        Dx = r_discriminator_output.mean().item()

        fake_input = {
            "img": generated,
            "encoded": encoded,
        }

        f_label = torch.FloatTensor(
            np.random.uniform(low=0.005, high=0.155, size=(images.size(0)))
        ).to(device)
        f_discriminator_output = discriminator(fake_input).view(-1)
        f_discriminator_error = discriminator_loss(f_discriminator_output, f_label)
        DGz = f_discriminator_output.mean().item()

        discriminator_error = r_discriminator_error + f_discriminator_error

        # generator feedforward
        generator.zero_grad()
        generated_error = generator_loss(
            epoch, f_discriminator_output, real_label, generated, images
        )

        # encoder feedforward
        encoder.zero_grad()
        encoder_error = encoder_loss(
            f_discriminator_output, real_label, generated, images
        )

        # back propagation
        discriminator_error.backward(retain_graph=True)
        generated_error.backward(retain_graph=True)
        encoder_error.backward(retain_graph=True)

        # optimization
        discriminator_optimizer.step()
        generator_optimizer.step()
        encoder_optimizer.step()

        # status
        if i % 100 == 0:
            tensorboard_manager.writer.add_scalar(
                "Encoder Loss", encoder_error.item(), i
            )
            tensorboard_manager.writer.add_scalar(
                "Generator Loss", generated_error.item(), i
            )
            tensorboard_manager.writer.add_scalar(
                "Discriminator Loss", discriminator_error.item(), i
            )
            print(
                "[%d/%d][%d/%d]"
                "\tDiscriminator Loss: %.4f"
                "\tGenerator Loss: %.4f"
                "\tEncoder Loss: %.4f"
                % (
                    epoch + 1,
                    EPOCHS,
                    i,
                    len(dataloader),
                    discriminator_error.item(),
                    generated_error.item(),
                    encoder_error.item(),
                )
            )

        torch.cuda.empty_cache()

    print("Saving model...")
    if not path.exists(MODEL_PATH):
        makedirs(MODEL_PATH)
    torch.save(encoder.state_dict(), MODEL_PATH + "encoder.pth")
    torch.save(generator.state_dict(), MODEL_PATH + "generator.pth")
    torch.save(discriminator.state_dict(), MODEL_PATH + "discriminator.pth")
