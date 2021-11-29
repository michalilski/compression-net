import torch
from torch import nn
from torchvision.models import vgg

from settings import device


class VGG16Model(torch.nn.Module):
    """
    VGG16 model class

    This class is a wrapper on VGG16 PyTorch implementation.
    Returned result is extracted relu4_3 feature from model.
    """
    def __init__(self):
        super(VGG16Model, self).__init__()
        self.model = vgg.vgg16(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.extracted_layer = "15"

    def forward(self, x):
        """Function feedforwarding VGG16 model

        :param x: vgg16 input
        :return: relu4_3 feature
        """
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name == self.extracted_layer:
                return x


vgg_16_model = VGG16Model()
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


def discriminator_loss(
    discriminator_output,
    label,
):
    """
    BCE loss function for discriminator

    :param discriminator_output:
    :param label:
    :return: calculated loss value
    """
    return bce_loss(discriminator_output, label)


def encoder_loss(
    fake_discriminator_output,
    real_label,
    generated_images,
    real_images,
):
    """
    Loss function for encoder - sum of BCE and L1 loss functions

    :param fake_discriminator_output: discriminator result for generated images
    :param real_label: discriminator label representing original images
    :param generated_images: batch of generated images
    :param real_images: batch of original images
    :return: BCE and L1 loss functions sum
    """
    return bce_loss(fake_discriminator_output, real_label) + 2 * l1_loss(
        generated_images, real_images
    )


def perceptual_loss(generated_images, real_images):
    """
    Perceptual loss function, component of generator loss function

    :param generated_images: batch of generated images
    :param real_images: batch of real images
    :return: MSE between relu4_3 feature results from VGG16 model
    """
    gen_features = vgg_16_model(generated_images)
    raw_img_features = vgg_16_model(real_images)

    with torch.no_grad():
        img_features = raw_img_features.detach()

    return mse_loss(gen_features, img_features)


def generator_final_loss(
    fake_discriminator_output,
    real_label,
    generated_images,
    real_images,
):
    """
    Generator loss function for final phase

    :param fake_discriminator_output: discriminator results for generated images
    :param real_label: discriminator label representing original images
    :param generated_images: batch of generated images
    :param real_images: batch of original images
    :return: sum of encoder loss, perceptual loss and regularization loss
    """
    kwargs = locals()
    vgg_loss = perceptual_loss(generated_images, real_images)
    reg_loss = (
        5
        * 1e-6
        * (
            torch.sum(
                torch.abs(
                    generated_images[:, :, :, :-1] - generated_images[:, :, :, 1:]
                )
            )
            + torch.sum(
                torch.abs(
                    generated_images[:, :, :-1, :] - generated_images[:, :, 1:, :]
                )
            )
        )
    )

    return encoder_loss(**kwargs) + vgg_loss + reg_loss


generator_phases = {
    "initial": encoder_loss,
    "final": generator_final_loss,
}


def generator_loss(
    epoch,
    fake_discriminator_output,
    real_label,
    generated_images,
    real_images,
):
    """
    Generator loss function

    :param epoch: current epoch
    :param fake_discriminator_output: discriminator results for generated images
    :param real_label: discriminator label representing original images
    :param generated_images: batch of generated images
    :param real_images: batch of original images
    :return: generator loss function result based on current epoch
    """
    if epoch == 0:
        loss = generator_phases["initial"]
    else:
        loss = generator_phases["final"]

    return loss(
        fake_discriminator_output,
        real_label,
        generated_images,
        real_images,
    )
