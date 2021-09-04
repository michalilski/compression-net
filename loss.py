import torch
from torch import nn
from torchvision.models import vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.model = vgg.vgg16(pretrained=True)
        self.model.to(device)
        self.model.eval()
        self.extracted_layer = "15"

    def forward(self, x):
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name == self.extracted_layer:
                return x


perceptual_loss = PerceptualLoss()
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()


def discriminator_loss(
    discriminator_output, label,
):
    return bce_loss(discriminator_output, label)


def encoder_loss(
    fake_discriminator_output, real_label, generated_images, real_images,
):
    return bce_loss(fake_discriminator_output, real_label) + 2 * l1_loss(
        generated_images, real_images
    )


def generator_final_loss(
    fake_discriminator_output, real_label, generated_images, real_images,
):
    kwargs = locals()

    gen_features = perceptual_loss(generated_images)
    raw_img_features = perceptual_loss(real_images)

    with torch.no_grad():
        img_features = raw_img_features.detach()

    vgg_loss = mse_loss(gen_features, img_features)

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
    epoch, fake_discriminator_output, real_label, generated_images, real_images,
):
    if epoch < 10:
        loss = generator_phases["initial"]
    else:
        loss = generator_phases["final"]

    return loss(fake_discriminator_output, real_label, generated_images, real_images,)
