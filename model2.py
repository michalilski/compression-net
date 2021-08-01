import torch.nn as nn
from torch.nn.modules.activation import Tanh

ENCODER_OUTPUT_CHANNELS = 32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=ENCODER_OUTPUT_CHANNELS, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        print(x.shape)
        x = self.l2(x)
        print(x.shape)
        x = self.l3(x)
        print(x.shape)
        x = self.l4(x)
        print(x.shape)
        x = self.l5(x)
        print(x.shape)
        x = self.l6(x)
        print(x.shape)
        return nn.Tanh()(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=16, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        print(x.shape)
        x = self.l2(x)
        print(x.shape)
        x = self.l3(x)
        print(x.shape)
        x = self.l4(x)
        print(x.shape)
        return nn.Tanh()(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
    def forward(self, x):
        pass