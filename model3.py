import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Tanh

ENCODER_OUTPUT_CHANNELS = 32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), stride=(3,3)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(5,5), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(2,2)),
            nn.LeakyReLU()
        )
        self.l8 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=ENCODER_OUTPUT_CHANNELS, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.Tanh()
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
        x = self.l7(x)
        print(x.shape)
        x = self.l8(x)
        print(x.shape)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l8 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4,4), stride=(1,1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=32, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(1,1), stride=(1,1)),
            nn.Tanh()
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=1),
            nn.LeakyReLU()
        )
        self.l8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=(4,4)),
            nn.LeakyReLU()
        )

        self.l11 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.Sigmoid()
        )

        self.l12 = nn.Sequential(
            nn.Linear(in_features=256, out_features=32),
            nn.Sigmoid()
        )

        self.l13 = nn.Sequential(
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = x['encoded']
        out = self.l1(enc)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)

        out = torch.cat((x['img'],out), 1)


        out = self.l6(out)
        out = self.l7(out)
        out = self.l8(out)
        out = self.l9(out)
        out = self.l10(out)


        out = out.view(-1, 2048)
        out = self.l11(out)
        out = self.l12(out)
        out = self.l13(out)

        return out