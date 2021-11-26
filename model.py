import torch
import torch.nn as nn
from torch.nn.modules.padding import ReflectionPad2d


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
        )
        self.l2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
            nn.LeakyReLU(),
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        self.post_l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=4),
            nn.ReflectionPad2d(2),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        x = self.l1(x)
        x = self.post_l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ReflectionPad2d(1),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ReflectionPad2d((1,0,1,0)),
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.9),
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(14112, 2000),
            nn.Sigmoid(),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(2000,200),
            nn.Sigmoid(),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(200,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = x['encoded'].to('cuda')
        y = self.l1(y)
        y = self.l2(y)
        y = self.l3(y)
        x = x['img'].to('cuda')
        x = torch.cat((x, y), 1)
        x = self.l6(x)
        x = self.l7(x)

        x = x.reshape((x.shape[0], -1))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x
