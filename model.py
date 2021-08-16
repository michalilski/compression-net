import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Softmax, Tanh
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.padding import ReflectionPad2d, ZeroPad2d

from config import batch_size

ENCODER_OUTPUT_CHANNELS = 64

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(4,4), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l34 = nn.Sequential(
            nn.ZeroPad2d((2,1,2,1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),
        
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l56 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l78 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l9 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l10 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=ENCODER_OUTPUT_CHANNELS, kernel_size=(5,5), stride=(1,1)),
            nn.Softmax2d()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l34(x) + x
        x = self.l56(x) + x
        x = self.l78(x) + x
        x = self.l9(x)
        x = self.l10(x)
        # print(x.size())
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
        )

        self.l2 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self.l3 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self.l4 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self.l5 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d(2),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=2, stride=2)
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

        
        
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x) + x
        x = self.l3(x) + x
        x = self.l4(x) + x
        x = self.l5(x)
        x = self.l6(x)
        # print(dec.size())
        return x


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l5 = nn.Sequential(
            nn.ZeroPad2d((1,0,1,0)),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Sigmoid(),
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.l10 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        
        self.lin1 = nn.Sequential(
            nn.Linear(8*26*26,2000),
            nn.Sigmoid(),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(2000,500),
            nn.Sigmoid(),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(500,100),
            nn.Sigmoid(),
        )
        self.lin4 = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x):
        y = x['encoded'].to('cuda')
        #print(y.shape)
        y = self.l1(y)
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        y = self.l5(y)
        #print(y.shape)
        x = x['img'].to('cuda')
        #print(x.shape)
        x = torch.cat((x,y),1)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        
        x= x.reshape((x.shape[0],-1))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        return x