import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Tanh
from torch.nn.modules.padding import ReflectionPad2d

ENCODER_OUTPUT_CHANNELS = 32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=(4,4), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l34 = nn.Sequential(
            nn.ReflectionPad2d((2,1,2,1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),
        
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l56 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l8 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=ENCODER_OUTPUT_CHANNELS, kernel_size=(5,5), stride=(1,1)),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l34(x) + x
        x = self.l56(x) + x
        x = self.l7(x)
        x = self.l8(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=(2,2)),
        )

        self.l34 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),
        
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l56 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )

        self.l78 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1)),
            nn.LeakyReLU()
        )

        self.l9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(2,2)),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=(1,1)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.l1(x)       
        x = self.l34(x) + x       
        x = self.l56(x) + x       
        x = self.l78(x) + x
        x = self.l9(x)
        x = self.l10(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.latent_layer1 = nn.Sequential(
            nn.ConvTranspose2d(ENCODER_OUTPUT_CHANNELS, 12, (3,3), stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.latent_layer2 = nn.Sequential(
            nn.ConvTranspose2d(12, 16, (3,3), stride=1, padding=2, output_padding=0, groups=1, bias=True, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.latent_layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 24, (3,3), stride=2, padding=2, output_padding=1, groups=1, bias=True, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.latent_layer4 = nn.Sequential(
            nn.ConvTranspose2d(24, 36, (5,5), stride=2, padding=0, output_padding=1, groups=1, bias=True, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.latent_layer5 = nn.Sequential(
            nn.ConvTranspose2d(36, 3, (3,3), stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Tanh(),
        )

        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3,stride = 1,padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,stride = 2,padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3,stride = 2,padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,stride = 1,padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3,stride = 1,padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Tanh(),
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(4096,100),
            nn.Sigmoid(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(100,10),
            nn.Sigmoid(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(10,1),
            nn.Sigmoid(),
        )
        
        
    def forward(self, x):
        y = x['encoded'].to('cuda')
        y = self.latent_layer1(y)
        y = self.latent_layer2(y)
        y = self.latent_layer3(y)
        y = self.latent_layer4(y)
        y = self.latent_layer5(y)
#         
        x = x['img'].to('cuda')
#         
        x = torch.cat((x,y),1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
#         
        x = x.reshape((x.shape[0],-1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x