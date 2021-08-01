import torch.nn as nn
from torch.nn.modules.activation import Tanh

ENCODER_OUTPUT_CHANNELS = 32

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.l1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(2,2)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l6 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l8 = nn.Sequential(
            nn.ZeroPad2d((1,2,1,2)),
            nn.Conv2d(in_channels=32, out_channels=ENCODER_OUTPUT_CHANNELS, kernel_size=(2,2), stride=(1,1)),
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
        x = self.l7(x)
        print(x.shape)
        x = self.l8(x)
        print(x.shape)
        return nn.Tanh()(x)

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
            nn.LeakyReLU()
        )

    def forward(self, x):
        print(x.shape)
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
        return nn.Tanh()(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ENCODER_OUTPUT_CHANNELS, out_channels=32, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(2,1), stride=(2,1)),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(1,1), stride=(1,1)),
            nn.LeakyReLU()
        )

    def forward(self, x):
        enc = x['encoded']
        print(enc.shape)
        out = self.l1(enc)
        print(out.shape)
        out = self.l2(out)
        print(out.shape)
        out = self.l3(out)
        print(out.shape)
        out = self.l4(out)
        print(out.shape)
        out = self.l5(out)
        print(out.shape)

#         print(y.shape)
#         x = x['img'].to('cuda')
# #         print(x.shape)
#         x = torch.cat((x,y),1)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
# #         print(x.shape)
#         x = x.reshape((x.shape[0],-1))
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
        return out