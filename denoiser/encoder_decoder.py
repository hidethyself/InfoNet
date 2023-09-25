import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=2, padding=1, dilation=1, norm=False):
        super(DownBlock, self).__init__()
        self.conv_block = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.activation = nn.LeakyReLU(0.2)
        self.batch_normalization = nn.BatchNorm2d(num_features=out_channel)
        self.norm = norm

    def forward(self, x):
        x = self.conv_block(x)
        x = self.activation(x)
        if self.norm:
            x = self.batch_normalization(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(4, 2), stride=2, dilation=1, norm=False, last=False):
        super(UpBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation
        )
        self.activation = nn.Sigmoid() if last else nn.LeakyReLU(0.2)
        self.batch_normalization = nn.BatchNorm2d(num_features=out_channel)
        self.norm = norm

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.activation(x)
        if self.norm:
            x = self.batch_normalization(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down_1 = DownBlock(in_channel=21, out_channel=32)
        self.down_2 = DownBlock(in_channel=32, out_channel=64, kernel_size=(5, 3))
        self.down_3 = DownBlock(in_channel=64, out_channel=128, kernel_size=(5, 3))
        self.down_4 = DownBlock(in_channel=128, out_channel=256, kernel_size=(3, 3))
        self.down_5 = DownBlock(in_channel=256, out_channel=512, kernel_size=(5, 3))

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)
        x = self.down_5(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up_1 = UpBlock(in_channel=512, out_channel=256)
        self.up_2 = UpBlock(in_channel=256, out_channel=128, kernel_size=(2, 2))
        self.up_3 = UpBlock(in_channel=128, out_channel=64)
        self.up_4 = UpBlock(in_channel=64, out_channel=32)
        self.up_5 = UpBlock(in_channel=32, out_channel=21, kernel_size=(2, 2), last=True)

    def forward(self, x):
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)
        x = self.up_4(x)
        x = self.up_5(x)
        return x


class BaseEncoderDecoder(nn.Module):
    def __init__(self):
        super(BaseEncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main():
    x = torch.randn(2, 21, 300, 64)
    encoder_decoder = BaseEncoderDecoder()
    out = encoder_decoder(x)
    print(out.shape)


if __name__ == '__main__':
    main()