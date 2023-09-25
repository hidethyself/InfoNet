import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x


class CRNN(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = 1
        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = params['nb_cnn2d_filt'] * int(
                np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'], params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1],
                            bias=True)
        )

    def forward(self, x):

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))

        return doa


def test():
    params = {
        "dropout_rate": 0.05,
        "nb_cnn2d_filt": 64,
        "f_pool_size": [4, 4, 2],
        "nb_rnn_layers": 2,
        "nb_fnn_layers": 1,
        "rnn_size": 128,
        "fnn_size": 128,
        "t_pool_size": [5, 5, 2]
    }
    in_feat_shape = (1, 21, 300, 64)
    out_shape = (1, 1, 3)
    model = CRNN(
        in_feat_shape=in_feat_shape,
        out_shape=out_shape,
        params=params
    )
    summary(model, (21, 300, 64))


if __name__ == '__main__':
    test()