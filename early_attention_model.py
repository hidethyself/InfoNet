import torch
import torch.nn as nn
from parameters.parameters import seldnet_params
from attention_modules.bam import BAMModule
from models.seld_net import CRNN


class EarlyAttention(nn.Module):
    def __init__(self, data_in, data_out, params):
        super(EarlyAttention, self).__init__()
        self.early_attention = BAMModule()
        self.downstream_task = CRNN(data_in, data_out, params)

    def forward(self, x):
        attention, channel_map, spatial_map = None, None, None
        if self.early_attention is not None:
            x, attention, channel_map, spatial_map = self.early_attention(x)
        doa = self.downstream_task(x)
        return doa, attention, channel_map, spatial_map


def main():
    params = seldnet_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_in, data_out = (2, 21, 300, 64), (2, 1, 3)
    model = EarlyAttention(data_in=data_in, data_out=data_out, params=params).to(device=device)
    x = torch.randn(2, 21, 300, 64).to(device=device)
    doa, attention, channel_map, spatial_map = model(x)
    print(f'DoA shape: {doa.shape}\nChannel map shape: {channel_map.shape}\nSpatial map shape:{spatial_map.shape}')
    print('\n')
    print(model.early_attention)
    print('---------------------------')
    print(model.downstream_task)
    print('---------------------------')
    x, attention, channel_map, spatial_map = model.early_attention(x)
    print(f'Att Feature:{x.shape}\n Channel map shape: {channel_map.shape}\nSpatial map shape:{spatial_map.shape}')
    doa = model.downstream_task(x)
    print(f'DoA shape: {doa.shape}')


if __name__ == '__main__':
    main()