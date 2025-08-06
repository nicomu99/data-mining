import torch
import torch.nn as nn


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()

        # Tail
        self.tail_batchnorm = nn.BatchNorm3d(1)

        # Backbone
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # Head
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def forward(self, input_batch):
        out = self.tail_batchnorm(input_batch)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = out.view(
            out.size(0),        # Batch size
            -1
        )

        out = self.head_linear(out)

        return out, self.head_softmax(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d)  ):
                with torch.no_grad():
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    with torch.no_grad():
                        nn.init.zeros_(m.bias)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        out = self.conv1(input_batch)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        return self.maxpool(out)