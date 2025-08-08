import torch
import torch.nn as nn

from utils import init_model_weights


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8, norm='batch'):
        super().__init__()

        # Tail
        if norm == 'fixed':
            self.norm = self.normalize
        elif norm == 'sqrt':
            self.norm = self.square_root_normalization
        else:
            self.norm = nn.BatchNorm3d(1)

        # Backbone
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)

        # Head
        self.head_linear = nn.Linear(1152, 2)
        self.head_softmax = nn.Softmax(dim=1)

        init_model_weights(self)

    @staticmethod
    def normalize(batch, mean=-603, std=396):
        # Mean and std were calculated from the train split of the dataset
        return (batch - mean) / std

    @staticmethod
    def square_root_normalization(batch, clip_min=-1000, clip_max=1000):
        batch_shifted = batch - clip_min
        return torch.sqrt(batch_shifted) / ((clip_max - clip_min) ** 0.5)

    def forward(self, input_batch):
        out = self.norm(input_batch)

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