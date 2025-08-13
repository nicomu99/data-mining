import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logging, UNet, init_model_weights

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class AugmentWrapper(nn.Module):
    def __init__(
            self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv',
            flip=None, offset=None, scale=None, rotate=None, noise=None
    ):
        super().__init__()

        self.augmentation_model = SegmentationAugmentation(flip, offset, scale, rotate, noise)
        self.segmentation_model = UNetWrapper(
            in_channels=in_channels, n_classes=n_classes, depth=depth, wf=wf,
            padding=padding, batch_norm=batch_norm, up_mode=up_mode
        )

    def forward(self, input_batch, labels, augment=False):
        if augment:
            input_batch, labels = self.augmentation_model(input_batch, labels)
        return self.segmentation_model(input_batch), labels

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        init_model_weights(self)

    def forward(self, input_batch):
        x = self.input_batchnorm(input_batch)
        x = self.unet(x)
        return self.final(x)

class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label):
        transform_t = self._build_2d_transform_matrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)      # Expand to all channels
        transform_t = transform_t.to(input_g.device, torch.float32)

        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)

        # The same transformation is applied to both the inputs and labels
        augmented_input_g = F.grid_sample(
            input_g, affine_t, padding_mode='border', align_corners=False
        )
        augmented_label_g = F.grid_sample(
            label.to(torch.float32), affine_t, padding_mode='border', align_corners=False
        )

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise
            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5   # Convert labels back to boolean

    def _build_2d_transform_matrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            transform_t @= rotation_t

        return transform_t