"""
models.py - Generator and Discriminator networks for liver CT segmentation.

The Generator is an encoder-decoder CNN that predicts segmentation masks.
The Discriminator is a 3-layer convolutional critic that also serves as
a feature extractor for the perceptual loss.

Usage:
    from models import Generator, Discriminator
    generator = Generator(input_channels=1)
    discriminator = Discriminator(input_channels=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for CT image segmentation.

    Encoder-decoder architecture with 4 strided downsampling layers,
    3 same-resolution bottleneck layers, and 4 nearest-neighbor upsampling stages.

    Parameters
    ----------
    input_channels : int
        Number of input channels (1 for grayscale CT).
    """

    def __init__(self, input_channels):
        super(Generator, self).__init__()

        # Encoder (downsampling)
        self.layer1 = self._conv_block(input_channels, 64, 4, 2, 1)
        self.layer2 = self._conv_block(64, 128, 4, 2, 1, batch_norm=True)
        self.layer3 = self._conv_block(128, 256, 4, 2, 1, batch_norm=True)
        self.layer4 = self._conv_block(256, 512, 4, 2, 1, batch_norm=True)

        # Bottleneck
        self.layer5 = self._conv_block(512, 256, 3, 1, 1, batch_norm=True)
        self.layer6 = self._conv_block(256, 128, 3, 1, 1, batch_norm=True)
        self.layer7 = self._conv_block(128, 64, 3, 1, 1, batch_norm=True)

        # Output
        self.layer8 = self._conv_block(64, 1, 3, 1, 1)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=False):
        """
        Convolutional block: Conv2d + ReLU + optional BatchNorm.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the convolutional kernel.
        stride : int
            Stride for the convolution.
        padding : int
            Padding for the convolution.
        batch_norm : bool, optional
            Whether to include batch normalization. Default is False.

        Returns
        -------
        nn.Sequential
            The convolutional block.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the generator.

        Parameters
        ----------
        x : torch.Tensor
            Input CT image tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Predicted segmentation mask of shape (B, 1, H, W).
        """
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Bottleneck + Decoder
        x5 = self.layer5(x4)
        x5 = F.interpolate(x5, scale_factor=2, mode='nearest')

        x6 = self.layer6(x5)
        x6 = F.interpolate(x6, scale_factor=2, mode='nearest')

        x7 = self.layer7(x6)
        x7 = F.interpolate(x7, scale_factor=2, mode='nearest')

        x8 = self.layer8(x7)
        x8 = F.interpolate(x8, scale_factor=2, mode='nearest')

        return x8


class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training.

    3-layer convolutional network that also serves as a hierarchical
    feature extractor for the perceptual loss. Features are extracted
    with adaptive average pooling to 8x8 to reduce memory and compute.

    Parameters
    ----------
    input_channels : int
        Number of input channels (1 for grayscale).
    """

    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: input -> 128 channels
            nn.Conv2d(input_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Layer 2: 128 -> 256 channels
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # Layer 3: 256 -> 1 channel
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Standard forward pass through the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Discriminator output.
        """
        return self.model(x)

    def forward_all_layers(self, x):
        """
        Extract features from all 3 layers in a single forward pass.

        Features are pooled to 8x8 spatial resolution to reduce memory
        usage and backward pass cost.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        list[torch.Tensor]
            List of pooled feature maps from each layer.
        """
        features = []
        for i in range(3):
            x = self.model[i * 3:i * 3 + 3](x)
            pooled = F.adaptive_avg_pool2d(x, 8)
            features.append(pooled)
        return features
