"""
Custom layers for Progressive GAN
Implements equalized learning rate, pixel normalization, minibatch stddev
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        
        # Initialize with N(0,1)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        # He initialization scale
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = np.sqrt(2.0 / fan_in)
    
    def forward(self, x):
        # Scale weights at runtime
        weight = self.weight * self.scale
        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # He initialization scale
        self.scale = np.sqrt(2.0 / in_features)
    
    def forward(self, x):
        weight = self.weight * self.scale
        return F.linear(x, weight, self.bias)


class PixelNorm(nn.Module):
    """Pixel-wise feature vector normalization"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        # Normalize across channel dimension
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class MinibatchStddev(nn.Module):
    """Minibatch standard deviation layer for discriminator"""
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Calculate group size
        group_size = min(batch_size, self.group_size)
        if batch_size % group_size != 0:
            group_size = batch_size
        
        # Reshape for grouped statistics
        y = x.reshape(group_size, -1, channels, height, width)
        
        # Calculate stddev across group
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + 1e-8)
        
        # Average over channels and pixels
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        
        # Replicate over group and pixels
        y = y.repeat(group_size, 1, height, width)
        
        # Concatenate as new feature map
        return torch.cat([x, y], dim=1)


class BlurLayer(nn.Module):
    """Blur layer for smooth upsampling/downsampling"""
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        # Apply blur to each channel separately
        channels = x.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=channels)


class GeneratorBlock(nn.Module):
    """Generator block with progressive growing support"""
    def __init__(self, in_channels, out_channels, initial_block=False):
        super().__init__()
        self.initial_block = initial_block
        
        if initial_block:
            # First block: 4x4 from latent
            self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=4, padding=3)
            self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            # Subsequent blocks: upsample then convolve
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.pixel_norm1 = PixelNorm()
        self.pixel_norm2 = PixelNorm()
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        if not self.initial_block:
            x = self.upsample(x)
        
        x = self.leaky_relu(self.conv1(x))
        x = self.pixel_norm1(x)
        
        x = self.leaky_relu(self.conv2(x))
        x = self.pixel_norm2(x)
        
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator block with progressive growing support"""
    def __init__(self, in_channels, out_channels, final_block=False):
        super().__init__()
        self.final_block = final_block
        
        if final_block:
            # Final block: includes minibatch stddev
            self.minibatch_stddev = MinibatchStddev()
            self.conv1 = EqualizedConv2d(in_channels + 1, out_channels, kernel_size=3, padding=1)
            self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=4)
            self.fc = EqualizedLinear(out_channels, 1)
        else:
            # Intermediate blocks
            self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        if self.final_block:
            x = self.minibatch_stddev(x)
        
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        
        if self.final_block:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.downsample(x)
        
        return x