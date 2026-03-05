"""
Progressive GAN Generator and Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    EqualizedConv2d, GeneratorBlock, DiscriminatorBlock,
    PixelNorm, MinibatchStddev
)


class Generator(nn.Module):
    """Progressive GAN Generator"""
    def __init__(self, latent_dim=512, max_resolution=1024, feature_maps=None):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_resolution = max_resolution
        
        if feature_maps is None:
            feature_maps = {
                4: 512, 8: 512, 16: 512, 32: 512,
                64: 256, 128: 128, 256: 64, 512: 32, 1024: 16
            }
        self.feature_maps = feature_maps
        
        # Build progressive blocks
        self.blocks = nn.ModuleDict()
        self.to_rgb = nn.ModuleDict()
        
        # Initial block (4x4)
        self.blocks['4x4'] = GeneratorBlock(latent_dim, feature_maps[4], initial_block=True)
        self.to_rgb['4x4'] = EqualizedConv2d(feature_maps[4], 3, kernel_size=1)
        
        # Progressive blocks
        resolutions = [8, 16, 32, 64, 128, 256, 512, 1024]
        for res in resolutions:
            if res > max_resolution:
                break
            
            prev_res = res // 2
            in_ch = feature_maps[prev_res]
            out_ch = feature_maps[res]
            
            self.blocks[f'{res}x{res}'] = GeneratorBlock(in_ch, out_ch)
            self.to_rgb[f'{res}x{res}'] = EqualizedConv2d(out_ch, 3, kernel_size=1)
        
        self.current_resolution = 4
        self.alpha = 1.0  # For fade-in
    
    def forward(self, z, resolution=None, alpha=None):
        """
        Args:
            z: Latent vector [B, latent_dim, 1, 1]
            resolution: Target resolution (if None, uses current_resolution)
            alpha: Fade-in parameter (if None, uses self.alpha)
        """
        if resolution is None:
            resolution = self.current_resolution
        if alpha is None:
            alpha = self.alpha
        
        # Initial block
        x = self.blocks['4x4'](z)
        
        # Progressive blocks
        res = 4
        x_prev = None
        while res < resolution:
            # Save previous feature map before processing next block
            if res == resolution//2:
                x_prev = x
                        
            next_res = res * 2
            x = self.blocks[f'{next_res}x{next_res}'](x)
            res = next_res
        
        # Handle fade-in during transition
        if alpha < 1.0 and resolution > 4 and x_prev is not None:
            # Get output from previous resolution
            prev_res = resolution // 2
            
            # RGB conversion for current resolution
            rgb_new = self.to_rgb[f'{resolution}x{resolution}'](x)
            
            # RGB conversion for previous resolution, then upsample
            prev_key = f'{prev_res}x{prev_res}'
            if prev_key in self.to_rgb:

                rgb_prev = self.to_rgb[prev_key](x_prev)
                rgb_prev = F.interpolate(rgb_prev, scale_factor=2, mode='nearest')
                
                # Blend based on alpha
                rgb = alpha * rgb_new + (1 - alpha) * rgb_prev
            else:
                rgb = rgb_new
        else:
            rgb = self.to_rgb[f'{resolution}x{resolution}'](x)
        
        return torch.tanh(rgb)
    
    def grow(self, new_resolution):
        """Grow to new resolution"""
        self.current_resolution = new_resolution
        self.alpha = 0.0


class Discriminator(nn.Module):
    """Progressive GAN Discriminator"""
    def __init__(self, max_resolution=1024, feature_maps=None):
        super().__init__()
        
        self.max_resolution = max_resolution
        
        if feature_maps is None:
            feature_maps = {
                4: 512, 8: 512, 16: 512, 32: 512,
                64: 256, 128: 128, 256: 64, 512: 32, 1024: 16
            }
        self.feature_maps = feature_maps
        
        # Build progressive blocks (in reverse order)
        self.blocks = nn.ModuleDict()
        self.from_rgb = nn.ModuleDict()
        
        # Build all resolution blocks
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for i, res in enumerate(resolutions):
            if res > max_resolution:
                break
            
            # From RGB layer
            self.from_rgb[f'{res}x{res}'] = EqualizedConv2d(3, feature_maps[res], kernel_size=1)
            
            # Discriminator block
            if res == 4:
                self.blocks[f'{res}x{res}'] = DiscriminatorBlock(
                    feature_maps[res], feature_maps[res], final_block=True
                )
            else:
                next_res = res // 2
                self.blocks[f'{res}x{res}'] = DiscriminatorBlock(
                    feature_maps[res], feature_maps[next_res]
                )
        
        self.current_resolution = 4
        self.alpha = 1.0
    
    def forward(self, x, resolution=None, alpha=None):
        """
        Args:
            x: Input image [B, 3, H, W]
            resolution: Current resolution (if None, uses current_resolution)
            alpha: Fade-in parameter (if None, uses self.alpha)
        """
        if resolution is None:
            resolution = self.current_resolution
        if alpha is None:
            alpha = self.alpha
        
        # Handle fade-in during transition
        if alpha < 1.0 and resolution > 4:
            # New path: from_rgb at current resolution
            x_new = self.from_rgb[f'{resolution}x{resolution}'](x)
            
            # Old path: downsample then from_rgb
            prev_res = resolution // 2
            x_old = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_old = self.from_rgb[f'{prev_res}x{prev_res}'](x_old)
            
            # Process new path through one block
            x_new = self.blocks[f'{resolution}x{resolution}'](x_new)
            
            # Blend
            x = alpha * x_new + (1 - alpha) * x_old
            
            # Continue from previous resolution
            res = prev_res
        else:
            # Standard path
            x = self.from_rgb[f'{resolution}x{resolution}'](x)
            res = resolution
        
        # Process through remaining blocks
        while res >= 4:
            x = self.blocks[f'{res}x{res}'](x)
            if res == 4:
                break
            res = res // 2
        
        return x
    
    def grow(self, new_resolution):
        """Grow to new resolution"""
        self.current_resolution = new_resolution
        self.alpha = 0.0


class ProgressiveGAN(nn.Module):
    """Combined Progressive GAN model"""
    def __init__(self, latent_dim=512, max_resolution=1024, feature_maps=None):
        super().__init__()
        
        self.generator = Generator(latent_dim, max_resolution, feature_maps)
        self.discriminator = Discriminator(max_resolution, feature_maps)
        
        self.latent_dim = latent_dim
        self.current_resolution = 4
    
    def generate(self, batch_size, device='cuda'):
        """Generate random images"""
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=device)
        return self.generator(z)
    
    def grow(self, new_resolution):
        """Grow both networks to new resolution"""
        self.generator.grow(new_resolution)
        self.discriminator.grow(new_resolution)
        self.current_resolution = new_resolution
    
    def set_alpha(self, alpha):
        """Set fade-in alpha for both networks"""
        self.generator.alpha = alpha
        self.discriminator.alpha = alpha