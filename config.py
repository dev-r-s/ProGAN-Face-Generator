"""
Progressive GAN Configuration
Optimized for RTX 4000 Ada (20GB VRAM)
"""

import torch
from pathlib import Path

class Config:
    # Hardware settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 8  # For Intel i9-14900
    pin_memory = True
    
    # Dataset paths
    dataset_path = Path("./celeba_hq")  # Update to your path
    output_dir = Path("./outputs")
    checkpoint_dir = Path("./checkpoints")
    
    # Model architecture
    latent_dim = 512
    max_resolution = 1024
    start_resolution = 4
    feature_maps = {
        4: 512,
        8: 512,
        16: 512,
        32: 512,
        64: 256,
        128: 128,
        256: 64,
        512: 32,
        1024: 16
    }
    
    # Training parameters
    learning_rate = 0.001
    adam_betas = (0.0, 0.99)
    adam_eps = 1e-8
    
    # Progressive growing schedule
    images_per_stage = {
        4: 800_000,
        8: 800_000,
        16: 800_000,
        32: 800_000,
        64: 800_000,
        128: 800_000,
        256: 800_000,
        512: 800_000,
        1024: 800_000
    }
    
    # Batch sizes optimized for 20GB VRAM
    batch_sizes = {
        4: 32,
        8: 32,
        16: 32,
        32: 32,
        64: 16,
        128: 16,
        256: 14,
        512: 6,
        1024: 3
    }
    
    # Loss parameters (WGAN-GP)
    lambda_gp = 10.0
    drift = 0.001
    n_critic = 1
    
    # Normalization
    pixel_norm_epsilon = 1e-8
    
    # Mixed precision training
    use_amp = True
    
    # Logging and checkpointing
    log_interval = 100
    save_interval = 5000
    num_sample_images = 64
    
    # EMA for generator
    ema_decay = 0.999

    #validation and monitoring settings
    generate_samples_interval = 5000  # Generate samples every N steps
    checkpoint_interval = 5000  # Save checkpoint every N steps
    keep_last_n_checkpoints = 5  # Keep only last N checkpoints to save disk space

    #learning rate decay
    use_lr_decay = False  # Set to True to enable
    lr_decay_start_step = 100000
    lr_decay_factor = 0.5
    
    @classmethod
    def get_resolution_config(cls, resolution):
        """Get configuration for specific resolution"""
        return {
            'batch_size': cls.batch_sizes.get(resolution, 4),
            'num_channels': cls.feature_maps[resolution],
            'images_per_stage': cls.images_per_stage[resolution]
        }
    
    @classmethod
    def initialize_directories(cls):
        """Create necessary directories"""
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (cls.output_dir / "samples").mkdir(exist_ok=True)
        (cls.output_dir / "logs").mkdir(exist_ok=True)