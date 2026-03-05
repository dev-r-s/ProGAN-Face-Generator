# inference.py
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from model import Generator


class ProgressiveGANInference:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(
            device if device is not None else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Read config safely
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            self.latent_dim = cfg.get("latent_dim", 512)
            self.max_resolution = cfg.get("max_resolution", 1024)
            feature_maps = cfg.get("feature_maps", None)
        else:
            self.latent_dim = 512
            self.max_resolution = 1024
            feature_maps = None

        self.generator = Generator(
            latent_dim=self.latent_dim,
            max_resolution=self.max_resolution,
            feature_maps=feature_maps
        ).to(self.device)

        # Load weights
        if "g_ema_state" in checkpoint:
            self.generator.load_state_dict(checkpoint["g_ema_state"])
        elif "generator_state" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state"])
        else:
            raise RuntimeError("Checkpoint missing generator weights")

        self.generator.eval()
        self.generator.alpha = 1.0

        self.current_resolution = checkpoint.get(
            "current_resolution", self.max_resolution
        )
        self.generator.current_resolution = self.current_resolution

    @torch.no_grad()
    def generate(self, num_images=1, seed=None, truncation=1.0):
        if seed is not None:
            torch.manual_seed(seed)

        z = torch.randn(
            num_images, self.latent_dim, 1, 1, device=self.device
        ) * truncation

        return self.generator(
            z,
            resolution=self.current_resolution,
            alpha=1.0
        )

    @torch.no_grad()
    def interpolate(self, start_seed, end_seed, num_frames=8, truncation=1.0):
        torch.manual_seed(start_seed)
        z1 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        torch.manual_seed(end_seed)
        z2 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)

        z1 *= truncation
        z2 *= truncation

        frames = []
        for alpha in torch.linspace(0, 1, num_frames):
            z = self._slerp(z1, z2, alpha)
            img = self.generator(z, self.current_resolution, 1.0)
            frames.append(img[0])

        return frames

    @torch.no_grad()
    def generate_from_latent(self, z):
        if z.dim() == 1:
            z = z.view(1, -1, 1, 1)
        return self.generator(
            z.to(self.device),
            self.current_resolution,
            1.0
        )

    @staticmethod
    def _slerp(z1, z2, alpha):
        z1_n = F.normalize(z1, dim=1)
        z2_n = F.normalize(z2, dim=1)
        dot = torch.clamp((z1_n * z2_n).sum(1, keepdim=True), -1, 1)
        theta = torch.acos(dot)
        sin_t = torch.sin(theta)
        return (
            torch.sin((1 - alpha) * theta) / sin_t * z1 +
            torch.sin(alpha * theta) / sin_t * z2
        )
