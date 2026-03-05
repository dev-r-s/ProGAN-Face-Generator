"""
Training script for Progressive GAN
Implements WGAN-GP loss with progressive growing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from config import Config
from model import ProgressiveGAN
from dataset import create_dataloader, InfiniteDataLoader, save_image_grid


class ProgressiveGANTrainer:
    """Trainer for Progressive GAN"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        print("Initializing Progressive GAN...")
        self.model = ProgressiveGAN(
            latent_dim=config.latent_dim,
            max_resolution=config.max_resolution,
            feature_maps=config.feature_maps
        ).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_eps
        )
        
        self.d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config.learning_rate,
            betas=config.adam_betas,
            eps=config.adam_eps
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=config.use_amp)
        
        # EMA for generator
        self.g_ema = self._create_ema_model()
        
        # Training state
        self.current_resolution = config.start_resolution
        self.global_step = 0
        self.current_stage_steps = 0
        self.is_transition = False
        
        # Initialize directories
        config.initialize_directories()
        
        print(f"Model initialized on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.model.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.model.discriminator.parameters()):,}")
    
    def _create_ema_model(self):
        """Create EMA version of generator"""
        from copy import deepcopy
        g_ema = deepcopy(self.model.generator).eval()
        for param in g_ema.parameters():
            param.requires_grad_(False)
        return g_ema
    
    def _update_ema(self):
        """Update EMA generator"""
        with torch.no_grad():
            for p_ema, p in zip(self.g_ema.parameters(), self.model.generator.parameters()):
                p_ema.copy_(p.lerp(p_ema, self.config.ema_decay))
    
    def compute_gradient_penalty(self, real_images, fake_images):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_images.size(0)
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        
        # Get discriminator output
        d_interpolates = self.model.discriminator(
            interpolates,
            resolution=self.current_resolution,
            alpha=self.model.generator.alpha
        )
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(self, real_images, latent_vectors):
        """Train discriminator for one step"""
        self.d_optimizer.zero_grad()
        
        with autocast(device_type='cuda',enabled=self.config.use_amp):
            # Generate fake images
            with torch.no_grad():
                fake_images = self.model.generator(
                    latent_vectors,
                    resolution=self.current_resolution,
                    alpha=self.model.generator.alpha
                )
            
            # Discriminator outputs
            real_pred = self.model.discriminator(
                real_images,
                resolution=self.current_resolution,
                alpha=self.model.discriminator.alpha
            )
            
            fake_pred = self.model.discriminator(
                fake_images.detach(),
                resolution=self.current_resolution,
                alpha=self.model.discriminator.alpha
            )
            
            # WGAN loss
            d_loss = fake_pred.mean() - real_pred.mean()
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(real_images, fake_images.detach())
            
            # Drift loss (keep discriminator outputs near zero)
            drift = (real_pred ** 2).mean()
            
            # Total discriminator loss
            d_loss_total = d_loss + self.config.lambda_gp * gp + self.config.drift * drift
        
        # Backward pass
        self.scaler.scale(d_loss_total).backward()
        self.scaler.step(self.d_optimizer)
        self.scaler.update()
        
        return {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'real_score': real_pred.mean().item(),
            'fake_score': fake_pred.mean().item()
        }
    
    def train_generator(self, latent_vectors):
        """Train generator for one step"""
        self.g_optimizer.zero_grad()
        
        with autocast(device_type='cuda',enabled=self.config.use_amp):
            # Generate fake images
            fake_images = self.model.generator(
                latent_vectors,
                resolution=self.current_resolution,
                alpha=self.model.generator.alpha
            )
            
            # Get discriminator prediction
            fake_pred = self.model.discriminator(
                fake_images,
                resolution=self.current_resolution,
                alpha=self.model.discriminator.alpha
            )
            
            # Generator loss (maximize discriminator output)
            g_loss = -fake_pred.mean()
        
        # Backward pass
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optimizer)
        self.scaler.update()
        
        # Update EMA
        self._update_ema()
        
        return {'g_loss': g_loss.item()}
    
    def train_step(self, real_images):
        """Single training step"""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # Update alpha for fade-in BEFORE forward pass
        if self.is_transition:
            res_config = self.config.get_resolution_config(self.current_resolution)
            fade_in_steps = res_config['images_per_stage'] // res_config['batch_size']

            alpha = min(1.0, self.current_stage_steps / fade_in_steps)
            self.model.set_alpha(alpha)

        # Sample latent vectors
        latent_vectors = torch.randn(
            batch_size, self.config.latent_dim, 1, 1,
            device=self.device
        )
        
        # Train discriminator
        d_metrics = self.train_discriminator(real_images, latent_vectors)
        
        # Train generator
        g_metrics = self.train_generator(latent_vectors)
        
        self.global_step += 1
        self.current_stage_steps += 1
        
        return {**d_metrics, **g_metrics}
    
    def save_checkpoint(self, filename=None):
        """Save training checkpoint"""
        if filename is None:
            filename = f"checkpoint_res{self.current_resolution}_step{self.global_step}.pt"
        
        checkpoint_path = self.config.checkpoint_dir / filename

        # Extract only serializable config values
        config_dict = {
            'latent_dim': self.config.latent_dim,
            'max_resolution': self.config.max_resolution,
            'feature_maps': self.config.feature_maps,
            'start_resolution': self.config.start_resolution,
            'learning_rate': self.config.learning_rate,
            'batch_sizes': self.config.batch_sizes,
            'images_per_stage': self.config.images_per_stage
        }
        torch.save({
            'global_step': self.global_step,
            'current_resolution': self.current_resolution,
            'current_stage_steps': self.current_stage_steps,
            'is_transition': self.is_transition,
            'generator_state': self.model.generator.state_dict(),
            'discriminator_state': self.model.discriminator.state_dict(),
            'g_ema_state': self.g_ema.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'config': config_dict
        }, checkpoint_path)

        #Cleanup old checkpoints
        #self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints to save disk space"""
        checkpoint_dir = self.config.checkpoint_dir

        # Get all checkpoints for current resolution
        checkpoints = sorted(
            checkpoint_dir.glob(f"checkpoint_res{self.current_resolution}_*.pt"),
            key=lambda x: x.stat().st_mtime
        )

        # Keep only last N checkpoints
        keep_n = getattr(self.config, 'keep_last_n_checkpoints', 5)
        if len(checkpoints) > keep_n:
            for old_checkpoint in checkpoints[:-keep_n]:
                old_checkpoint.unlink()
                print(f"Deleted old checkpoint: {old_checkpoint.name}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.current_resolution = checkpoint['current_resolution']
        self.current_stage_steps = checkpoint.get('current_stage_steps', 0)
        self.is_transition = checkpoint.get('is_transition', False)

        self.model.generator.load_state_dict(checkpoint['generator_state'], strict=False)
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state'], strict=False)
        self.g_ema.load_state_dict(checkpoint['g_ema_state'], strict=False)
        
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])

        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])
        
        self.model.generator.current_resolution = self.current_resolution
        self.model.discriminator.current_resolution = self.current_resolution
        self.g_ema.current_resolution = self.current_resolution

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, resolution {self.current_resolution}")
        print(f"Phase: {'Transition' if self.is_transition else 'Stabilization'}, Stage steps: {self.current_stage_steps}")

        # Print info about missing keys
        if self.current_resolution < self.config.max_resolution:
            print(f"Note: Higher resolution layers (>{self.current_resolution}x{self.current_resolution}) initialized randomly (expected behavior)")
    
    def generate_samples(self, num_samples=64, use_ema=True):
        """Generate sample images"""
        generator = self.g_ema if use_ema else self.model.generator
        generator.eval()
        
        with torch.no_grad():
            latent_vectors = torch.randn(
                num_samples, self.config.latent_dim, 1, 1,
                device=self.device
            )
            
            fake_images = generator(
                latent_vectors,
                resolution=self.current_resolution,
                alpha=1.0
            )
        
        generator.train()
        return fake_images
    
    def train(self, resume_from=None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Training schedule
        resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        resolutions = [r for r in resolutions if r <= self.config.max_resolution]
        
        start_idx = resolutions.index(self.current_resolution)
        
        for res_idx in range(start_idx, len(resolutions)):
            resolution = resolutions[res_idx]
            
            # Skip if we've already done this resolution
            if resolution < self.current_resolution:
                continue
            
            print(f"\n{'='*60}")
            print(f"Training at {resolution}x{resolution}")
            print(f"{'='*60}")
            
            # Grow networks if needed
            if resolution > self.current_resolution:
                self.model.grow(resolution)
                self.g_ema.grow(resolution)
                self.current_resolution = resolution
            
            #MEMORY MANAGEMENT FIX
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared. Available: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

            
            res_config = self.config.get_resolution_config(resolution)
            batch_size = res_config['batch_size']
            total_images = res_config['images_per_stage']
            
            # Create dataloader
            try:
                dataloader = create_dataloader(
                    data_dir=str(self.config.dataset_path),
                    resolution=resolution,
                    batch_size=batch_size,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory
                )
                dataloader = InfiniteDataLoader(dataloader)
            except Exception as e:
                print(f"Error creating dataloader: {e}")
                return
            
            # Calculate steps
            steps_per_stage = total_images // batch_size
            
            # Fade-in phase
            self.is_transition = True
            self.current_stage_steps = 0
            print(f"\nFade-in phase: {steps_per_stage} steps")
            
            self._train_phase(dataloader, steps_per_stage, "fade-in")
            
            # Stabilization phase
            self.is_transition = False
            self.current_stage_steps = 0
            self.model.set_alpha(1.0)
            print(f"\nStabilization phase: {steps_per_stage} steps")
            
            self._train_phase(dataloader, steps_per_stage, "stabilization")
        
        print("\nTraining completed!")
        self.save_checkpoint("final_model.pt")
    
    def _train_phase(self, dataloader, num_steps, phase_name):
        """Train for a specific phase"""
        pbar = tqdm(range(num_steps), desc=f"{phase_name} @ {self.current_resolution}x{self.current_resolution}")
        
        for step in pbar:
            try:
                # Get batch
                real_images = next(dataloader)
                
                # Train step
                metrics = self.train_step(real_images)
                
                # Update progress bar
                pbar.set_postfix({
                    'D': f"{metrics['d_loss']:.3f}",
                    'G': f"{metrics['g_loss']:.3f}",
                    'GP': f"{metrics['gp']:.3f}",
                    'alpha': f"{self.model.generator.alpha:.3f}"
                })
                
                # Log metrics
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics)
                
                # Save samples
                if self.global_step % self.config.save_interval == 0:
                    self._save_samples()
                    self.save_checkpoint()
                
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  OOM Error at step {self.global_step}!")
                    print(f"Current resolution: {self.current_resolution}x{self.current_resolution}")
                    print(f"Current batch size: {self.config.batch_sizes[self.current_resolution]}")
                    print(f"\n💡 Solutions:")
                    print(f"   1. Reduce batch size in config.py for {self.current_resolution}x{self.current_resolution}")
                    print(f"   2. Reduce num_workers in config.py")
                    print(f"   3. Close other GPU applications")

                    torch.cuda.empty_cache()
                    self.save_checkpoint(f"oom_emergency_res{self.current_resolution}.pt")
                    print(f"\n✅ Emergency checkpoint saved!")
                    raise
                else:
                    raise
            except KeyboardInterrupt:
                print("\n⚠️  Training interrupted by user!")
                self.save_checkpoint(f"interrupted_res{self.current_resolution}.pt")
                raise
    
    def _log_metrics(self, metrics):
        """Log training metrics"""
        log_file = self.config.output_dir / "logs" / "training_log.json"
        
        log_entry = {
            'step': self.global_step,
            'resolution': self.current_resolution,
            'alpha': self.model.generator.alpha,
            **metrics
        }
        
        # Append to log file
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _save_samples(self):
        """Save generated samples"""
        samples = self.generate_samples(num_samples=self.config.num_sample_images)
        
        output_path = (self.config.output_dir / "samples" / 
                      f"samples_step{self.global_step}_res{self.current_resolution}.png")
        
        save_image_grid(samples, str(output_path))


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Progressive GAN")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to CelebA-HQ dataset")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--max_res', type=int, default=1024, help="Maximum resolution")
    
    args = parser.parse_args()
    
    # Update config
    Config.dataset_path = Path(args.data_dir)
    Config.max_resolution = args.max_res
    
    # Create trainer
    trainer = ProgressiveGANTrainer(Config)
    
    # Train
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pt")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint("error_checkpoint.pt")


if __name__ == "__main__":
    main()