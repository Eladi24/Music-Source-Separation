import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from src.model import UNet, MaskBasedUNet
from src.dataset import MusicSeparationDataset
from src.loss import StemSeparationLoss
from src.metrics import compute_separation_metrics, MetricsTracker
from src.utils import load_config, istft, save_audio


class Trainer:
    """
    Main trainer class for music source separation.
    """
    
    def __init__(self, config_path: str, resume_from: str = None):
        """
        Args:
            config_path: Path to configuration file
            resume_from: Optional path to checkpoint to resume from
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['training']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Initialize datasets and dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Initialize loss function
        self.criterion = self._build_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _build_model(self):
        """Build model based on configuration."""
        model_config = self.config['model']
        
        if model_config['name'] == 'UNet':
            model = UNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                num_stems=model_config['num_stems'],
                base_channels=model_config['base_channels'],
                depth=model_config['depth'],
                kernel_size=model_config['kernel_size']
            )
        elif model_config['name'] == 'MaskBasedUNet':
            model = MaskBasedUNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                num_stems=model_config['num_stems'],
                base_channels=model_config['base_channels'],
                depth=model_config['depth'],
                kernel_size=model_config['kernel_size']
            )
        else:
            raise ValueError(f"Unknown model: {model_config['name']}")
        
        return model
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders."""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Training dataset
        train_dataset = MusicSeparationDataset(
            raw_dir=data_config['raw_dir'],
            stems_dir=data_config['stems_dir'],
            stems=data_config['stems'],
            sample_rate=data_config['sample_rate'],
            segment_length=data_config['segment_length'],
            n_fft=data_config['n_fft'],
            hop_length=data_config['hop_length'],
            win_length=data_config['win_length'],
            train=True
        )
        
        # Validation dataset (use same data for now, but sequential crops)
        val_dataset = MusicSeparationDataset(
            raw_dir=data_config['raw_dir'],
            stems_dir=data_config['stems_dir'],
            stems=data_config['stems'],
            sample_rate=data_config['sample_rate'],
            segment_length=data_config['segment_length'],
            n_fft=data_config['n_fft'],
            hop_length=data_config['hop_length'],
            win_length=data_config['win_length'],
            train=False
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers'],
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _build_loss_function(self):
        """Build loss function based on configuration."""
        loss_config = self.config['loss']
        
        criterion = StemSeparationLoss(
            stem_names=self.config['data']['stems'],
            spectral_weight=loss_config['weights']['spectral'],
            multi_stft_weight=loss_config['weights'].get('temporal', 0.0),
            spectral_loss_type=loss_config['type'],
            use_multi_stft=loss_config['weights'].get('temporal', 0.0) > 0
        )
        
        return criterion
    
    def _build_optimizer(self):
        """Build optimizer based on configuration."""
        training_config = self.config['training']
        optimizer_config = self.config['optimizer']
        
        if optimizer_config['type'] == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps']
            )
        elif optimizer_config['type'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        return optimizer
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            mix_mag = batch['mix_magnitude'].to(self.device)
            stem_mags = {
                stem: batch['stem_magnitudes'][stem].to(self.device)
                for stem in self.config['data']['stems']
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(mix_mag)
            
            # Reshape output to separate stems
            batch_size, _, freq, time = output.shape
            output = output.view(
                batch_size, 
                self.config['model']['num_stems'], 
                self.config['model']['out_channels'], 
                freq, 
                time
            )
            
            # Create predicted stems dictionary
            pred_stems = {
                stem: output[:, i, :, :, :]
                for i, stem in enumerate(self.config['data']['stems'])
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_stems, stem_mags)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            self.global_step += 1
        
        avg_loss = np.mean(epoch_losses)
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        
        val_losses = []
        metrics_tracker = MetricsTracker(self.config['data']['stems'])
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                # Move data to device
                mix_mag = batch['mix_magnitude'].to(self.device)
                mix_phase = batch['mix_phase'].to(self.device)
                mix_audio = batch['mix_audio'].to(self.device)
                
                stem_mags = {
                    stem: batch['stem_magnitudes'][stem].to(self.device)
                    for stem in self.config['data']['stems']
                }
                
                # Forward pass
                output = self.model(mix_mag)
                
                # Reshape output to separate stems
                batch_size, _, freq, time = output.shape
                output = output.view(
                    batch_size, 
                    self.config['model']['num_stems'], 
                    self.config['model']['out_channels'], 
                    freq, 
                    time
                )
                
                # Create predicted stems dictionary
                pred_stems = {
                    stem: output[:, i, :, :, :]
                    for i, stem in enumerate(self.config['data']['stems'])
                }
                
                # Compute loss
                loss, loss_dict = self.criterion(pred_stems, stem_mags)
                val_losses.append(loss.item())
                
                # Reconstruct audio for metrics (only first batch to save time)
                if batch_idx == 0:
                    pred_audios = {}
                    ref_audios = {}
                    
                    for stem in self.config['data']['stems']:
                        # Use mix phase for reconstruction (simple approach)
                        pred_complex = pred_stems[stem] * torch.exp(1j * mix_phase)
                        pred_audio = istft(
                            pred_complex,
                            n_fft=self.config['data']['n_fft'],
                            hop_length=self.config['data']['hop_length'],
                            win_length=self.config['data']['win_length']
                        )
                        
                        ref_complex = stem_mags[stem] * torch.exp(1j * mix_phase)
                        ref_audio = istft(
                            ref_complex,
                            n_fft=self.config['data']['n_fft'],
                            hop_length=self.config['data']['hop_length'],
                            win_length=self.config['data']['win_length']
                        )
                        
                        pred_audios[stem] = pred_audio
                        ref_audios[stem] = ref_audio
                    
                    # Compute separation metrics
                    batch_metrics = compute_separation_metrics(
                        pred_audios,
                        ref_audios,
                        self.config['data']['stems']
                    )
                    metrics_tracker.update(batch_metrics)
        
        avg_val_loss = np.mean(val_losses)
        avg_metrics = metrics_tracker.compute_average()
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_val_loss, self.current_epoch)
        
        if 'average' in avg_metrics:
            for metric_name, value in avg_metrics['average'].items():
                self.writer.add_scalar(f'val/{metric_name}', value, self.current_epoch)
        
        return avg_val_loss, avg_metrics
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        save_every = self.config['training']['save_every']
        validate_every = self.config['training']['validate_every']
        
        print(f"\nStarting training for {num_epochs} epochs...")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validation
            if epoch % validate_every == 0:
                val_loss, val_metrics = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                print("\nValidation Metrics:")
                print(MetricsTracker(self.config['data']['stems']).get_summary_string() 
                      if val_metrics else "No metrics available")
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                    print("Best model saved!")
            
            # Save checkpoint periodically
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("\nTraining completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train music source separation model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = Trainer(config_path=args.config, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()