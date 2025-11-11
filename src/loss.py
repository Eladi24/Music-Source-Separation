import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class SpectralLoss(nn.Module):
    """
    Loss function for magnitude spectrograms.
    Supports L1, L2, and combination losses.
    """
    
    def __init__(self, loss_type='l1', reduction='mean'):
        """
        Args:
            loss_type: Type of loss ('l1', 'l2', or 'l1_l2')
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1_l2':
            self.l1 = nn.L1Loss(reduction=reduction)
            self.l2 = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, predicted, target):
        """
        Args:
            predicted: Predicted magnitude spectrogram (batch, channels, freq, time)
            target: Target magnitude spectrogram (batch, channels, freq, time)
            
        Returns:
            loss: Scalar loss value
        """
        if self.loss_type == 'l1_l2':
            return self.l1(predicted, target) + self.l2(predicted, target)
        else:
            return self.criterion(predicted, target)


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for better perceptual quality.
    Computes loss across multiple STFT resolutions.
    """
    
    def __init__(self, 
                 fft_sizes=[2048, 1024, 512, 256],
                 hop_sizes=[512, 256, 128, 64],
                 win_lengths=[2048, 1024, 512, 256],
                 sc_weight=1.0,
                 mag_weight=1.0):
        """
        Args:
            fft_sizes: List of FFT sizes
            hop_sizes: List of hop sizes
            win_lengths: List of window lengths
            sc_weight: Weight for spectral convergence loss
            mag_weight: Weight for log magnitude loss
        """
        super().__init__()
        
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sc_weight = sc_weight
        self.mag_weight = mag_weight
    
    def stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT."""
        window = torch.hann_window(win_length, device=x.device)
        
        # Reshape if needed: (batch, channels, samples) -> (batch * channels, samples)
        batch_size, channels, samples = x.shape
        x = x.reshape(batch_size * channels, samples)
        
        spec = torch.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True
        )
        
        # Reshape back: (batch * channels, freq, time) -> (batch, channels, freq, time)
        freq_bins, time_frames = spec.shape[-2:]
        spec = spec.reshape(batch_size, channels, freq_bins, time_frames)
        
        return spec
    
    def spectral_convergence(self, predicted_mag, target_mag):
        """Compute spectral convergence loss."""
        return torch.norm(target_mag - predicted_mag, p='fro') / torch.norm(target_mag, p='fro')
    
    def log_magnitude_loss(self, predicted_mag, target_mag):
        """Compute log magnitude loss."""
        return F.l1_loss(torch.log1p(predicted_mag), torch.log1p(target_mag))
    
    def forward(self, predicted_audio, target_audio):
        """
        Args:
            predicted_audio: Predicted audio waveform (batch, channels, samples)
            target_audio: Target audio waveform (batch, channels, samples)
            
        Returns:
            loss: Scalar loss value
        """
        total_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT
            predicted_spec = self.stft(predicted_audio, fft_size, hop_size, win_length)
            target_spec = self.stft(target_audio, fft_size, hop_size, win_length)
            
            # Get magnitudes
            predicted_mag = torch.abs(predicted_spec)
            target_mag = torch.abs(target_spec)
            
            # Spectral convergence loss
            sc_loss = self.spectral_convergence(predicted_mag, target_mag)
            
            # Log magnitude loss
            mag_loss = self.log_magnitude_loss(predicted_mag, target_mag)
            
            # Combine losses
            total_loss += self.sc_weight * sc_loss + self.mag_weight * mag_loss
        
        # Average over resolutions
        return total_loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """
    Combined loss for music source separation.
    Combines spectral loss with optional multi-resolution STFT loss.
    """
    
    def __init__(self, 
                 spectral_weight=1.0,
                 multi_stft_weight=0.0,
                 spectral_loss_type='l1',
                 use_multi_stft=False):
        """
        Args:
            spectral_weight: Weight for spectral domain loss
            multi_stft_weight: Weight for multi-resolution STFT loss
            spectral_loss_type: Type of spectral loss ('l1', 'l2', 'l1_l2')
            use_multi_stft: Whether to use multi-resolution STFT loss
        """
        super().__init__()
        
        self.spectral_weight = spectral_weight
        self.multi_stft_weight = multi_stft_weight
        self.use_multi_stft = use_multi_stft
        
        self.spectral_loss = SpectralLoss(loss_type=spectral_loss_type)
        
        if use_multi_stft:
            self.multi_stft_loss = MultiResolutionSTFTLoss()
    
    def forward(self, predicted_mags, target_mags, predicted_audio=None, target_audio=None):
        """
        Args:
            predicted_mags: Dict or tensor of predicted magnitude spectrograms
            target_mags: Dict or tensor of target magnitude spectrograms
            predicted_audio: Optional predicted audio for multi-resolution STFT loss
            target_audio: Optional target audio for multi-resolution STFT loss
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Spectral loss
        if isinstance(predicted_mags, dict):
            # Multiple stems
            spectral_loss = 0.0
            for stem_name in predicted_mags.keys():
                stem_loss = self.spectral_loss(predicted_mags[stem_name], target_mags[stem_name])
                spectral_loss += stem_loss
            spectral_loss /= len(predicted_mags)
        else:
            # Single output
            spectral_loss = self.spectral_loss(predicted_mags, target_mags)
        
        loss_dict['spectral'] = spectral_loss.item()
        total_loss = self.spectral_weight * spectral_loss
        
        # Multi-resolution STFT loss
        if self.use_multi_stft and predicted_audio is not None and target_audio is not None:
            if isinstance(predicted_audio, dict):
                # Multiple stems
                stft_loss = 0.0
                for stem_name in predicted_audio.keys():
                    stem_stft_loss = self.multi_stft_loss(
                        predicted_audio[stem_name], 
                        target_audio[stem_name]
                    )
                    stft_loss += stem_stft_loss
                stft_loss /= len(predicted_audio)
            else:
                # Single output
                stft_loss = self.multi_stft_loss(predicted_audio, target_audio)
            
            loss_dict['multi_stft'] = stft_loss.item()
            total_loss += self.multi_stft_weight * stft_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class StemSeparationLoss(nn.Module):
    """
    Complete loss function for stem separation that handles multiple stems.
    """
    
    def __init__(self, 
                 stem_names: List[str],
                 spectral_weight=1.0,
                 multi_stft_weight=0.0,
                 spectral_loss_type='l1',
                 use_multi_stft=False,
                 stem_weights=None):
        """
        Args:
            stem_names: List of stem names
            spectral_weight: Weight for spectral domain loss
            multi_stft_weight: Weight for multi-resolution STFT loss
            spectral_loss_type: Type of spectral loss
            use_multi_stft: Whether to use multi-resolution STFT loss
            stem_weights: Optional dict of weights per stem
        """
        super().__init__()
        
        self.stem_names = stem_names
        self.combined_loss = CombinedLoss(
            spectral_weight=spectral_weight,
            multi_stft_weight=multi_stft_weight,
            spectral_loss_type=spectral_loss_type,
            use_multi_stft=use_multi_stft
        )
        
        # Default equal weights for all stems
        if stem_weights is None:
            self.stem_weights = {name: 1.0 for name in stem_names}
        else:
            self.stem_weights = stem_weights
    
    def forward(self, predicted_stems, target_stems, predicted_audio=None, target_audio=None):
        """
        Args:
            predicted_stems: Dict of predicted spectrograms {stem_name: tensor}
            target_stems: Dict of target spectrograms {stem_name: tensor}
            predicted_audio: Optional dict of predicted audio {stem_name: tensor}
            target_audio: Optional dict of target audio {stem_name: tensor}
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with per-stem losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        for stem_name in self.stem_names:
            # Compute loss for this stem
            stem_pred_mag = predicted_stems[stem_name]
            stem_target_mag = target_stems[stem_name]
            
            stem_pred_audio = predicted_audio[stem_name] if predicted_audio else None
            stem_target_audio = target_audio[stem_name] if target_audio else None
            
            stem_loss, stem_loss_dict = self.combined_loss(
                stem_pred_mag,
                stem_target_mag,
                stem_pred_audio,
                stem_target_audio
            )
            
            # Weight and accumulate
            weighted_loss = stem_loss * self.stem_weights[stem_name]
            total_loss += weighted_loss
            
            # Store per-stem losses
            loss_dict[f'{stem_name}_total'] = weighted_loss.item()
            for key, value in stem_loss_dict.items():
                loss_dict[f'{stem_name}_{key}'] = value
        
        # Average over stems
        total_loss /= len(self.stem_names)
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test spectral loss
    print("Testing SpectralLoss...")
    loss_fn = SpectralLoss(loss_type='l1')
    pred = torch.randn(4, 2, 1025, 256).to(device)
    target = torch.randn(4, 2, 1025, 256).to(device)
    loss = loss_fn(pred, target)
    print(f"L1 Loss: {loss.item():.4f}")
    
    # Test multi-resolution STFT loss
    print("\nTesting MultiResolutionSTFTLoss...")
    stft_loss_fn = MultiResolutionSTFTLoss()
    pred_audio = torch.randn(4, 2, 44100).to(device)
    target_audio = torch.randn(4, 2, 44100).to(device)
    loss = stft_loss_fn(pred_audio, target_audio)
    print(f"Multi-Resolution STFT Loss: {loss.item():.4f}")
    
    # Test stem separation loss
    print("\nTesting StemSeparationLoss...")
    stem_names = ['vocals', 'drums', 'bass', 'other']
    stem_loss_fn = StemSeparationLoss(
        stem_names=stem_names,
        spectral_weight=1.0,
        multi_stft_weight=0.1,
        use_multi_stft=True
    )
    
    # Create dummy predictions and targets
    pred_stems = {name: torch.randn(4, 2, 1025, 256).to(device) for name in stem_names}
    target_stems = {name: torch.randn(4, 2, 1025, 256).to(device) for name in stem_names}
    pred_audio_stems = {name: torch.randn(4, 2, 44100).to(device) for name in stem_names}
    target_audio_stems = {name: torch.randn(4, 2, 44100).to(device) for name in stem_names}
    
    loss, loss_dict = stem_loss_fn(pred_stems, target_stems, pred_audio_stems, target_audio_stems)
    print(f"Total Loss: {loss.item():.4f}")
    print("Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")