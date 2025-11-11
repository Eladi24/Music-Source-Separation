import torch
import torchaudio
import yaml
import numpy as np
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_audio(file_path: str, sample_rate: int = 44100) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        audio: Tensor of shape (channels, samples)
    """
    audio, sr = torchaudio.load(file_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    
    return audio


def save_audio(audio: torch.Tensor, file_path: str, sample_rate: int = 44100):
    """
    Save audio tensor to file.
    
    Args:
        audio: Tensor of shape (channels, samples)
        file_path: Output file path
        sample_rate: Sample rate
    """
    torchaudio.save(file_path, audio.cpu(), sample_rate)


def stft(audio: torch.Tensor, n_fft: int = 2048, hop_length: int = 512, 
         win_length: int = 2048, window: str = 'hann') -> torch.Tensor:
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        audio: Input audio tensor (channels, samples)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        window: Window function
        
    Returns:
        spectrogram: Complex spectrogram (channels, freq_bins, time_frames)
    """
    if window == 'hann':
        window_fn = torch.hann_window(win_length, device=audio.device)
    else:
        window_fn = torch.hamming_window(win_length, device=audio.device)
    
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_fn,
        return_complex=True,
        center=True,
        normalized=False
    )
    
    return spec


def istft(spectrogram: torch.Tensor, n_fft: int = 2048, hop_length: int = 512,
          win_length: int = 2048, window: str = 'hann') -> torch.Tensor:
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        spectrogram: Complex spectrogram (channels, freq_bins, time_frames)
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        window: Window function
        
    Returns:
        audio: Reconstructed audio (channels, samples)
    """
    if window == 'hann':
        window_fn = torch.hann_window(win_length, device=spectrogram.device)
    else:
        window_fn = torch.hamming_window(win_length, device=spectrogram.device)
    
    audio = torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_fn,
        center=True,
        normalized=False
    )
    
    return audio


def magnitude_phase(complex_spec: torch.Tensor) -> tuple:
    """
    Convert complex spectrogram to magnitude and phase.
    
    Args:
        complex_spec: Complex spectrogram
        
    Returns:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
    """
    magnitude = torch.abs(complex_spec)
    phase = torch.angle(complex_spec)
    return magnitude, phase


def complex_from_magnitude_phase(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct complex spectrogram from magnitude and phase.
    
    Args:
        magnitude: Magnitude spectrogram
        phase: Phase spectrogram
        
    Returns:
        complex_spec: Complex spectrogram
    """
    return magnitude * torch.exp(1j * phase)