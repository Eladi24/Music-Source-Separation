"""Utility functions for audio processing."""

import torch
import torchaudio
import numpy as np
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_audio(path, sample_rate=44100):
    """
    Load audio file and resample if necessary.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        audio: Tensor of shape (channels, samples)
        sr: Sample rate
    """
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
    return audio, sample_rate


def save_audio(audio, path, sample_rate=44100):
    """Save audio tensor to file."""
    torchaudio.save(path, audio.cpu(), sample_rate)


def stft(audio, n_fft=2048, hop_length=512):
    """
    Compute Short-Time Fourier Transform.
    
    Args:
        audio: Audio tensor (channels, samples)
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        spectrogram: Complex spectrogram (channels, freq_bins, time_frames)
    """
    window = torch.hann_window(n_fft).to(audio.device)
    spec = torch.stft(
        audio, 
        n_fft=n_fft, 
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    return spec


def istft(spectrogram, n_fft=2048, hop_length=512):
    """
    Compute Inverse Short-Time Fourier Transform.
    
    Args:
        spectrogram: Complex spectrogram
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        audio: Reconstructed audio
    """
    window = torch.hann_window(n_fft).to(spectrogram.device)
    audio = torch.istft(
        spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window
    )
    return audio