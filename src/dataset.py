import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import random
from typing import Dict, List, Tuple
from src.utils import load_audio, stft


class MusicSeparationDataset(Dataset):
    """
    Dataset for music source separation.
    Expects structure:
    - data/raw/song_name.wav (mixed audio)
    - data/stems/song_name/vocals.wav
    - data/stems/song_name/drums.wav
    - data/stems/song_name/bass.wav
    - data/stems/song_name/other.wav
    """
    
    def __init__(self, 
                 raw_dir: str,
                 stems_dir: str,
                 stems: List[str],
                 sample_rate: int = 44100,
                 segment_length: float = 6.0,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 train: bool = True):
        """
        Args:
            raw_dir: Directory containing mixed audio files
            stems_dir: Directory containing stem subdirectories
            stems: List of stem names (e.g., ['vocals', 'drums', 'bass', 'other'])
            sample_rate: Sample rate
            segment_length: Length of audio segments in seconds
            n_fft: FFT size
            hop_length: STFT hop length
            win_length: STFT window length
            train: If True, use random crops; if False, use sequential crops
        """
        self.raw_dir = Path(raw_dir)
        self.stems_dir = Path(stems_dir)
        self.stems = stems
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.train = train
        
        # Find all songs (assume .wav files in raw_dir)
        self.song_files = sorted(list(self.raw_dir.glob("*.wav")))
        
        if len(self.song_files) == 0:
            raise ValueError(f"No .wav files found in {self.raw_dir}")
        
        print(f"Found {len(self.song_files)} songs in dataset")
    
    def __len__(self) -> int:
        return len(self.song_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'mix': Mixed audio spectrogram (channels, freq, time)
                - 'stems': Dict of stem spectrograms {stem_name: (channels, freq, time)}
                - 'mix_audio': Mixed audio waveform (for validation)
        """
        song_file = self.song_files[idx]
        song_name = song_file.stem
        
        # Load mixed audio
        mix_audio = load_audio(str(song_file), self.sample_rate)
        
        # Ensure stereo
        if mix_audio.shape[0] == 1:
            mix_audio = mix_audio.repeat(2, 1)
        
        # Load stems
        stem_audios = {}
        for stem in self.stems:
            stem_path = self.stems_dir / song_name / f"{stem}.wav"
            
            if not stem_path.exists():
                # If stem doesn't exist, create silence
                stem_audio = torch.zeros_like(mix_audio)
            else:
                stem_audio = load_audio(str(stem_path), self.sample_rate)
                if stem_audio.shape[0] == 1:
                    stem_audio = stem_audio.repeat(2, 1)
            
            stem_audios[stem] = stem_audio
        
        # Random or sequential crop
        if self.train:
            # Random crop for training
            max_start = max(0, mix_audio.shape[1] - self.segment_samples)
            start_sample = random.randint(0, max_start) if max_start > 0 else 0
        else:
            # Use beginning for validation
            start_sample = 0
        
        end_sample = min(start_sample + self.segment_samples, mix_audio.shape[1])
        
        # Crop audio
        mix_audio_crop = mix_audio[:, start_sample:end_sample]
        
        # Pad if necessary
        if mix_audio_crop.shape[1] < self.segment_samples:
            padding = self.segment_samples - mix_audio_crop.shape[1]
            mix_audio_crop = torch.nn.functional.pad(mix_audio_crop, (0, padding))
        
        # Crop and pad stems
        for stem in self.stems:
            stem_audio_crop = stem_audios[stem][:, start_sample:end_sample]
            if stem_audio_crop.shape[1] < self.segment_samples:
                padding = self.segment_samples - stem_audio_crop.shape[1]
                stem_audio_crop = torch.nn.functional.pad(stem_audio_crop, (0, padding))
            stem_audios[stem] = stem_audio_crop
        
        # Compute STFT
        mix_spec = stft(mix_audio_crop, self.n_fft, self.hop_length, self.win_length)
        stem_specs = {
            stem: stft(audio, self.n_fft, self.hop_length, self.win_length)
            for stem, audio in stem_audios.items()
        }
        
        # Convert to magnitude and phase
        mix_mag = torch.abs(mix_spec)
        mix_phase = torch.angle(mix_spec)
        
        stem_mags = {stem: torch.abs(spec) for stem, spec in stem_specs.items()}
        
        return {
            'mix_magnitude': mix_mag,
            'mix_phase': mix_phase,
            'stem_magnitudes': stem_mags,
            'mix_audio': mix_audio_crop,  # Keep for validation
            'song_name': song_name
        }