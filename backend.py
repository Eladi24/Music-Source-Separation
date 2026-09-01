"""
Separation backend: a thin wrapper around Demucs (Meta's pretrained
source-separation models).

We deliberately don't train anything ourselves here — see README.md for the
reasoning. This module just picks a model preset, runs it, and writes the
resulting stems to disk.
"""
from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import demucs.api

logger = logging.getLogger(__name__)

# Presets shown in the GUI, mapped to the actual Demucs model name.
# - htdemucs_ft: Meta's fine-tuned 4-source model (vocals/drums/bass/other).
#   Best quality on the core 4 stems.
# - htdemucs_6s: same architecture trained for 6 sources, adding guitar and
#   piano. Slightly noisier on the core 4 than the _ft model.
MODEL_PRESETS = {
    "4-stem (best quality)": "htdemucs_ft",
    "6-stem (adds guitar & piano)": "htdemucs_6s",
}
DEFAULT_PRESET = "4-stem (best quality)"

# Display order for stems in the UI; not every preset produces every stem.
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


@dataclass
class SeparationResult:
    track_name: str
    stems: dict[str, Path]  # stem name -> path to a saved .wav file


def _pick_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(
    audio_path: str,
    preset: str = DEFAULT_PRESET,
    use_gpu: bool = False,
    output_dir: str = "output",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> SeparationResult:
    """
    Run Demucs separation on a single audio file and save each stem as a wav.

    Falls back to CPU automatically if a GPU run fails (e.g. out of memory --
    relevant on small GPUs like a 2GB laptop card).

    Args:
        progress_callback: optional callable(fraction 0..1, description) invoked
            as separation proceeds. Driven by Demucs's own callback hook, so the
            percentage reflects real progress through the model(s), not a guess.
    """
    model_name = MODEL_PRESETS.get(preset, preset)
    device = _pick_device(use_gpu)

    def _demucs_progress(state: dict):
        if progress_callback is None:
            return
        # htdemucs_ft is a "bag" of several sub-models averaged together;
        # `models`/`model_idx_in_bag` tell us which one we're on. `shifts`
        # defaults to 1 (we never set it higher), so shift_idx never adds a
        # second dimension here. `audio_length`/`segment_offset` give position
        # within the current model's pass.
        models_total = max(state.get("models", 1), 1)
        model_idx = state.get("model_idx_in_bag", 0)
        audio_length = state.get("audio_length") or 1
        segment_offset = state.get("segment_offset", 0)
        frac_within_model = min(segment_offset / audio_length, 1.0)
        frac = (model_idx + frac_within_model) / models_total
        progress_callback(min(frac, 0.99), "Separating")

    def _run(device: str):
        separator = demucs.api.Separator(
            model=model_name,
            device=device,
            progress=False,
            callback=_demucs_progress,
        )
        _, separated = separator.separate_audio_file(audio_path)
        return separator, separated

    if progress_callback:
        progress_callback(0.0, "Loading model")

    try:
        separator, separated = _run(device)
    except RuntimeError as e:
        if device == "cuda" and "out of memory" in str(e).lower():
            logger.warning("GPU ran out of memory, retrying on CPU")
            torch.cuda.empty_cache()
            if progress_callback:
                progress_callback(0.0, "GPU out of memory, retrying on CPU")
            separator, separated = _run("cpu")
        else:
            raise

    if progress_callback:
        progress_callback(0.95, "Saving stems")

    track_name = Path(audio_path).stem
    out_dir = Path(output_dir) / track_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_paths = {}
    for stem_name, wav in separated.items():
        out_path = out_dir / f"{stem_name}.wav"
        demucs.api.save_audio(wav, str(out_path), samplerate=separator.samplerate)
        stem_paths[stem_name] = out_path

    if progress_callback:
        progress_callback(1.0, "Done")

    return SeparationResult(track_name=track_name, stems=stem_paths)


def _read_wav_stereo(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        sw = f.getsampwidth()
        ch = f.getnchannels()
        raw = f.readframes(n)
    if ch != 2:
        raise ValueError(f"{path}: expected stereo audio, got {ch} channel(s)")
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    max_val = float(2 ** (8 * sw - 1))
    data = np.frombuffer(raw, dtype=dtype).reshape(-1, ch).astype(np.float64) / max_val
    return data[:, 0], data[:, 1], sr


def _write_wav_stereo(path: str, left: np.ndarray, right: np.ndarray, sr: int) -> None:
    stereo = np.stack([left, right], axis=1)
    stereo = np.clip(stereo, -1.0, 1.0)
    pcm = (stereo * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def split_by_stereo_position(
    stem_path: str,
    strength: float = 0.3,
    output_dir: str = "output",
) -> dict[str, Path]:
    """
    EXPERIMENTAL. Split an already-separated stem (e.g. a combined "guitar"
    stem) into a left-leaning and right-leaning pair, based purely on
    stereo position -- for cases where the stem still contains multiple
    simultaneous performances of the same instrument that were panned
    apart in the mix (very common for double-tracked guitars in rock/metal).

    This has no notion of instrument identity -- it cannot separate sources
    that were mixed to the same position. Validated (synthetic tests +
    real-track stats, see CLAUDE.md) to give a real, measurable improvement
    when sources are panned apart, and correctly no improvement at all when
    they're centered/mono -- there's no way around that with this approach.

    `strength` (0-1): 0 = plain per-channel selection (best when sources are
    panned hard apart -- practically perfect in that case). Higher values
    additionally cancel content shared with the opposite channel, which
    helps for moderately-panned sources but starts hurting already-hard-
    panned audio if pushed too high. ~0.2-0.3 is a good default; expose it
    as a slider rather than hardcoding it, since the right value depends on
    how the specific track was mixed.
    """
    left, right, sr = _read_wav_stereo(stem_path)
    left_est = left - strength * right
    right_est = right - strength * left

    stem_name = Path(stem_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_path = out_dir / f"{stem_name}_left.wav"
    right_path = out_dir / f"{stem_name}_right.wav"
    _write_wav_stereo(str(left_path), left_est, left_est, sr)
    _write_wav_stereo(str(right_path), right_est, right_est, sr)

    return {"left": left_path, "right": right_path}
