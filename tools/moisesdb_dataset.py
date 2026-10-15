"""
Streaming dataset over the MoisesDB lead/rhythm guitar pairs (see
CLAUDE.md "The step change" / 2026-10-15 session). Unlike GuitarSet's
synthetic pipeline, there is no DSP chain here -- real audio is read
directly off disk with on-the-fly random crops, so there's no
fixed-3000-examples-repeat-every-epoch tradeoff this time.

Crop sampling is biased toward windows where BOTH tracks are actually
active (the "overlap fraction" diagnostic from the 2026-10-15 session --
some qualifying songs have the two guitars alternate rather than overlap,
e.g. "Is It All On Me" measured at 0% overlap, which teaches the model
nothing about separating simultaneous content).
"""
import json
import os
import random

import numpy as np
import soundfile as sf
import torch

SR = 44100


def rms_norm(x, target=0.1, floor=1e-6):
    rms = np.sqrt(np.mean(x ** 2))
    if rms < floor:
        return x
    return x * (target / rms)


def find_active_windows(path, win_sec=2.0, thresh=0.01):
    """2s-window RMS envelope for one track, used to find overlap regions."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    win = int(win_sec * sr)
    n = len(mono) // win
    rms = np.array([np.sqrt(np.mean(mono[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    return rms, sr, len(mono)


class MoisesLeadRhythmDataset:
    def __init__(self, data_dir, song_ids, seg_sec=4.0, overlap_bias=0.7, active_thresh=0.01):
        with open(os.path.join(data_dir, "role_labels.json")) as f:
            labels = {r["track_id"]: r for r in json.load(f) if r["usable"]}

        self.seg_sec = seg_sec
        self.overlap_bias = overlap_bias
        self.songs = []
        for tid in song_ids:
            r = labels[tid]
            lead_path = os.path.join(data_dir, tid, "guitar", r["lead_id"] + ".wav")
            rhythm_path = os.path.join(data_dir, tid, "guitar", r["rhythm_id"] + ".wav")
            lead_rms, sr, n_lead = find_active_windows(lead_path, thresh=active_thresh)
            rhythm_rms, sr2, n_rhythm = find_active_windows(rhythm_path, thresh=active_thresh)
            assert sr == sr2 == SR
            n = min(len(lead_rms), len(rhythm_rms))
            both_active = np.where((lead_rms[:n] > active_thresh) & (rhythm_rms[:n] > active_thresh))[0]
            n_samples = min(n_lead, n_rhythm)
            self.songs.append({
                "track_id": tid, "artist": r["artist"], "song": r["song"],
                "lead_path": lead_path, "rhythm_path": rhythm_path,
                "n_samples": n_samples, "overlap_windows": both_active,  # indices of 2s windows, both active
                "overlap_frac": len(both_active) / max(1, n),
            })

    def _sample_start(self, song):
        seg_n = int(self.seg_sec * SR)
        max_start = max(1, song["n_samples"] - seg_n)
        if len(song["overlap_windows"]) > 0 and random.random() < self.overlap_bias:
            win_idx = random.choice(song["overlap_windows"])
            center = int(win_idx * 2.0 * SR) + SR  # center of that 2s window
            start = max(0, min(max_start, center - seg_n // 2))
        else:
            start = random.randint(0, max_start)
        return start

    def sample_batch(self, batch_size):
        seg_n = int(self.seg_sec * SR)
        leads, rhythms = [], []
        for _ in range(batch_size):
            song = random.choice(self.songs)
            start = self._sample_start(song)
            lead, _ = sf.read(song["lead_path"], dtype="float32", always_2d=True,
                               start=start, frames=seg_n)
            rhythm, _ = sf.read(song["rhythm_path"], dtype="float32", always_2d=True,
                                 start=start, frames=seg_n)
            if lead.shape[0] < seg_n:
                lead = np.pad(lead, ((0, seg_n - lead.shape[0]), (0, 0)))
            if rhythm.shape[0] < seg_n:
                rhythm = np.pad(rhythm, ((0, seg_n - rhythm.shape[0]), (0, 0)))
            lead = rms_norm(lead)
            rhythm = rms_norm(rhythm)
            mix_peak = np.abs(lead + rhythm).max()
            if mix_peak > 0.95:
                scale = 0.95 / mix_peak
                lead, rhythm = lead * scale, rhythm * scale
            leads.append(lead.T)  # (C, T)
            rhythms.append(rhythm.T)

        lead_t = torch.from_numpy(np.stack(leads))
        rhythm_t = torch.from_numpy(np.stack(rhythms))
        mix_t = lead_t + rhythm_t
        return mix_t, lead_t, rhythm_t
