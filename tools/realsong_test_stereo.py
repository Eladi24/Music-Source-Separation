"""
Real-song lead/rhythm test for the stereo/44.1kHz HTDemucs line (Stage 4).
Separate from tools/realsong_test.py (mono/16kHz, ConvTasNet) rather than a
shared rewrite -- different sample rate, channel count, and model entirely.

First version deliberately simple: naive overlap-add, no cross-chunk
continuity stitching or gain-fix (those were ConvTasNet-specific fixes for
problems not yet confirmed to exist here -- see tools/stitch.py's history).
Add them only if a real symptom shows up on this model specifically.

Usage:
    source ~/pytorch_env/bin/activate
    python tools/realsong_test_stereo.py --ckpt output/htd_leadrhythm_lr2e4_epoch8.pt \
        --stem "output/04 - The Sentinel/guitar.wav" --out-prefix output/stage4_local_test/sentinel
"""
import argparse
import os
import sys
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
from htdemucs_leadrhythm import load_leadrhythm_checkpoint

SR = 44100
TIPTON_SOLO, DOWNING_SOLO = (141.0, 146.0), (147.0, 151.0)


def load_stereo(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"expected {SR}Hz, got {sr}Hz -- no resampling implemented in this tool"
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data.T  # (2, T)


def write_stereo(path, x, sr=SR):
    sf.write(path, x.T, sr)


def build_model(ckpt_path, device):
    model = load_leadrhythm_checkpoint(ckpt_path)   # plain or v4a MixConsistent
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  model: {n:,} params, loaded {ckpt_path}")
    return model


@torch.no_grad()
def separate_real(model, audio, device, chunk_sec=4.0):
    """audio: (2, T). Naive chunking, Hann overlap-add, no swap/gain logic."""
    chunk = int(round(chunk_sec * SR))
    hop = chunk // 2
    total_len = audio.shape[1]
    starts = list(range(0, max(total_len - chunk + 1, 1), hop))
    win = np.hanning(chunk)

    o0 = np.zeros((2, total_len))
    o1 = np.zeros((2, total_len))
    ws = np.zeros(total_len)
    for s in starts:
        seg = audio[:, s:s + chunk]
        if seg.shape[1] < chunk:
            seg = np.pad(seg, ((0, 0), (0, chunk - seg.shape[1])))
        x = torch.from_numpy(seg).float().unsqueeze(0).to(device)  # (1, 2, chunk)
        out = model(x).squeeze(0).float().cpu().numpy()  # (2 sources, 2 ch, chunk)
        m = min(chunk, total_len - s)
        o0[:, s:s + m] += out[0, :, :m] * win[:m]
        o1[:, s:s + m] += out[1, :, :m] * win[:m]
        ws[s:s + m] += win[:m]
    ws[ws < 1e-8] = 1e-8
    return o0 / ws, o1 / ws


def rms_win(x_mono, a, b):
    seg = x_mono[int(a * SR):int(b * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stem", default="output/04 - The Sentinel/guitar.wav")
    ap.add_argument("--chunk-sec", type=float, default=7.8,
                     help="inference chunk length. Default 7.8s = HTDemucs's native window (no padding); "
                          "the user preferred it by ear over padded 4s chunks on v5 ep12 (CLAUDE.md 2026-10-19). "
                          "Shorter chunks are zero-padded to 7.8s unless --match-train-segment is set")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--match-train-segment", action="store_true",
                     help="set the model's internal window to --chunk-sec, as in training. HTDemucs sets "
                          "segment = input length in train mode (no padding), but a freshly loaded model keeps "
                          "its pretrained 7.8s window and zero-pads shorter chunks to it at inference "
                          "(CLAUDE.md 2026-10-19)")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    model = build_model(a.ckpt, device)
    if a.match_train_segment:
        core = model.base if hasattr(model, "base") else model
        core.segment = Fraction(int(round(a.chunk_sec * SR)), SR)
        print(f"  model segment set to {float(core.segment):.2f}s (no padding)")
    audio = load_stereo(a.stem)
    print(f"  stem: {a.stem}  {audio.shape[1]/SR:.1f}s  stereo")

    chunk = int(round(a.chunk_sec * SR))
    print(f"  chunk {chunk/SR:.1f}s  hop {chunk/2/SR:.1f}s  "
          f"~{max(audio.shape[1]//(chunk//2)-1, 0)} boundaries")
    out0, out1 = separate_real(model, audio, device, chunk_sec=a.chunk_sec)

    out0_m, out1_m = out0.mean(0), out1.mean(0)  # mono downmix, for the diagnostic metrics only
    corr = float(np.corrcoef(out0_m, out1_m)[0, 1])
    print(f"\ncorr(out0, out1)              : {corr:+.3f}   (lower = more distinct)")
    for name, (lo, hi) in [("Tipton-solo ", TIPTON_SOLO), ("Downing-solo", DOWNING_SOLO)]:
        r0, r1 = rms_win(out0_m, lo, hi), rms_win(out1_m, lo, hi)
        print(f"{name} window  out0/out1: {r0:.4f} / {r1:.4f}   ratio {r0/(r1+1e-9):.2f}  (want > 1)")

    if a.out_prefix:
        os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
        write_stereo(f"{a.out_prefix}_out0_lead.wav", out0)
        write_stereo(f"{a.out_prefix}_out1_rhythm.wav", out1)
        print(f"\nwrote {a.out_prefix}_out0_lead.wav / {a.out_prefix}_out1_rhythm.wav")


if __name__ == "__main__":
    main()
