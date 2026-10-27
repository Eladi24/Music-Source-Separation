"""Extract a whole song's guitar stem with BS-Roformer SW (ONNX), for real-song
tests of the lead/rhythm model on a better input than htdemucs_6s.

Same model and wrapper as tools/roformer_vs_demucs.py (validated against the
ONNX repo's reference.npz). Fixed 4s chunks, 2s hop, periodic-Hann overlap-add
(every sample gets two chunks of context). CPU: ~15s per chunk, so a 5-minute
song takes ~35 min. Research only: the weights have no stated licence.

    python tools/roformer_guitar_stem.py "output/04 - The Sentinel/<original>" \
        "output/04 - The Sentinel/guitar_roformer.wav"
"""
import argparse
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
from roformer_vs_demucs import CH, HOP, Roformer, _syn  # noqa: E402

SR = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--onnx", default="output/models/bs_roformer_sw/bs_roformer_sw_6stem_fp16.onnx")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    x, sr = sf.read(args.inp, dtype="float32", always_2d=True)   # libsndfile 1.2: wav/flac/mp3/ogg (no m4a)
    x = torch.from_numpy(x.T.copy())   # (C, T)
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    x = x[:2].numpy().astype(np.float32)
    n = x.shape[1]
    pad = np.pad(x, ((0, 0), (HOP, HOP + CH)))   # first chunk starts at -2s, so sample 0 also gets two chunks
    out = np.zeros_like(pad)
    model = Roformer(args.onnx, args.threads)
    starts = list(range(0, n + HOP, HOP))   # chunk starts in padded coordinates
    t0 = time.time()
    for k, s in enumerate(starts):
        out[:, s:s + CH] += model.guitar(np.ascontiguousarray(pad[:, s:s + CH])) * _syn
        if k % 20 == 0:
            el = time.time() - t0
            print(f"chunk {k+1}/{len(starts)}  {el/60:.1f} min, ~{el/(k+1)*(len(starts)-k-1)/60:.0f} min left", flush=True)
    g = out[:, HOP:HOP + n]
    sf.write(args.out, g.T, SR, subtype="PCM_24")
    print(f"wrote {args.out}: {n/SR:.0f}s, peak {np.abs(g).max():.3f}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
