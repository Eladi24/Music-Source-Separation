"""
v6 step 2 (2026-10-23): the real-song input for each MoisesDB pair song.

Real songs reach the lead/rhythm model as a guitar stem that htdemucs_6s
extracted from a finished mix; training only ever saw clean guitar tracks
(input-mismatch test, 2026-10-19: that costs 6-10dB). This rebuilds each pair
song's full mix (24-bit guitar tracks from output/moisesdb_guitar/ + the
category stems from tools/pull_moisesdb_stems.py), runs htdemucs_6s on it
with the app's settings (demucs.api defaults), and saves the guitar stem:

    output/moisesdb_stems/<track_id>/demucs_guitar.flac   (16-bit, gain in manifest)

Also logs how close the Demucs guitar is to the true sum of guitar tracks.
Resumable; CPU only (~1-2x real time per song).

    python tools/demucs_guitar_inputs.py
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

SR = 44100


def load(path, gain=1.0):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, path
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.T / gain  # (2, T)


def add(acc, x):
    if acc is None:
        return x.copy()
    n = max(acc.shape[1], x.shape[1])
    out = np.zeros((2, n), np.float32)
    out[:, :acc.shape[1]] += acc
    out[:, :x.shape[1]] += x
    return out


def sisdr(e, r, eps=1e-8):
    e, r = e.reshape(-1) - e.mean(), r.reshape(-1) - r.mean()
    t = (e @ r) / (r @ r + eps) * r
    return 10 * np.log10((t @ t) / ((e - t) @ (e - t) + eps) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="output/moisesdb_stems")
    ap.add_argument("--guitar", default="output/moisesdb_guitar")
    a = ap.parse_args()
    from demucs.api import Separator
    torch.set_num_threads(os.cpu_count())
    sep = Separator(model="htdemucs_6s", device="cpu", progress=False)

    man_path = os.path.join(a.stems, "stems_manifest.json")
    manifest = json.load(open(man_path))
    t0 = time.time()
    for k, (tid, e) in enumerate(manifest.items()):
        if "demucs_guitar" in e:
            continue
        d = os.path.join(a.stems, tid)
        guitar = None
        for p in sorted(glob.glob(os.path.join(a.guitar, tid, "guitar", "*.wav"))):
            guitar = add(guitar, load(p))
        mix = guitar
        for c, info in e["cats"].items():
            mix = add(mix, load(os.path.join(d, f"{c}.flac"), info["gain"]))
        n = guitar.shape[1]
        _, stems = sep.separate_tensor(torch.from_numpy(mix), SR)
        dg = stems["guitar"].numpy()[:, :mix.shape[1]]
        q = sisdr(dg[:, :n], guitar)
        peak = float(np.abs(dg).max())
        gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
        sf.write(os.path.join(d, "demucs_guitar.flac"), (dg * gain).T, SR, subtype="PCM_16")
        e["demucs_guitar"] = {"gain": gain, "frames": int(dg.shape[1]), "sisdr_vs_true_guitar": round(float(q), 2)}
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"[{k+1}/{len(manifest)}] {e['artist']} - {e['song']}: {mix.shape[1]/SR:.0f}s, "
              f"Demucs guitar vs true guitar SI-SDR {q:.1f}dB; {(time.time()-t0)/60:.1f} min so far", flush=True)
        del mix, guitar, stems, dg
    print("done", flush=True)


if __name__ == "__main__":
    main()
