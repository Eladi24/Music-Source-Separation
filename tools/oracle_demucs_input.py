"""How much of the lead/rhythm targets can be recovered from the Demucs input?

v6 and v6a trained on crops whose INPUT is the htdemucs_6s guitar of the full
mix and whose TARGETS are the true clean tracks. Both runs see-sawed into the
copy shortcut. Hypothesis: part of those targets is simply not in the input
(Demucs dropped or smeared it), and the loss answers unreachable targets by
hedging. This measures it, per song, with an oracle mask: the best real-valued
STFT mask in [0, 1] (phase-sensitive mask, MSE-optimal for masking) computed
from the true target, applied to the input. Same for the clean input (lead +
rhythm) as the reference ceiling.

Windows: 4s, both lead and rhythm playing, no lead-like extra guitar playing
(the same rule the notebook's Demucs-input crops use). Targets as in training:
Demucs input -> lead, rhythm + accompaniment-like extras.

Reported per song (medians over windows), SI-SDR in dB:
  faith   Demucs input vs the true guitar sum (how faithful the input is)
  oc_L/R  oracle mask on the CLEAN input -> lead / rhythm
  od_L/R  oracle mask on the DEMUCS input -> lead / rhythm
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

SR = 44100
WIN = 4 * SR
ACTIVE = 0.01
N_FFT, HOP = 4096, 1024


def load(path, gain=1.0):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, path
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.T / gain   # (2, T)


def rms2s(x):
    m = x.mean(axis=0)
    w = 2 * SR
    return np.array([np.sqrt(np.mean(m[i*w:(i+1)*w] ** 2)) for i in range(len(m) // w)])


def sisdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    a = np.sum(est * ref) / max(np.sum(ref * ref), 1e-12)
    t = a * ref
    return 10 * np.log10(max(np.sum(t * t), 1e-12) / max(np.sum((est - t) ** 2), 1e-12))


_win = torch.hann_window(N_FFT)


def stft(x):
    return torch.stft(torch.from_numpy(x), N_FFT, HOP, window=_win, return_complex=True)


def oracle(inp, target):
    """Phase-sensitive mask clipped to [0, 1], applied to the input."""
    X, T = stft(inp), stft(target)
    m = ((T * X.conj()).real / X.abs().clamp_min(1e-8) ** 2).clamp(0, 1)
    return torch.istft(m * X, N_FFT, HOP, window=_win, length=inp.shape[-1]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar-dir", default="output/moisesdb_guitar")
    ap.add_argument("--stems-dir", default="output/moisesdb_stems")
    ap.add_argument("--extras", default="tools/extra_guitar_roles.json")
    ap.add_argument("--out", default="tools/oracle_demucs_input.json")
    args = ap.parse_args()

    labels = [r for r in json.load(open(os.path.join(args.guitar_dir, "role_labels.json"))) if r["usable"]]
    stems = json.load(open(os.path.join(args.stems_dir, "stems_manifest.json")))
    extras = json.load(open(args.extras))
    results = {}
    print(f"{'song':38s} {'n':>3s} {'faith':>6s} {'oc_L':>6s} {'oc_R':>6s} {'od_L':>6s} {'od_R':>6s}")
    for r in labels:
        tid = r["track_id"]
        gdir = os.path.join(args.guitar_dir, tid, "guitar")
        L = load(os.path.join(gdir, r["lead_id"] + ".wav"))
        R = load(os.path.join(gdir, r["rhythm_id"] + ".wav"))
        D = load(os.path.join(args.stems_dir, tid, "demucs_guitar.flac"), stems[tid]["demucs_guitar"]["gain"])
        n = min(L.shape[1], R.shape[1], D.shape[1])
        L, R, D = L[:, :n], R[:, :n], D[:, :n]
        R_ext = R.copy()
        blocked = set()
        for gid, e in extras.get(tid, {}).items():
            if e["use"] == "rhythm_extra":
                R_ext += load(os.path.join(gdir, gid + ".wav"))[:, :n]
            elif e["use"] == "block":
                act = np.where(rms2s(load(os.path.join(gdir, gid + ".wav"))[:, :n]) > ACTIVE)[0]
                blocked |= {int(i) for i in act}
        lr, rr = rms2s(L), rms2s(R)
        rows = []
        for w in range(n // WIN):
            i0, i1 = 2 * w, 2 * w + 1   # the two 2s windows inside this 4s window
            if not all(lr[i] > ACTIVE and rr[i] > ACTIVE for i in (i0, i1)):
                continue
            if {i0 - 1, i0, i1, i1 + 1} & blocked:
                continue
            s = slice(w * WIN, (w + 1) * WIN)
            l, rc, re, d = L[:, s], R[:, s], R_ext[:, s], D[:, s]
            x = l + rc
            rows.append([
                sisdr(d, l + re),
                sisdr(oracle(x, l), l), sisdr(oracle(x, rc), rc),
                sisdr(oracle(d, l), l), sisdr(oracle(d, re), re),
                float(w * 4),
            ])
        rows = np.array(rows) if rows else np.zeros((0, 6))
        med = np.median(rows[:, :5], axis=0) if len(rows) else [np.nan] * 5
        results[tid] = {"song": r["song"], "artist": r["artist"], "n": len(rows),
                        "median": dict(zip(["faith", "oc_L", "oc_R", "od_L", "od_R"], map(float, med))),
                        "windows": rows.tolist()}
        print(f"{r['artist'][:14] + ' - ' + r['song']:38.38s} {len(rows):3d} " + " ".join(f"{v:6.1f}" for v in med),
              flush=True)
    allw = np.concatenate([np.array(v["windows"])[:, :5] for v in results.values() if v["n"]])
    print(f"{'ALL windows (median)':38s} {len(allw):3d} " + " ".join(f"{v:6.1f}" for v in np.median(allw, axis=0)))
    json.dump(results, open(args.out, "w"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
