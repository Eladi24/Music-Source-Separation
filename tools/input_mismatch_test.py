"""
Input-mismatch test for the Stage 4 lead/rhythm model (2026-10-19).

Training and the notebook's test crops feed the model a CLEAN sum of two
MoisesDB guitar tracks. Real songs feed it a guitar stem that Demucs
(htdemucs_6s) extracted from a finished mix: separation artifacts, bleed from
other instruments, extra guitar parts. This scores the same held-out songs
through three inputs, against the true lead and rhythm tracks:

  A  lead + rhythm tracks only          (the training/test condition)
  B  all guitar tracks of the song      (+ extra guitar parts, still clean)
  C  Demucs 6-stem guitar of the full mix of ALL stems   (the real-song condition)

The A -> B -> C drop says how much of the real-song gap is the input, and which
part of it. Metrics on 4s windows where both true tracks play.

Needs the song's full stems (output/moisesdb_full/<track_id>/, pulled with
range reads) and its role labels (output/moisesdb_guitar/role_labels.json).

    python tools/input_mismatch_test.py output/htd_v5_epoch12.pt --songs "Electric" "False Black Bear"
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
from htdemucs_leadrhythm import load_leadrhythm_checkpoint  # noqa: E402
from realsong_test_stereo import separate_real, SR  # noqa: E402

FULL = "output/moisesdb_full"
LABELS = "output/moisesdb_guitar/role_labels.json"
WIN = 4.0
ACTIVE_RMS = 0.01


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return x.T  # (2, T)


def sisdr(e, r, eps=1e-8):
    e, r = e.reshape(-1) - e.mean(), r.reshape(-1) - r.mean()
    a = (e @ r) / (r @ r + eps)
    t = a * r
    return 10 * np.log10((t @ t) / ((e - t) @ (e - t) + eps) + eps)


def db(x, ref):
    return 10 * np.log10((x ** 2).sum() / ((ref ** 2).sum() + 1e-12) + 1e-12)


def demucs_guitar(mix):
    from demucs.api import Separator
    sep = Separator(model="htdemucs_6s", device="cpu")
    _, stems = sep.separate_tensor(torch.from_numpy(mix), SR)
    return stems["guitar"].numpy()


def score(out0, out1, lead, rhy, wins):
    q0, q1, lk, l0, l1 = [], [], [], [], []
    for a, b in wins:
        q0.append(sisdr(out0[:, a:b], lead[:, a:b])); q1.append(sisdr(out1[:, a:b], rhy[:, a:b]))
        lk.append(sisdr(out1[:, a:b], lead[:, a:b]))
        l0.append(db(out0[:, a:b], lead[:, a:b])); l1.append(db(out1[:, a:b], rhy[:, a:b]))
    m = np.median
    return m(q0), m(q1), m(lk), m(l0), m(l1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--songs", nargs="+", required=True)
    ap.add_argument("--chunk-sec", type=float, default=7.8)
    a = ap.parse_args()
    labels = {r["song"]: r for r in json.load(open(LABELS))}
    model = load_leadrhythm_checkpoint(a.ckpt)
    dev = torch.device("cpu")
    print(f"{os.path.basename(a.ckpt)}, {a.chunk_sec}s chunks; medians over 4s windows where both true tracks play")
    print(f"{'song / input':42s} {'lead SI-SDR':>11s} {'rhythm SI-SDR':>13s} {'leak10':>7s} {'lead lvl':>8s} {'rhy lvl':>8s}")
    for name in a.songs:
        r = labels[name]
        tid = r["track_id"]
        d = os.path.join(FULL, tid)
        lead = load(os.path.join(d, "guitar", r["lead_id"] + ".wav"))
        rhy = load(os.path.join(d, "guitar", r["rhythm_id"] + ".wav"))
        guitars = [load(p) for p in sorted(glob.glob(os.path.join(d, "guitar", "*.wav")))]
        stems = [load(p) for p in sorted(glob.glob(os.path.join(d, "*", "*.wav")))]
        n = min(x.shape[1] for x in stems + guitars + [lead, rhy])
        lead, rhy = lead[:, :n], rhy[:, :n]
        w = int(WIN * SR)
        wins = [(i, i + w) for i in range(0, n - w, w)
                if np.sqrt(np.mean(lead[:, i:i + w] ** 2)) > ACTIVE_RMS and np.sqrt(np.mean(rhy[:, i:i + w] ** 2)) > ACTIVE_RMS]
        inputs = {
            "A lead+rhythm (training condition)": lead + rhy,
            f"B all {len(guitars)} guitar tracks": sum(g[:, :n] for g in guitars),
        }
        mix = sum(s[:, :n] for s in stems)
        dg = demucs_guitar(mix)[:, :n]
        inputs["C Demucs guitar of the full mix"] = dg
        gq = np.median([sisdr(dg[:, a:b], inputs[f"B all {len(guitars)} guitar tracks"][:, a:b]) for a, b in wins])
        print(f"{name}: {len(wins)} windows; {len(stems)} stem files; Demucs guitar vs true guitar sum: SI-SDR {gq:.1f}dB")
        for label, x in inputs.items():
            o0, o1 = separate_real(model, x, dev, chunk_sec=a.chunk_sec)
            q0, q1, lk, l0, l1 = score(o0, o1, lead, rhy, wins)
            print(f"  {label:40s} {q0:11.1f} {q1:13.1f} {lk:7.1f} {l0:+8.1f} {l1:+8.1f}", flush=True)


if __name__ == "__main__":
    main()
