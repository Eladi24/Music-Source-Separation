"""
Per-window lead check for the two-rhythm-guitar MoisesDB songs (Stage 4 v3).
These songs were rated rhythm-like on average, but some tracks still have
short lead-like stretches (melody lines, fills): 13-15% of windows in 3 songs.
Training those as "rhythm" would contradict the main lead/rhythm data, so this
writes the lead-rated windows to tools/dbl_rhythm_blocked_windows.json, which
tools/build_stage4_notebook.py embeds in the notebook. The dataset then never
samples a two-rhythm crop overlapping a blocked window. See CLAUDE.md 2026-10-18.

    python tools/dbl_rhythm_windows.py
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
from train_role_classifier import RoleNet, SR, SEG  # noqa: E402

DATA = "output/moisesdb_guitar"
OUT = os.path.join(os.path.dirname(__file__), "dbl_rhythm_blocked_windows.json")
P_LEAD_BLOCK = 0.5


def lead_windows(net, path):
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torchaudio.functional.resample(torch.from_numpy(d.mean(1)), sr, SR)
    n = w.shape[0] // SEG
    segs = torch.stack([w[i * SEG:(i + 1) * SEG] for i in range(n)])
    pk = segs.abs().amax(1, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        p = torch.softmax(net(segs / pk * 0.8), 1)[:, 1].numpy()
    active = (pk.squeeze(1) > 0.01).numpy()
    return [i for i in range(n) if active[i] and p[i] > P_LEAD_BLOCK]


def main():
    net = RoleNet()
    net.load_state_dict(torch.load("output/role_classifier.pt", map_location="cpu"))
    net.eval()
    labels = json.load(open(f"{DATA}/role_labels.json"))
    win_sec = SEG / SR
    out = {"window_sec": win_sec, "blocked": {}}
    for r in labels:
        if r["usable"] or r["lead_p"] >= 0.3 or r["rhythm_p"] >= 0.3:
            continue
        idx = set()
        for tid in (r["lead_id"], r["rhythm_id"]):
            idx |= set(lead_windows(net, f"{DATA}/{r['track_id']}/guitar/{tid}.wav"))
        out["blocked"][r["track_id"]] = sorted(int(i) for i in idx)
        print(f"{r['song']:30s} blocked windows: {len(idx)}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
