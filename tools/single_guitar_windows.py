"""
Per-window role lists for the 82 MoisesDB songs with exactly ONE distorted
electric guitar (pulled 2026-10-19 to output/moisesdb_guitar_single/ by
tools/pull_moisesdb_guitar.py --n-distorted 1). These songs have no
lead/rhythm pair, but their single distorted track adds lead and rhythm
material for cross-song remixing:
  - lead_ok:   2s windows rated lead (P(lead) > 0.8) and playing
  - rhythm_ok: 2s windows rated rhythm (P(lead) < 0.2) and playing
Only confident windows are used, because a part heard alone can be judged
differently than in context (blind tests, CLAUDE.md 2026-10-19).
Writes tools/single_guitar_windows.json, for tools/build_stage4_notebook.py.

    python tools/single_guitar_windows.py
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

DATA = "output/moisesdb_guitar_single"
OUT = os.path.join(os.path.dirname(__file__), "single_guitar_windows.json")
P_LEAD_OK, P_RHYTHM_OK, ACTIVE_RMS = 0.8, 0.2, 0.01


def windows(net, path):
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = d.mean(1)
    w = 2 * sr
    n = len(mono) // w
    rms = np.array([np.sqrt(np.mean(mono[i * w:(i + 1) * w] ** 2)) for i in range(n)])
    x = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR)
    segs = torch.stack([x[i * SEG:(i + 1) * SEG] for i in range(min(n, x.shape[0] // SEG))])
    pk = segs.abs().amax(1, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        p = torch.softmax(net(segs / pk * 0.8), 1)[:, 1].numpy()
    act = rms[:len(p)] > ACTIVE_RMS
    return ([int(i) for i in range(len(p)) if act[i] and p[i] > P_LEAD_OK],
            [int(i) for i in range(len(p)) if act[i] and p[i] < P_RHYTHM_OK])


def main():
    net = RoleNet()
    net.load_state_dict(torch.load("output/role_classifier.pt", map_location="cpu"))
    net.eval()
    songs = json.load(open(f"{DATA}/manifest.json"))["songs"]
    out = {}
    for s in songs:
        dist = [t for t in s["guitar_tracks"] if t["trackType"] == "distorted_electric_guitar"]
        assert len(dist) == 1, s["song"]
        path = f"{DATA}/{s['track_id']}/guitar/{dist[0]['id']}.wav"
        lead, rhy = windows(net, path)
        out[s["track_id"]] = {"song": s["song"], "artist": s["artist"], "genre": s["genre"],
                              "file": dist[0]["id"], "lead_ok": lead, "rhythm_ok": rhy}
        print(f"{s['artist'][:18]:18s} {s['song'][:28]:28s} {s['genre'][:10]:10s} lead {len(lead):4d}  rhythm {len(rhy):4d}")
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
