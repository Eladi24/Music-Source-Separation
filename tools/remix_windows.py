"""
Window lists for cross-song remix training (Stage 4 v5, 2026-10-19).

Remixing pairs one song's lead track with another song's rhythm track. Track
labels are mostly right in context (blind tests, CLAUDE.md 2026-10-19), but a
remix removes that context, so crops are restricted to windows where the
isolated track itself sounds like its role -- the case where the role
classifier agreed with the user's ear:
  - lead_ok:     lead-track windows rated lead (P(lead) > 0.8)
  - rhythm_bad:  rhythm-track windows rated lead (P(lead) > 0.5), never used as rhythm
Two-rhythm songs reuse tools/dbl_rhythm_blocked_windows.json.
Writes tools/remix_windows.json, which tools/build_stage4_notebook.py embeds.

    python tools/remix_windows.py
"""
import json
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
from train_role_classifier import RoleNet, SR, SEG  # noqa: E402

DATA = "output/moisesdb_guitar"
OUT = os.path.join(os.path.dirname(__file__), "remix_windows.json")
P_LEAD_OK, P_RHYTHM_BAD = 0.8, 0.5


def p_lead(net, path):
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torchaudio.functional.resample(torch.from_numpy(d.mean(1)), sr, SR)
    n = w.shape[0] // SEG
    segs = torch.stack([w[i * SEG:(i + 1) * SEG] for i in range(n)])
    pk = segs.abs().amax(1, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        return torch.softmax(net(segs / pk * 0.8), 1)[:, 1].numpy()


def main():
    net = RoleNet()
    net.load_state_dict(torch.load("output/role_classifier.pt", map_location="cpu"))
    net.eval()
    labels = [r for r in json.load(open(f"{DATA}/role_labels.json")) if r["usable"]]
    out = {"window_sec": SEG / SR, "lead_ok": {}, "rhythm_bad": {}}
    for r in labels:
        pl = p_lead(net, f"{DATA}/{r['track_id']}/guitar/{r['lead_id']}.wav")
        pr = p_lead(net, f"{DATA}/{r['track_id']}/guitar/{r['rhythm_id']}.wav")
        out["lead_ok"][r["track_id"]] = [int(i) for i in range(len(pl)) if pl[i] > P_LEAD_OK]
        out["rhythm_bad"][r["track_id"]] = [int(i) for i in range(len(pr)) if pr[i] > P_RHYTHM_BAD]
        print(f"{r['song']:30s} lead_ok {len(out['lead_ok'][r['track_id']]):4d}  "
              f"rhythm_bad {len(out['rhythm_bad'][r['track_id']]):4d}")
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
