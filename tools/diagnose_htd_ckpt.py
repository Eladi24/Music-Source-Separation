"""
Scale-free diagnostic for HTDemucs lead/rhythm checkpoints on the Stage 4
test songs (the 2 highest-overlap MoisesDB songs, same split as the notebook).

Separates content quality from output level, which the notebook's level-aware
`sep` score can't: per both-active crop it reports SI-SDR best-perm (content,
scale-invariant) and each output's level relative to its true target (gain).
See CLAUDE.md 2026-10-18.

    python tools/diagnose_htd_ckpt.py output/htd_v2_epoch1.pt output/htd_v2_epoch2.pt ...
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from htdemucs_leadrhythm import build_leadrhythm_htdemucs  # noqa: E402
from moisesdb_dataset import MoisesLeadRhythmDataset  # noqa: E402

DATA = "output/moisesdb_guitar"
N_CROPS = int(os.environ.get("N_CROPS", 24))


def sisdr(e, r, eps=1e-8):
    e, r = e.reshape(-1) - e.mean(), r.reshape(-1) - r.mean()
    a = (e @ r) / (r @ r + eps)
    t = a * r
    return 10 * torch.log10((t @ t) / ((e - t) @ (e - t) + eps) + eps).item()


def db(x, ref):
    return 10 * np.log10(((x ** 2).sum() / ((ref ** 2).sum() + 1e-12)).item() + 1e-12)


def main(paths):
    import json
    labels = {r["track_id"]: r for r in json.load(open(f"{DATA}/role_labels.json")) if r["usable"]}
    tmp = MoisesLeadRhythmDataset(DATA, list(labels), overlap_bias=0.0)
    test_ids = [s["track_id"] for s in sorted(tmp.songs, key=lambda s: s["overlap_frac"], reverse=True)[:2]]
    ds = MoisesLeadRhythmDataset(DATA, test_ids, overlap_bias=0.7)
    random.seed(0); np.random.seed(0)
    crops = [ds.sample_batch(1) for _ in range(N_CROPS)]
    model = build_leadrhythm_htdemucs()[0].eval()
    print(f"test songs: {[s['song'] for s in ds.songs]}  crops={N_CROPS}")
    # Reference: what "rhythm output = copy of the input mixture" would score on
    # these same crops. A checkpoint close to this has learned the copy shortcut.
    b_q, b_leak = [], []
    for mix, lead, rhy in crops:
        if db(lead[0], mix[0]) > -20 and db(rhy[0], mix[0]) > -20:
            b_q.append(sisdr(mix[0], rhy[0])); b_leak.append(sisdr(mix[0], lead[0]))
    print(f"{'[copy-the-mixture reference]':40s} out1->rhythm {np.mean(b_q):6.1f}dB  out1->lead (leak10) {np.mean(b_leak):6.1f}dB")
    for p in paths:
        model.load_state_dict(torch.load(p, map_location="cpu"))
        perm, role, g0, g1, buzz, q0, q1, cm, lk = [], [], [], [], [], [], [], [], []
        with torch.no_grad():
            for mix, lead, rhy in crops:
                out = model(mix)[0]
                o0, o1, L, R = out[0], out[1], lead[0], rhy[0]
                on_l, on_r = db(L, mix[0]) > -20, db(R, mix[0]) > -20
                if on_l and on_r:
                    same = (sisdr(o0, L) + sisdr(o1, R)) / 2
                    swap = (sisdr(o0, R) + sisdr(o1, L)) / 2
                    perm.append(max(same, swap)); role.append(sisdr(o0, L) > sisdr(o0, R))
                    g0.append(db(o0, L)); g1.append(db(o1, R))
                    q0.append(sisdr(o0, L)); q1.append(sisdr(o1, R))
                    cm.append(sisdr(o1, mix[0])); lk.append(sisdr(o1, L))
                if not on_l: buzz.append(max(db(o0, mix[0]), -60))
                if not on_r: buzz.append(max(db(o1, mix[0]), -60))
        f = lambda v: f"{np.mean(v):6.1f}" if v else "   nan"
        print(f"{os.path.basename(p):40s} SI-SDR best-perm {f(perm)}dB (out0->lead {f(q0)}, out1->rhythm {f(q1)})  role {np.mean(role):.2f}  "
              f"out-vs-target level: out0 {f(g0)}dB out1 {f(g1)}dB  buzz {f(buzz)}dB(n={len(buzz)})  [both-active n={len(perm)}]\n"
              f"{'':40s} out1 vs MIXTURE {f(cm)}dB  out1->lead (leak10) {f(lk)}dB")


if __name__ == "__main__":
    main(sys.argv[1:])
