"""
Overfit test: can a loss teach BOTH outputs (esp. the lead) on a few fixed
crops? Trains a fresh lead/rhythm HTDemucs (pretrained backbone) on N fixed
MoisesDB crops with a given loss and prints per-output SI-SDR and level every
few steps. Memorising 4 crops is the easiest possible task, so a loss that
can't do it here won't do it on Kaggle. See CLAUDE.md 2026-10-18.

    python tools/overfit_loss_test.py v2|hybrid [steps] [seg_sec]
"""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from htdemucs_leadrhythm import build_leadrhythm_htdemucs  # noqa: E402
import moisesdb_dataset as mdb  # noqa: E402

DATA = "output/moisesdb_guitar"
TAU = 10 ** (-30 / 10)
ACTIVE_DB = -20.0


def _flat(x):
    return x.float().reshape(x.shape[0], -1)


def level_db(x, ref):
    return 10 * torch.log10((_flat(x) ** 2).sum(-1) / ((_flat(ref) ** 2).sum(-1) + 1e-12) + 1e-12)


def snr_mix(est, ref, mix):
    e, r, m = _flat(est), _flat(ref), _flat(mix)
    floor = TAU * (m ** 2).sum(-1)
    return 10 * torch.log10(((r ** 2).sum(-1) + floor) / (((e - r) ** 2).sum(-1) + floor))


def sisdr_flat(est, ref, eps=1e-8):
    e, r = _flat(est), _flat(ref)
    r = r - r.mean(-1, keepdim=True)
    e = e - e.mean(-1, keepdim=True)
    a = (e * r).sum(-1, keepdim=True) / ((r ** 2).sum(-1, keepdim=True) + eps)
    t = a * r
    q = 10 * torch.log10((t ** 2).sum(-1) / (((e - t) ** 2).sum(-1) + eps) + eps)
    return q.clamp(-40.0, 40.0)


def recon_v2(o, ref, mix, on):
    return -snr_mix(o, ref, mix)


LAMBDA_LVL = 0.5
LVL_DEADZONE = 3.0
LAMBDA_SIL = float(os.environ.get("LAMBDA_SIL", 1.0))   # weight on the not-playing silence term


def recon_hybrid(o, ref, mix, on):
    # playing: SI-SDR (no reward for going quiet) + bounded level match with a deadzone
    # not playing: v2's silence term (same math as -snr_mix on a silent ref)
    lvl = F.relu((level_db(o, ref) - 0).abs() - LVL_DEADZONE).clamp(max=30.0)
    active = -sisdr_flat(o, ref) + LAMBDA_LVL * lvl
    inactive = LAMBDA_SIL * -snr_mix(o, torch.zeros_like(ref), mix)
    return torch.where(on, active, inactive)


def loss_fn(kind, est, mix, lead, rhy):
    rec = recon_v2 if kind == "v2" else recon_hybrid
    o0, o1 = est[:, 0], est[:, 1]
    lead_on, rhy_on = level_db(lead, mix) > ACTIVE_DB, level_db(rhy, mix) > ACTIVE_DB
    both = lead_on & rhy_on
    same = (rec(o0, lead, mix, lead_on) + rec(o1, rhy, mix, rhy_on)) / 2
    swap = (rec(o0, rhy, mix, rhy_on) + rec(o1, lead, mix, lead_on)) / 2
    recon = torch.where(both, torch.minimum(same, swap), same).mean()
    q00, q01 = sisdr_flat(o0, lead), sisdr_flat(o0, rhy)
    bf = both.float()
    hinge = (F.relu(1.0 - (q00 - q01)) * bf).sum() / bf.sum().clamp(min=1)
    return recon + 3.0 * hinge


def main():
    kind = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    seg = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    mdb.SR = 44100
    labels = {r["track_id"]: r for r in json.load(open(f"{DATA}/role_labels.json")) if r["usable"]}
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    ds = mdb.MoisesLeadRhythmDataset(DATA, list(labels), seg_sec=seg, overlap_bias=0.7)
    # 3 both-active crops + 1 where the lead isn't playing
    crops, need_both, need_rest = [], 3, 1
    while need_both or need_rest:
        mix, lead, rhy = ds.sample_batch(1)
        lo, ro = level_db(lead, mix).item() > ACTIVE_DB, level_db(rhy, mix).item() > ACTIVE_DB
        if lo and ro and need_both:
            crops.append((mix, lead, rhy, "both")); need_both -= 1
        elif ro and not lo and need_rest:
            crops.append((mix, lead, rhy, "lead-rest")); need_rest -= 1
    model = build_leadrhythm_htdemucs()[0].train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    t0 = time.time()

    def report(step):
        model.eval()
        q0, q1, g0, g1, bz = [], [], [], [], []
        with torch.no_grad():
            for mix, lead, rhy, tag in crops:
                out = model(mix)
                if tag == "both":
                    q0.append(sisdr_flat(out[:, 0], lead).item()); q1.append(sisdr_flat(out[:, 1], rhy).item())
                    g0.append(level_db(out[:, 0], lead).item()); g1.append(level_db(out[:, 1], rhy).item())
                else:
                    bz.append(max(level_db(out[:, 0], mix).item(), -60))
        model.train()
        print(f"[{kind} sil={LAMBDA_SIL}] step {step:4d} ({time.time()-t0:5.0f}s)  lead SI-SDR {np.mean(q0):6.1f} lvl {np.mean(g0):6.1f}dB | "
              f"rhythm SI-SDR {np.mean(q1):6.1f} lvl {np.mean(g1):6.1f}dB | lead-rest out0 {np.mean(bz):6.1f}dB vs mix",
              flush=True)

    report(0)
    for step in range(1, steps + 1):
        mix, lead, rhy, _ = crops[step % len(crops)]
        opt.zero_grad()
        loss = loss_fn(kind, model(mix), mix, lead, rhy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 20 == 0:
            report(step)


if __name__ == "__main__":
    main()
