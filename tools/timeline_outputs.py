"""
Song-agnostic lead/rhythm check for the Stage 4 HTDemucs line.

Each output's level is measured per 2s window relative to the input stem.
Solo windows come from --solo (by ear, e.g. "2:20-2:52,3:40-3:55"). Without
--solo, the tool lists the stretches where the FIRST checkpoint's lead output
switches on (> --on-db vs the input), as candidates to confirm by ear; scores
computed from those are circular for out0 but still meaningful for out1.
A working separator should give:
  - out0 (lead): loud in solo windows, quiet elsewhere (contrast, want high)
  - out1 (rhythm): about the same level in both (swell; a rise = solo bleed)

Why not the GuitarSet role classifier: tried first (2026-10-19). On The
Sentinel's full guitar stem it rated the whole 2:20-2:52 solo section as
rhythm (P(lead) 0.01-0.09) -- with a solo over dense rhythm, the rhythm
dominates. It cannot find solos in a mixed guitar stem.

    python tools/timeline_outputs.py --stem "output/01 Hot Blooded/guitar.wav" \
        output/stage4_local_test/hotblooded_v3_epoch5 output/stage4_local_test/hotblooded_v31_epoch3
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

WIN = 2.0          # seconds
ACTIVE_DB = -40.0  # window counts as "guitar playing" above this (dB vs the stem's loudest window)


def mono(path):
    """Mono, with DC and sub-20Hz removed. v3 epoch 5's rhythm output carries a
    -0.055 DC offset (the DC loophole, CLAUDE.md 2026-10-18); left in, it adds
    a constant level floor that flattens the swell and hides solo bleed."""
    d, sr = sf.read(path, dtype="float32", always_2d=True)
    sos = butter(2, 20, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, d.mean(1)).astype(np.float32), sr


def win_db(x, sr, n):
    w = int(WIN * sr)
    seg = x[:n * w].reshape(n, w)
    return 10 * np.log10((seg ** 2).mean(1) + 1e-12)


def parse_t(t):
    m, s = t.strip().split(":")
    return int(m) * 60 + float(s)


def fmt_t(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def runs(mask):
    """Contiguous True runs as (start_idx, end_idx_exclusive)."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+", help="output prefixes (..._out0_lead.wav / ..._out1_rhythm.wav)")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--solo", default=None, help='solo windows by ear, "m:ss-m:ss,m:ss-m:ss"')
    ap.add_argument("--on-db", type=float, default=-15.0,
                    help="without --solo: lead output this close to the input counts as 'on'")
    ap.add_argument("--timeline", action="store_true", help="print the per-window table")
    a = ap.parse_args()

    x, sr = mono(a.stem)
    n = int(len(x) // (WIN * sr))
    in_db = win_db(x, sr, n)
    active = in_db > in_db.max() + ACTIVE_DB
    t = np.arange(n) * WIN + WIN / 2
    if a.solo:
        solo = np.zeros(n, bool)
        for rng in a.solo.split(","):
            lo, hi = (parse_t(v) for v in rng.split("-"))
            solo |= (t >= lo) & (t < hi)
        src = "by ear (--solo)"
    else:
        o0, _ = mono(a.prefixes[0] + "_out0_lead.wav")
        solo = (win_db(o0, sr, n) - in_db) > a.on_db
        src = f"lead output of {os.path.basename(a.prefixes[0])} > {a.on_db:.0f}dB (confirm by ear)"
    lead_w = active & solo
    rhy_w = active & ~solo

    print(f"stem: {a.stem}  {len(x)/sr:.0f}s, {n} windows, {active.sum()} active")
    print(f"solo windows {src}: {lead_w.sum()} of {active.sum()}")
    print("solo stretches:", ", ".join(f"{fmt_t(s*WIN)}-{fmt_t(e*WIN)}" for s, e in runs(lead_w)) or "none")

    # Swell = how much louder a signal is in solo windows than in its own other
    # windows (absolute dB, medians). The input swells because the solo is added
    # on top; a clean rhythm output should not (~0dB), and an rhythm output that
    # swells as much as the input is carrying the whole solo.
    in_swell = np.median(in_db[lead_w]) - np.median(in_db[rhy_w])
    print(f"input swell during solos: {in_swell:+.1f}dB  (out1 swell: 0 = clean, ~{in_swell:+.1f} = full solo bleed)")
    print(f"\n{'checkpoint':40s} {'out0 solo':>10s} {'out0 other':>10s} {'out0 contrast':>13s} {'out1 swell':>10s}")
    print(f"{'':40s} {'(dB vs input)':>21s} {'(want high)':>13s} {'(want ~0)':>10s}")
    tables = {}
    for pre in a.prefixes:
        o0, _ = mono(pre + "_out0_lead.wav")
        o1, _ = mono(pre + "_out1_rhythm.wav")
        d1 = win_db(o1, sr, n)
        r0 = win_db(o0, sr, n) - in_db
        r1 = d1 - in_db
        tables[pre] = (r0, r1)
        # median, so a few misjudged windows don't dominate
        l0, h0 = np.median(r0[lead_w]), np.median(r0[rhy_w])
        sw1 = np.median(d1[lead_w]) - np.median(d1[rhy_w])
        print(f"{os.path.basename(pre):40s} {l0:10.1f} {h0:10.1f} {l0-h0:13.1f} {sw1:+10.1f}")

    if a.timeline:
        names = [os.path.basename(p)[-12:] for p in a.prefixes]
        print(f"\n{'time':>5s} {'in dB':>6s} {'solo':>4s}  " + "  ".join(f"{nm:>13s}" for nm in names))
        print(f"{'':>5s} {'':>6s} {'':>4s}  " + "  ".join(f"{'out0 / out1':>13s}" for _ in names))
        for i in range(n):
            if not active[i]:
                continue
            mark = "  S " if lead_w[i] else "    "
            cols = "  ".join(f"{tables[pp][0][i]:6.1f}/{tables[pp][1][i]:6.1f}" for pp in a.prefixes)
            print(f"{fmt_t(i*WIN):>5s} {in_db[i]:6.1f} {mark}  {cols}")


if __name__ == "__main__":
    main()
