"""Training-data distortion chain, with the aliasing defect fixed.

Measured 2026-10-14: the chain `lead4_epoch5.pt` was trained on runs
`Distortion(drive_db=25)` at SR=16000 with no oversampling. Waveshaping
generates harmonics far above the 8kHz Nyquist; they fold back down as
inharmonic aliasing, and the chain's LowpassFilter(6500) runs AFTER the
distortion so it cannot remove anything already folded below 6500Hz.
Measured corruption vs a 4x-oversampled reference: -25.2dB (solo clip) /
-27.4dB (comp clip) -- i.e. 4-6% of target energy is aliasing garbage,
present in the ground-truth targets as well as the mixtures, so the model
was trained to reproduce it.

Fix: do the whole chain at OS x the base rate and resample properly on the
way in and out. Filter cutoffs are specified in Hz so they are unchanged by
running at a higher rate. resample_poly applies a real anti-aliasing filter,
unlike the np.interp resampling used in load_wav (measured harmless for
GuitarSet at -38/-47dB, since it has little content above 8kHz, but still
the wrong tool).

CHAIN_LEGACY is the exact pre-Update-4 chain, recovered from the
never-patched Stage 1 notebook -- kept so the two can be compared and so
`lead4_epoch5.pt`'s actual training distribution stays reproducible.
"""
import numpy as np
from scipy.signal import resample_poly
from pedalboard import (Pedalboard, Distortion, LowpassFilter, HighpassFilter,
                        Compressor)

SR = 16000
OS = 4          # oversampling factor for the waveshaper


def _chain():
    """The pre-Update-4 chain lead4_epoch5.pt was trained on. Note Update 4's
    amp-realism variant (asymmetric clipping + tone stack + cab IR) is a
    concluded negative result -- it caused a real-audio routing-stability
    regression -- so this, not that, is the base to build on."""
    return Pedalboard([
        Distortion(drive_db=25),
        LowpassFilter(cutoff_frequency_hz=6500),   # < Nyquist at SR, and at SR*OS
        HighpassFilter(cutoff_frequency_hz=100),
        Compressor(threshold_db=-20, ratio=4),
    ])


def distort_legacy(x, drive=1.0, sr=SR):
    """What the model was actually trained on. Aliases."""
    y = _chain()(np.asarray(x, dtype=np.float32) * drive, sr)
    assert np.isfinite(y).all(), 'non-finite output -- check cutoffs vs Nyquist'
    return np.asarray(y, dtype=np.float32)


def distort(x, drive=1.0, sr=SR, os=OS):
    """Oversampled: the same chain, without the aliasing."""
    x = np.asarray(x, dtype=np.float64) * drive
    up = resample_poly(x, os, 1)
    y = _chain()(up.astype(np.float32), sr * os)
    down = resample_poly(np.asarray(y, dtype=np.float64), 1, os)
    out = np.asarray(down[:len(x)], dtype=np.float32)
    assert np.isfinite(out).all(), 'non-finite output -- check cutoffs vs Nyquist'
    return out


def resample_to(x, sr_in, sr_out):
    """Proper anti-aliased resampling, replacing load_wav's bare np.interp."""
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    g = gcd(int(sr_in), int(sr_out))
    return np.asarray(resample_poly(np.asarray(x, dtype=np.float64),
                                    int(sr_out)//g, int(sr_in)//g), dtype=np.float32)


# --- v2 chain: spectrally matched to real electric guitar -------------------
# Measured 2026-10-14: the legacy chain's output holds ~1.3% of its energy at
# 2-4kHz and 54% at 250-500Hz, where real electric guitar in a real mix holds
# ~20% and ~26%. That mismatch is why the model never learned to allocate
# real treble content and starves the rhythm output of it (out1 5.1% vs the
# input's 24.5% at 2-4kHz on The Sentinel).
#
# Neither a better dataset nor more gain alone fixes it: Guitar-TECHS (real
# mic'd amp, CC BY 4.0) measured WORSE than GuitarSet (0.38% at 2-4kHz), and
# pushing the legacy chain to drive_db=65 only reached 4.85%. The gap is tone
# shaping, not recording quality or gain -- real metal tone has a deliberate
# low-mid scoop and presence boost, which the legacy chain has none of.
#
# Parameters found by grid search against the MEAN profile of the two real
# songs (aiming between them, not at one), scored on L1 distance plus a
# penalty for variance across source clips. That variance penalty matters:
# Update 4's amp-realism chain scored a good L1 on GuitarSet solo (42.5) but
# collapsed to 99% low-frequency energy on Guitar-TECHS DI -- wildly
# source-dependent, a newly-identified and plausible mechanism for the
# routing instability that appeared by epoch 8 in that experiment and was
# absent at epoch 1.
#
# Result vs the current chain (mean over 5 clips, share of sub-6500Hz energy):
#            20-250  250-500  500-1k   1-2k   2-4k    L1   inconsistency
#   target     22.9     26.0    16.4   13.3   20.1   0.0        --
#   v2         21.5     20.6    14.8   17.9   21.5  38.8      6.44
#   legacy     24.5     54.0    16.7    3.2    1.3  87.3      8.65
#
# NOT YET APPROVED: matching an average band profile is not the same as
# sounding like real guitar, and this is a deliberately aggressive setting
# (drive 45, +9dB at 3kHz). Gated on the user's listening per the project's
# hard rule.
from pedalboard import PeakFilter

# APPROVED by the user 2026-10-14 after listening (variant "C"). An earlier
# candidate at drive_db=45/presence+9 measured slightly better (L1 38.8) but
# the user judged it "much more gainy" and rejected it; drive 45 was only
# ever the grid search's cheapest route to the brightness target, not a goal.
# Holding drive near the approved tone and taking brightness from EQ instead
# costs a little L1 and buys a tone the user signed off on.
V2 = dict(drive_db=30, scoop_hz=300, scoop_db=-6, presence_hz=3000, presence_db=15)


def _chain_v2(p=V2):
    return Pedalboard([
        Distortion(drive_db=p['drive_db']),
        HighpassFilter(cutoff_frequency_hz=90),
        PeakFilter(cutoff_frequency_hz=p['scoop_hz'], gain_db=p['scoop_db'], q=0.8),
        PeakFilter(cutoff_frequency_hz=p['presence_hz'], gain_db=p['presence_db'], q=0.7),
        LowpassFilter(cutoff_frequency_hz=6500),    # < Nyquist at SR and at SR*OS
        Compressor(threshold_db=-20, ratio=4),
    ])


def distort_v2(x, drive=1.0, sr=SR, os=OS, p=V2):
    """Oversampled (no aliasing) AND spectrally matched. Combines both
    2026-10-14 fixes."""
    x = np.asarray(x, dtype=np.float64) * drive
    up = resample_poly(x, os, 1)
    y = _chain_v2(p)(up.astype(np.float32), sr * os)
    out = np.asarray(resample_poly(np.asarray(y, dtype=np.float64), 1, os)[:len(x)],
                     dtype=np.float32)
    assert np.isfinite(out).all(), 'non-finite output -- check cutoffs vs Nyquist'
    return out
