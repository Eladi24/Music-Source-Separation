"""
Assembles output/stage4_kaggle/Stage4_HTDemucs_MoisesDB.ipynb (+ a Colab
twin) -- the HTDemucs lead/rhythm fine-tune on real MoisesDB guitar pairs.
See CLAUDE.md "The step change" and the 2026-10-15 session for the full
rationale. Run this script to (re)generate the notebooks; edit the cell
source strings below, not the .ipynb files directly, so the two platform
twins stay in sync.
"""
import json
import os

MD = "markdown"
CODE = "code"


def cell(cell_type, source):
    lines = source.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": src,
        **({"outputs": [], "execution_count": None} if cell_type == CODE else {}),
    }


INTRO_MD = """\
# Stage 4: HTDemucs lead/rhythm fine-tune on real MoisesDB guitar pairs

**Run on Kaggle:** Settings -> Accelerator -> **GPU T4 x2**, Settings ->
**Internet: On**. Then Run All.

## Why this, and why now

Four consecutive quality-fix attempts on the Stage 3 lineage (`w_comp`,
STFT loss, amp-realism data swap, band-balance loss) all landed on one
measured trade curve between the rhythm output's treble and solo bleed --
see CLAUDE.md "Update 6" and "The step change". The structural constraints
behind that curve: 5M parameters, a 6.5kHz bandwidth ceiling, 3000
synthetic mixtures built from 360 acoustic GuitarSet clips run through a
synthetic distortion chain, and chunk-local inference with no cross-chunk
memory. None of those are things a loss function or an epoch count can
fix.

This stage attacks three of those at once: pretrained **HTDemucs** (42M
params, trained on real commercial music, natively 44.1kHz -- no
synthetic distortion chain, no bandwidth ceiling) fine-tuned on
**MoisesDB** -- real multitrack recordings with genuinely separate
distorted-electric-guitar tracks from the same performance. 31/240
MoisesDB songs have >=2 such tracks (confirmed via remote zip metadata
inspection, no 82.7GB download needed); 18 of those auto-label into a
clear lead/rhythm pair via a classifier trained on GuitarSet's own
`_solo_mix`/`_comp_mix` labels (97.2% held-out test accuracy). Listening-
approved by the user on both the raw tracks and the derived mixtures
(2026-10-15), per the standing hard gate on new/manipulated audio data.

## Architecture adaptation

Only **two** layers in HTDemucs encode the source count -- confirmed by
reading `demucs/htdemucs.py`: `chin_z = audio_channels * len(sources) *
(2 if cac else 1)` is computed once, only for the final decoder layer.
Everything else (the encoder, the cross-transformer, the rest of the
decoder) is source-count-independent and carries over from the pretrained
checkpoint unchanged:

- `decoder[-1].conv_tr`: `ConvTranspose2d(48, 16, ...)` -> resized to 8
  (2 sources x 2 channels x 2 complex re/im), freshly initialized.
- `tdecoder[-1].conv_tr`: `ConvTranspose1d(48, 8, ...)` -> resized to 4
  (2 sources x 2 channels), freshly initialized.

This is the same warm-start pattern that worked for the original
lead/rhythm pivot (Stage 3): a new, randomly-initialized output head
learns the routing while the pretrained backbone's real-music
representations are preserved. Validated locally (forward: ~1GB RSS,
correct `[1,2,2,T]` output, finite; backward+optimizer step: ~2GB RSS at
2s segments, ~4GB at 4s, finite gradients) before this notebook was
written. Batch size 2 OOM-killed locally (confirmed via `dmesg`) --
same memory wall this project hits on every real training attempt, just
reached one batch size earlier since this model is 8x bigger than the
ConvTasNet used in Stage 1-3. Real training needs this notebook's GPU.

## Data pipeline differences from every prior stage

No synthetic distortion chain at all -- this is real distorted electric
guitar from real mixes, so there's nothing to approve-by-listening beyond
the raw tracks and mixtures (already done). No fixed-example-count
precompute either: real audio is read directly off disk with on-the-fly
random crops each access, so (unlike GuitarSet's synthetic pipeline)
there's no "same N examples repeat every epoch" tradeoff.

**New finding this session, worth tracking:** not every song with a clear
lead/rhythm classifier split actually has the two guitars playing
*simultaneously*. Measured "overlap fraction" (fraction of 2s windows
where both tracks are active) ranges from 0% (`Is It All On Me` -- the
guitars alternate, never overlap) to 37% (`Firefly - Electric`). Crop
sampling here is biased 70/30 toward measured-overlap windows, since
non-overlapping sections don't teach the model anything about separating
simultaneous content (they still contribute single-source reconstruction
signal, which is why they're not excluded entirely).

## v2 (2026-10-18): the silent-source fix

v1 normalised each guitar track separately, which boosted the amp hiss in
a not-playing track's "silent" stretch by 40-90dB and made it the
target. 11% of source-crops were affected. That was very likely the cause
of v1's non-solo buzzing, its negative best_perm, and its NaNs. v2 fixes
the data (one joint gain per pair, so silence stays silent) and the loss
(mixture-thresholded SNR, defined when a target is silent; hinge/leak only
on both-active crops), and starts fresh from pretrained HTDemucs with
warmup + cosine LR and gradient clipping. Training samples were
listening-approved before this run, per the hard gate. See CLAUDE.md
2026-10-18.

## v3 (2026-10-18): hybrid loss + two-rhythm-guitar songs

v2 run 1 (20 epochs) fixed the silence problem, but the **lead output never
learned**: effectively silent all run. On The Sentinel, the rhythm output
kept only ~12-20% of the guitar in non-solo passages, which sounded
blurry/underwater, and was bright and well isolated only during the solos.
Two fixes:

1. **Hybrid loss.** SI-SDR for playing sources (no reward for going quiet) +
   a level-match term (+-3dB deadzone) + v2's silence term x4 for
   non-playing sources. Overfit test: lead +2.6 / rhythm +4.8dB vs v2's
   -2.9 / -1.4dB, with near-identical silence (-29.4 vs -31.3dB).
2. **Two-rhythm-guitar songs.** 9 MoisesDB songs where both guitars play
   rhythm (excluded before as "no clear lead/rhythm split"), trained with
   target lead = silence and rhythm = both guitars, at 25% of crops. This is
   The Sentinel's twin-rhythm non-solo case (confirmed by the user), which
   training never contained. 2 more such songs are held out for a dedicated
   eval line. Listening-approved before the run.

## What to watch

- **`sep`** -- both-active separation, mixture-thresholded SNR (capped
  ~30dB). New metric, **not comparable to v1's SI-SDR best_perm**.
- **`role`** -- fraction of both-active crops where out0 is closer to the
  lead than to the rhythm. ~50% = not working.
- **`leak01` / `leak10`** -- SI-SDR of each output against the WRONG
  source, both-active crops only. Lower is better.
- **`buzz`** -- how loud an output is (dB vs the mix) when its own target
  isn't playing. This measures v1's non-solo buzzing directly: it should
  fall toward -30dB.
- **`lead:` / `rhythm:` lines** -- per-output SI-SDR (content, ignores
  level) and level vs the true target. **The lead line is the one that
  failed in v2 run 1**: watch its SI-SDR climb and its level stay near 0dB,
  not drift to -25dB.
- **`two-rhythm:` line** (v3) -- held-out songs with no lead: lead output
  should stay far below the mix, and rhythm SI-SDR should be high with level
  near 0dB.
- Band-balance loss stays off (one change at a time).
- Real-song test: `tools/realsong_test_stereo.py`, locally, on downloaded
  checkpoints.
"""

SETUP_CODE = """\
!pip install -q -U demucs soundfile
"""

CONFIG_CODE = """\
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

SR = 44100
SEG_SEC = 4.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DP = torch.cuda.device_count() > 1
BATCH = 8 if USE_DP else 4   # conservative starting point -- this model is 8x the ConvTasNet's size
N_EPOCHS = 20
LR_PEAK = 2e-4       # v1: 1e-4 stalled, 5e-4 diverged, 2e-4 progressed (flat 2e-4 later NaN'd -> schedule below)
GRAD_CLIP = 5.0
CKPT_PREFIX = "htd_v3"
LAMBDA_SIL = 4.0     # v3: weight on the not-playing silence term (overfit test: x4 kept v2's silence, -29 vs -31dB)
LAMBDA_LVL = 0.5     # v3: level-match term for playing sources, outside a +-3dB deadzone
LVL_DEADZONE = 3.0
DOUBLE_RHYTHM_FRAC = 0.25   # v3: share of training crops drawn from two-rhythm-guitar songs (see split cell)
LAMBDA_HINGE = 3.0   # Stage 2/3 found this is where routing actually gets a real gradient
LAMBDA_LEAK = 1.0    # conservative starting weight, same philosophy as every prior leak-loss run
LAMBDA_BAND = 0.0    # deliberately off this run -- see markdown above

print(f"device={DEVICE} n_gpus={torch.cuda.device_count()} batch={BATCH}")
"""

DATA_LOCATE_CODE = """\
# Locate the uploaded MoisesDB-guitar-slice dataset. Upload output/moisesdb_guitar/
# (6.07GB, from the local pull via tools/pull_moisesdb_guitar.py) as a private
# Kaggle Dataset once, then attach it here via "+ Add Input". Kaggle slugifies
# dataset names (same gotcha as CP_EP50 in Stage 1) -- discover the real path
# instead of guessing it.
DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "manifest.json" in files and "role_labels.json" in files:
        DATA_DIR = root
        break
assert DATA_DIR, "Couldn't find the moisesdb_guitar dataset under /kaggle/input -- attach it via + Add Input."
print("DATA_DIR:", DATA_DIR)
"""

MODEL_CODE = """\
# --- HTDemucs adapted for 2-source lead/rhythm separation ------------------
# Only the two final output layers encode source count (verified by reading
# demucs/htdemucs.py: chin_z = audio_channels * len(sources) * (2 if cac else 1)
# is computed once, only for the LAST decoder layer). Everything upstream --
# encoder, cross-transformer, the rest of the decoder -- carries the pretrained
# real-music weights over unchanged. Same warm-start philosophy as every prior
# stage's checkpoint continuation, just applied to a different checkpoint.
from demucs.pretrained import get_model

def build_leadrhythm_htdemucs(sources=("lead", "rhythm")):
    bag = get_model("htdemucs")
    model = bag.models[0]
    n_src = len(sources)
    audio_ch = model.audio_channels
    cac_mult = 2 if model.cac else 1

    old_freq = model.decoder[-1].conv_tr
    new_freq = nn.ConvTranspose2d(old_freq.in_channels, audio_ch * n_src * cac_mult,
                                   kernel_size=old_freq.kernel_size, stride=old_freq.stride)
    model.decoder[-1].conv_tr = new_freq

    old_time = model.tdecoder[-1].conv_tr
    new_time = nn.ConvTranspose1d(old_time.in_channels, audio_ch * n_src,
                                   kernel_size=old_time.kernel_size, stride=old_time.stride)
    model.tdecoder[-1].conv_tr = new_time

    model.sources = list(sources)
    return model


model = build_leadrhythm_htdemucs()
backbone_params = sum(p.numel() for n, p in model.named_parameters()
                       if "decoder.3.conv_tr" not in n and "tdecoder.3.conv_tr" not in n)
head_params = sum(p.numel() for n, p in model.named_parameters()) - backbone_params
print(f"backbone (pretrained): {backbone_params:,}  new head: {head_params:,}")

model = model.to(DEVICE)
core = model
if USE_DP:
    model = nn.DataParallel(model)
"""

LOSS_CODE = """\
# --- Loss (v2, 2026-10-18). v1 used per-channel SI-SDR for everything,
# which breaks on real multitracks: when one guitar isn't playing, its target
# is silence, and SI-SDR against silence is about -80dB whatever the model
# outputs. It can't reward "be quiet" (it ignores level), and its gradient is
# garbage there. That plausibly caused v1's buzzing, negative best_perm and
# NaNs (see CLAUDE.md 2026-10-18).
#
# Reconstruction now uses the mixture-thresholded SNR from MixIT/FUSS
# (Wisdom et al.), scored over BOTH channels jointly (some tracks are panned
# hard, down to -19dB on one side):
#     score = 10log10((|ref|^2 + tau|mix|^2) / (|est-ref|^2 + tau|mix|^2))
# Active ref: ~ordinary SNR, capped ~30dB. Silent ref: 0dB when the output is
# silent, toward -30dB as the output gets as loud as the mix. Bounded both
# ways, defined everywhere, and level-aware -- fine because v2's data keeps the
# true joint level (see dataset cell).
#
# The routing hinge and leak penalty stay SI-SDR (their Stage 3-proven form),
# but are only computed on crops where BOTH sources are active -- the only
# case where "out0 is closer to lead than to rhythm" means anything.
TAU = 10 ** (-30 / 10)   # SNR cap 30dB below the mixture
ACTIVE_DB = -20.0        # source quieter than this (vs the mix) counts as not playing


def _flat(x):
    return x.float().reshape(x.shape[0], -1)   # (B, C*T), fp32 -- sums of squares overflow fp16


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
    return q.clamp(-40.0, 40.0)   # beyond +-40dB is meaningless for routing, and unbounded otherwise


def level_db(x, mix):
    return 10 * torch.log10((_flat(x) ** 2).sum(-1) / ((_flat(mix) ** 2).sum(-1) + 1e-12) + 1e-12)


def _mmean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def recon_hybrid(o, ref, mix, on):
    \"\"\"v3 reconstruction term, per example (lower is better).
    v2 used snr_mix for everything. Its level-aware score has a "turn it down
    when unsure" optimum, and in v2 run 1 the lead output did exactly that:
    effectively silent for all 20 epochs (CLAUDE.md 2026-10-18). Now:
      playing source:     -SI-SDR (no reward for going quiet)
                          + LAMBDA_LVL * level error beyond a +-3dB deadzone
                            (so outputs can't drift loud like v1, +11.5dB)
      not-playing source: LAMBDA_SIL * v2's silence term
    Overfit test (tools/overfit_loss_test.py, 4 crops, 160 steps): lead +2.6 /
    rhythm +4.8dB SI-SDR vs v2's -2.9 / -1.4dB, levels within +-1.5dB, resting-
    lead output -29.4dB vs v2's -31.3dB.\"\"\"
    lvl = F.relu(level_db(o, ref).abs() - LVL_DEADZONE).clamp(max=30.0)
    active = -sisdr_flat(o, ref) + LAMBDA_LVL * lvl
    inactive = LAMBDA_SIL * -snr_mix(o, torch.zeros_like(ref), mix)
    return torch.where(on, active, inactive)


def combined_loss(est, MIX, LEAD, RHYTHM, margin=1.0, leak_margin=None, lambda_leak=0.0):
    o0, o1 = est[:, 0], est[:, 1]
    lead_on = level_db(LEAD, MIX) > ACTIVE_DB
    rhy_on = level_db(RHYTHM, MIX) > ACTIVE_DB
    both = (lead_on & rhy_on).float()

    l_same = (recon_hybrid(o0, LEAD, MIX, lead_on) + recon_hybrid(o1, RHYTHM, MIX, rhy_on)) / 2
    l_swap = (recon_hybrid(o0, RHYTHM, MIX, rhy_on) + recon_hybrid(o1, LEAD, MIX, lead_on)) / 2
    # `sep` metric kept on v2's SNR score so it stays comparable with v2 run 1's log
    with torch.no_grad():
        s00, s11 = snr_mix(o0, LEAD, MIX), snr_mix(o1, RHYTHM, MIX)
        s01, s10 = snr_mix(o0, RHYTHM, MIX), snr_mix(o1, LEAD, MIX)
        sep_same, sep_swap = (s00 + s11) / 2, (s01 + s10) / 2
    # PIT only where both sources play. Where only one plays, PIT would let the
    # model swap slots for free (rhythm in out0 scores the same) and the hinge
    # is masked, so nothing would enforce routing in non-solo passages. There
    # the assignment is fixed (lead->out0, rhythm->out1). The score is bounded,
    # so this doesn't recreate Stage 2's directed-loss collapse:
    # correct < collapse-to-mixture < wrong route (re-checked for the v3 terms).
    pit_loss = torch.where(both > 0, torch.minimum(l_same, l_swap), l_same).mean()

    q00, q01 = sisdr_flat(o0, LEAD), sisdr_flat(o0, RHYTHM)
    q10, q11 = sisdr_flat(o1, LEAD), sisdr_flat(o1, RHYTHM)
    hinge = _mmean(F.relu(margin - (q00 - q01)), both)
    if lambda_leak > 0 and leak_margin is not None:
        leak = (_mmean(F.relu(q01 - leak_margin), both) + _mmean(F.relu(q10 - leak_margin), both)) / 2
    else:
        leak = torch.zeros((), device=est.device)

    m = {
        "sep": _mmean(torch.maximum(sep_same, sep_swap), both),   # both-active separation, dB (v2's SNR)
        "role": _mmean((q00 > q01).float(), both),
        # Per-output content quality (SI-SDR, ignores level) and level vs the true
        # target, both-active crops. Added after v2 run 1, where an averaged score
        # hid that the rhythm output learned (-2dB) while the lead output stayed
        # near-silent and got worse (-28dB). See CLAUDE.md 2026-10-18.
        "lead_q": _mmean(q00.detach(), both), "rhy_q": _mmean(q11.detach(), both),
        "lead_lvl": _mmean(level_db(o0, LEAD).detach(), both),
        "rhy_lvl": _mmean(level_db(o1, RHYTHM).detach(), both),
        "leak01": _mmean(q01.detach(), both), "leak10": _mmean(q10.detach(), both),
        # "buzz": how loud an output is when its own target isn't playing (dB vs the
        # mix; lower is better, -30 is the floor the loss asks for). This is v1's
        # non-solo-passage buzzing, measured directly.
        "buzz_sum": (level_db(o0, MIX).detach().clamp(min=-60) * (~lead_on)).sum()
                    + (level_db(o1, MIX).detach().clamp(min=-60) * (~rhy_on)).sum(),
        "buzz_n": (~lead_on).sum() + (~rhy_on).sum(),
        "n_both": both.sum(),
        # rhythm output over ALL crops where the rhythm plays, for the two-rhythm eval
        "rhy_q_on": _mmean(q11.detach(), rhy_on.float()),
        "rhy_lvl_on": _mmean(level_db(o1, RHYTHM).detach(), rhy_on.float()),
    }
    return pit_loss, hinge, leak, m


def leak_baseline(model, dataset, n=20):
    \"\"\"Self-calibrate LEAK_MARGIN from the model's own init leakage on
    both-active crops (margin = min(baseline leak01, leak10) - 2dB).\"\"\"
    model.eval()
    l01, l10, k = 0.0, 0.0, 0
    with torch.no_grad():
        while k < n:
            mix, lead, rhythm = dataset.sample_batch(1)
            mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
            _, _, _, m = combined_loss(model(mix), mix, lead, rhythm)
            if m["n_both"].item() > 0:
                l01 += m["leak01"].item(); l10 += m["leak10"].item(); k += 1
    model.train()
    return min(l01 / n, l10 / n) - 2.0
"""

DATASET_CODE = """\
# --- Streaming dataset (v2, 2026-10-18): real audio, on-the-fly random
# crops, biased toward measured overlap windows. Same as tools/moisesdb_dataset.py.
#
# THE v1 BUG THIS FIXES: v1 RMS-normalised each source SEPARATELY to 0.1 and
# only skipped it below RMS 1e-6. Real tracks' "silent" stretches hold amp
# hiss/hum/bleed at ~1e-4..1e-3, which got boosted 40-90dB to full loudness and
# used as the target -- 11% of source-crops >30dB boosted at overlap_bias=0.7,
# 18% at 0.3. v2 applies ONE gain to the pair (from the mixture), so a silent
# source stays silent and the model learns to output silence for it. Natural
# lead/rhythm levels in overlap regions already span about +-6dB around
# balanced (measured), so there is no loudness shortcut to remove.
MIX_RMS = 0.1
SILENT_MIX_RMS = 1e-3   # raw crops quieter than this (both guitars resting) are re-drawn


def find_active_windows(path, win_sec=2.0):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    win = int(win_sec * sr)
    n = len(mono) // win
    rms = np.array([np.sqrt(np.mean(mono[i*win:(i+1)*win] ** 2)) for i in range(n)])
    return rms, sr, len(mono)


class MoisesLeadRhythmDataset:
    def __init__(self, data_dir, track_ids, labels_by_id, seg_sec=SEG_SEC,
                 overlap_bias=0.7, active_thresh=0.01, double_rhythm=False, blocked=None):
        # double_rhythm (v3): both tracks are rhythm parts. Target is lead =
        # silence, rhythm = both guitars -- the "two rhythm guitars, nobody
        # soloing" case that dominates The Sentinel's non-solo passages and was
        # absent from v1/v2 training.
        self.seg_sec = seg_sec
        self.overlap_bias = overlap_bias
        self.double_rhythm = double_rhythm
        blocked = blocked or {}   # track_id -> 2s window indices never to sample (lead-rated)
        self.songs = []
        for tid in track_ids:
            r = labels_by_id[tid]
            lead_path = os.path.join(data_dir, tid, "guitar", r["lead_id"] + ".wav")
            rhythm_path = os.path.join(data_dir, tid, "guitar", r["rhythm_id"] + ".wav")
            lead_rms, sr, n_lead = find_active_windows(lead_path)
            rhythm_rms, sr2, n_rhythm = find_active_windows(rhythm_path)
            assert sr == sr2 == SR
            n = min(len(lead_rms), len(rhythm_rms))
            both = np.where((lead_rms[:n] > active_thresh) & (rhythm_rms[:n] > active_thresh))[0]
            bl = set(blocked.get(tid, []))
            both = np.array([i for i in both if not ({i - 1, i, i + 1} & bl)], dtype=int)
            self.songs.append({
                "blocked": bl,
                "track_id": tid, "artist": r["artist"], "song": r["song"],
                "lead_path": lead_path, "rhythm_path": rhythm_path,
                "n_samples": min(n_lead, n_rhythm), "overlap_windows": both,
                "overlap_frac": len(both) / max(1, n),
            })

    def _clear(self, song, start, seg_n):
        if not song["blocked"]:
            return True
        w0, w1 = start // (2 * SR), (start + seg_n - 1) // (2 * SR)
        return not (set(range(w0, w1 + 1)) & song["blocked"])

    def _sample_start(self, song):
        seg_n = int(self.seg_sec * SR)
        max_start = max(1, song["n_samples"] - seg_n)
        for _try in range(50):
            if len(song["overlap_windows"]) > 0 and random.random() < self.overlap_bias:
                center = int(random.choice(song["overlap_windows"]) * 2.0 * SR) + SR
                start = max(0, min(max_start, center - seg_n // 2))
            else:
                start = random.randint(0, max_start)
            if self._clear(song, start, seg_n):
                break
        return start

    def _read_crop(self, song, start, seg_n):
        out = []
        for key in ("lead_path", "rhythm_path"):
            x, _ = sf.read(song[key], dtype="float32", always_2d=True, start=start, frames=seg_n)
            if x.shape[0] < seg_n:
                x = np.pad(x, ((0, seg_n - x.shape[0]), (0, 0)))
            out.append(x)
        return out

    def sample_one(self):
        seg_n = int(self.seg_sec * SR)
        for _try in range(20):
            song = random.choice(self.songs)
            lead, rhythm = self._read_crop(song, self._sample_start(song), seg_n)
            mix_rms = np.sqrt(np.mean((lead + rhythm) ** 2))
            if mix_rms >= SILENT_MIX_RMS:
                break
        if self.double_rhythm:
            lead, rhythm = np.zeros_like(lead), lead + rhythm
        g = MIX_RMS / max(mix_rms, 1e-8)       # one gain for the pair
        peak = np.abs(lead + rhythm).max() * g
        if peak > 0.95:
            g *= 0.95 / peak
        return (lead * g).T, (rhythm * g).T

    def sample_batch(self, batch_size):
        return _stack([self.sample_one() for _ in range(batch_size)])


def _stack(items):
    lead_t = torch.from_numpy(np.stack([a for a, _ in items])).float()
    rhythm_t = torch.from_numpy(np.stack([b for _, b in items])).float()
    return lead_t + rhythm_t, lead_t, rhythm_t


class MixedDataset:
    \"\"\"Each crop comes from `extra` with probability `frac`, else from `main`.\"\"\"
    def __init__(self, main, extra, frac):
        self.main, self.extra, self.frac = main, extra, frac

    def sample_batch(self, batch_size):
        return _stack([(self.extra if random.random() < self.frac else self.main).sample_one()
                       for _ in range(batch_size)])
"""

SPLIT_CODE = """\
# --- Song-level train/val/test split, biased so val/test actually have
# overlap content worth measuring (a song with 0% overlap, like "Is It All
# On Me", can't tell you anything about separation quality).
with open(os.path.join(DATA_DIR, "role_labels.json")) as f:
    labels = json.load(f)
usable = {r["track_id"]: r for r in labels if r["usable"]}

tmp_ds = MoisesLeadRhythmDataset(DATA_DIR, list(usable.keys()), usable, overlap_bias=0.0)
by_overlap = sorted(tmp_ds.songs, key=lambda s: s["overlap_frac"], reverse=True)
for s in by_overlap:
    print(f"  overlap={s['overlap_frac']:.2f}  {s['artist']} - {s['song']}")

test_ids = [s["track_id"] for s in by_overlap[0:2]]
val_ids = [s["track_id"] for s in by_overlap[2:4]]
train_ids = [s["track_id"] for s in by_overlap[4:]]
print(f"\\ntrain={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

val_ds = MoisesLeadRhythmDataset(DATA_DIR, val_ids, usable)
test_ds = MoisesLeadRhythmDataset(DATA_DIR, test_ids, usable)

# v3: two-rhythm-guitar songs -- the ones v1/v2 excluded as "no clear lead/rhythm
# split", with BOTH tracks rated rhythm-like by the role classifier (mean
# P(lead) < 0.3; drops "Broken" at 0.35 and "Lifeboat", whose tracks are both
# lead-like). 11 songs; 2 held out (different artists) for a dedicated eval.
# Listening-approved before this run, per the hard gate.
dbl = {r["track_id"]: r for r in labels
       if not r["usable"] and r["lead_p"] < 0.3 and r["rhythm_p"] < 0.3}
# Some of these tracks still have short lead-like stretches (melody lines,
# fills; up to 13-15% of windows in 3 songs). Windows the role classifier rates
# lead (P>0.5) on either track are never sampled. Computed locally by
# tools/dbl_rhythm_windows.py and embedded here, so the Kaggle dataset needn't change.
DBL_BLOCKED = __DBL_BLOCKED__
DBL_EVAL_SONGS = {"Dear You", "Snake Eyes"}
dbl_eval_ids = [t for t, r in dbl.items() if r["song"] in DBL_EVAL_SONGS]
dbl_train_ids = [t for t, r in dbl.items() if r["song"] not in DBL_EVAL_SONGS]
print(f"two-rhythm songs: train={len(dbl_train_ids)} eval={len(dbl_eval_ids)}")
assert len(dbl_eval_ids) == 2

train_main = MoisesLeadRhythmDataset(DATA_DIR, train_ids, usable)
train_dbl = MoisesLeadRhythmDataset(DATA_DIR, dbl_train_ids, dbl, double_rhythm=True, blocked=DBL_BLOCKED)
train_ds = MixedDataset(train_main, train_dbl, DOUBLE_RHYTHM_FRAC)
dbl_eval_ds = MoisesLeadRhythmDataset(DATA_DIR, dbl_eval_ids, dbl, double_rhythm=True, blocked=DBL_BLOCKED)
"""

SMOKETEST_CODE = """\
# --- Smoke test: 3 batches with memory monitoring BEFORE committing to a
# real run (standing discipline since Stage 0's OOM). With AMP, the first 3-4
# steps are usually skipped while GradScaler calibrates, so identical-looking
# loss values here are expected (Stage 3 Update 5), not a dead optimizer.
LEAK_MARGIN = leak_baseline(model, train_ds, n=10)
print(f"self-calibrated LEAK_MARGIN: {LEAK_MARGIN:.2f}dB")

opt = torch.optim.Adam(model.parameters(), lr=LR_PEAK)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

for i in range(3):
    mix, lead, rhythm = train_ds.sample_batch(BATCH)
    mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
    opt.zero_grad()
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        out = model(mix)
    pit, hinge, leak, m = combined_loss(out, mix, lead, rhythm,
                                        leak_margin=LEAK_MARGIN, lambda_leak=LAMBDA_LEAK)
    loss = pit + LAMBDA_HINGE * hinge + LAMBDA_LEAK * leak
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(opt)
    scaler.update()
    mem = f"  gpu_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB" if DEVICE == "cuda" else ""
    print(f"  batch {i}: loss={loss.item():.3f} pit={pit.item():.2f} hinge={hinge.item():.2f} "
          f"both-active={int(m['n_both'].item())}/{BATCH}{mem}")

print("\\nSmoke test passed -- note the GPU memory above and adjust BATCH upward if there's headroom.")
"""

EVAL_CODE = """\
def evaluate(dataset, n_batches=10, batch_size=BATCH):
    \"\"\"Returns sep / role / leak01 / leak10 over both-active crops, and buzz
    (mean output level vs the mix when that output's target is silent).\"\"\"
    model.eval()
    acc = {"sep": 0.0, "role": 0.0, "leak01": 0.0, "leak10": 0.0,
           "lead_q": 0.0, "rhy_q": 0.0, "lead_lvl": 0.0, "rhy_lvl": 0.0}
    w, buzz_sum, buzz_n = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            mix, lead, rhythm = dataset.sample_batch(batch_size)
            mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
            _, _, _, m = combined_loss(model(mix), mix, lead, rhythm)
            nb = m["n_both"].item()
            for k in acc:
                acc[k] += m[k].item() * nb
            w += nb
            buzz_sum += m["buzz_sum"].item()
            buzz_n += int(m["buzz_n"].item())
    model.train()
    res = {k: v / max(w, 1) for k, v in acc.items()}
    res["buzz"] = buzz_sum / buzz_n if buzz_n else float("nan")
    res["buzz_n"] = buzz_n
    return res


def fmt(r):
    return (f"sep={r['sep']:6.2f}dB role={r['role']:.2f} leak01={r['leak01']:6.1f} "
            f"leak10={r['leak10']:6.1f} buzz={r['buzz']:6.1f}dB(n={r['buzz_n']})\\n"
            f"         lead: SI-SDR={r['lead_q']:6.1f}dB level={r['lead_lvl']:+6.1f}dB | "
            f"rhythm: SI-SDR={r['rhy_q']:6.1f}dB level={r['rhy_lvl']:+6.1f}dB")


def evaluate_dbl(n_batches=6, batch_size=BATCH):
    \"\"\"v3: held-out two-rhythm-guitar songs. Right answer: lead output silent
    (lead_vs_mix -> -30dB or lower), rhythm output = both guitars at the right
    level (SI-SDR high, level ~0dB). This is The Sentinel's non-solo case.\"\"\"
    model.eval()
    lead_db, q, lvl = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            mix, lead, rhythm = dbl_eval_ds.sample_batch(batch_size)
            mix, rhythm = mix.to(DEVICE), rhythm.to(DEVICE)
            out = model(mix)
            lead_db.append(level_db(out[:, 0], mix).clamp(min=-60).mean().item())
            q.append(sisdr_flat(out[:, 1], rhythm).mean().item())
            lvl.append(level_db(out[:, 1], rhythm).mean().item())
    model.train()
    return (f"two-rhythm: lead-out vs mix={np.mean(lead_db):6.1f}dB | "
            f"rhythm: SI-SDR={np.mean(q):6.1f}dB level={np.mean(lvl):+6.1f}dB")
"""

TRAIN_CODE = """\
# --- Main training loop (v2). Fresh start from pretrained HTDemucs -- NOT
# from v1's checkpoints, which learned the boosted-noise targets.
# Stability, after v1 hit NaN twice at a flat lr=2e-4 around epoch 5-8:
# linear warmup over the first epoch, cosine decay to 5% of peak, and
# gradient-norm clipping. The divergence guard is kept, but only checks for
# non-finite loss now -- the v1 floor on best_perm doesn't apply to the new score.
import math

STEPS_PER_EPOCH = 200
TOTAL_STEPS = N_EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = STEPS_PER_EPOCH


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR_PEAK * (0.1 + 0.9 * step / WARMUP_STEPS)
    p = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return LR_PEAK * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


os.makedirs("/kaggle/working", exist_ok=True)
print("INIT  test", fmt(evaluate(test_ds)))
print("INIT  " + evaluate_dbl())
print("(sep is the new mixture-thresholded SNR on both-active crops -- NOT comparable to v1's "
      "SI-SDR best_perm. buzz: lower is better, -30dB is the floor the loss asks for.)")

global_step, diverged = 0, False
for epoch in range(N_EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(global_step)
        mix, lead, rhythm = train_ds.sample_batch(BATCH)
        mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            out = model(mix)
        pit, hinge, leak, _ = combined_loss(out, mix, lead, rhythm,
                                            leak_margin=LEAK_MARGIN, lambda_leak=LAMBDA_LEAK)
        loss = pit + LAMBDA_HINGE * hinge + LAMBDA_LEAK * leak
        if not torch.isfinite(loss):
            print(f"  NON-FINITE LOSS at epoch {epoch+1} step {step} -- stopping; "
                  f"checkpoints up to epoch {epoch} are intact.")
            diverged = True
            break
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        global_step += 1
    if diverged:
        break

    print(f"epoch {epoch+1:2d} lr={lr_at(global_step):.1e}")
    print("   val ", fmt(evaluate(val_ds)))
    print("   test", fmt(evaluate(test_ds)))
    print("   " + evaluate_dbl())
    ckpt_path = f"/kaggle/working/{CKPT_PREFIX}_epoch{epoch+1}.pt"
    torch.save(core.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path}")
"""

NEXT_MD = """\
## After this run

1. Download checkpoints DURING the run (Kaggle Output panel), not only at
   the end -- Kaggle RAM crashes have eaten checkpoints before.
2. Score them locally on The Sentinel with `tools/realsong_test_stereo.py`
   and listen. Select by the real song, never by these in-domain numbers
   (five measured instances in this project of the in-domain metric
   picking the wrong checkpoint).
3. The real questions for this run (v3): does the **lead output** carry
   the solos, and do **non-solo passages** of the rhythm output stay full
   and clean, not thinned out? Compare against v2 epoch 13
   (`sentinel_v2_ep13_*`) and v1 (`sentinel_cont_ep7_*`) in
   `output/stage4_local_test/`. Locally, `tools/diagnose_htd_ckpt.py`
   gives per-output numbers on any downloaded checkpoint.

## History

v1 of this notebook (15 epochs at lr=1e-4, then resumes at 5e-4 / 2e-4 /
overlap_bias=0.3) produced `htd_leadrhythm_lr2e4_cont_epoch7.pt`: real lead
isolation on The Sentinel, but buzzing in non-solo passages, negative
in-domain best_perm, and NaN at epoch 5-8. Those resume cells were removed
in v2 because they continue from checkpoints trained on the boosted-noise
targets; they are in git history (commit 9896369 and earlier).
"""

def build_cells():
    with open(os.path.join(os.path.dirname(__file__), "dbl_rhythm_blocked_windows.json")) as f:
        blocked = json.load(f)["blocked"]
    split_code = SPLIT_CODE.replace("__DBL_BLOCKED__", json.dumps(blocked))
    return [
        cell(MD, INTRO_MD),
        cell(CODE, SETUP_CODE),
        cell(CODE, CONFIG_CODE),
        cell(CODE, DATA_LOCATE_CODE),
        cell(CODE, MODEL_CODE),
        cell(CODE, LOSS_CODE),
        cell(CODE, DATASET_CODE),
        cell(CODE, split_code),
        cell(CODE, SMOKETEST_CODE),
        cell(CODE, EVAL_CODE),
        cell(CODE, TRAIN_CODE),
        cell(MD, NEXT_MD),
    ]


NOTEBOOK_TEMPLATE = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write_notebook(path, cells):
    nb = dict(NOTEBOOK_TEMPLATE)
    nb["cells"] = cells
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path} ({len(cells)} cells)")


def main():
    kaggle_cells = build_cells()
    write_notebook("output/stage4_kaggle/Stage4_HTDemucs_MoisesDB.ipynb", kaggle_cells)

    # Colab twin: same cells, swap the data-location cell for an upload widget.
    colab_cells = list(kaggle_cells)
    colab_data_cell = cell(CODE, """\
# Colab: upload a zip of output/moisesdb_guitar/ (6.07GB -- this will be slow
# over the upload widget; mounting Google Drive is the faster alternative if
# you've already put the folder there) and extract it.
from google.colab import files
import zipfile

print("Upload moisesdb_guitar.zip (zip of the output/moisesdb_guitar/ folder)")
uploaded = files.upload()
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall("/content/moisesdb_guitar")
DATA_DIR = "/content/moisesdb_guitar"
for root, dirs, fs in os.walk(DATA_DIR):
    if "manifest.json" in fs:
        DATA_DIR = root
        break
print("DATA_DIR:", DATA_DIR)
""")
    data_idx = next(i for i, c in enumerate(colab_cells)
                     if c["cell_type"] == CODE and "/kaggle/input" in "".join(c["source"]))
    colab_cells[data_idx] = colab_data_cell
    write_notebook("output/stage4_kaggle/Stage4_HTDemucs_MoisesDB_Colab.ipynb", colab_cells)


if __name__ == "__main__":
    main()
