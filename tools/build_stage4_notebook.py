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

## v3.1 (2026-10-18): warm start from v3 epoch 5, rebalanced

v3 run 1: at epoch 5 the lead output broke out (test SI-SDR -5.5dB, near-
complete solo isolation by ear, the best lead of the project), then slid back
toward silence as the model over-learned "two guitars -> all rhythm" (two-
rhythm eval +20dB, but solos leaking into the rhythm output). Also found: a
DC-offset loophole (SI-SDR ignores DC; the level term counted it), so the
rhythm output's real content stayed ~20dB quiet. v3.1: warm start from
`htd_v3_epoch5.pt` (attach it as a Kaggle Dataset), level measured without
DC + a DC penalty, silence x4 -> x2, level weight 0.5 -> 1.0, two-rhythm
share 25% -> 10%, peak LR 1e-4, 15 epochs.

## v3.2 (2026-10-18): stop the "rhythm = copy of the input" shortcut

v3.1 epoch 3 is the best checkpoint so far: on The Sentinel the lead output
beats the rhythm output during both solos (ratios 1.90 / 1.39), and the
rhythm output is ~8dB fuller in the verse than v3 epoch 5. After that the
rhythm output drifted into copying the input (rhythm vs mixture SI-SDR
0 -> +13dB by epoch 5, solos included). Copying is correct whenever the lead
rests (~40% of crops) and easy to learn. v3.2, warm-started from
`htd_v31_epoch3.pt` (attach it as a Kaggle Dataset):

- **Rhythm-side hinge:** the rhythm output must resemble the rhythm >= 3dB
  more than the lead. This mirrors the existing lead hinge. In a static test
  on real crops it makes copying cost ~28 loss units more than separating.
- **Fixed leak margins (-20dB for each output)** instead of self-calibration,
  which ran on memorised training crops and came out at -31/-35dB.

## v4a (2026-10-19): rhythm output = input - lead output

v3.2 ran 15 epochs and no epoch beat its starting point. It was the third
run in a row (v3, v3.1, v3.2) to drift into "rhythm output = copy of the
input, lead output quiet", despite three different loss fixes. On real songs
(The Sentinel, Hot Blooded, Your Light) the rhythm output carries the whole
solo. Measured why the v3.2 rhythm hinge couldn't stop it: on 46% of
both-active training crops, copying the input already passes the hinge for
free.

v4a changes the structure instead of the loss: **the rhythm output is
computed as `mix - lead`** (`MixConsistent` in the model cell). Copying the
input now requires silencing the lead, which the lead loss punishes whenever
a solo plays, and a too-quiet lead leaves solo in the rhythm output, so the
rhythm loss punishes that too. Everything else is identical to v3.2 (data,
losses, LR, epochs, warm start from `htd_v31_epoch3.pt`), so any change in
behaviour is this one change. The data is unchanged, so there's no
listening gate.

**Lead polarity.** The v3/v3.1 lead head outputs the lead **phase-inverted
on 100% of measured crops** (both checkpoints, train and test songs): the
losses are sign-blind, so nothing ever fixed it. Uncorrected, the warm start
would give rhythm = mix + lead. The smoke-test cell measures the sign on
training crops and stores it in the wrapper (`lead_sign`, saved with the
checkpoint).

**Expect a bad-looking INIT line.** Before training, the inherited lead
output is ~16dB too quiet, so mix - lead is nearly a copy of the input
(measured on 24 test crops: rhythm level +3.4dB, leak10 0.0dB, rhythm vs
mixture 18.4dB). v4a starts AT the copy state; what changed is that the loss
now punishes it twice (lead too quiet, and solo left in the rhythm output).
**What to watch:** the lead level climbing toward 0dB and leak10 falling.
Judge epochs against v3.2's INIT (leak10 -8.3, lead SI-SDR -5.8, role 0.81),
the best starting point the old structure had, not against this run's own
INIT, which is easy to beat.

Local tools (`tools/realsong_test_stereo.py`, `tools/diagnose_htd_ckpt.py`)
load these checkpoints via `load_leadrhythm_checkpoint`, which recognises
the `lead_sign` key.

## v5 (2026-10-19): cross-song remix augmentation

v4a ran 11 epochs and no epoch beat v3.2's starting point. More telling:
across v3, v3.1, v3.2 and v4a, held-out lead quality stayed at about -6 to -9dB
SI-SDR, while on training songs the same model scores +1.5dB (warm start,
measured). Leak shows the same gap: -31 / -11.7dB on training crops vs -23 /
-1.5dB on test. Each loss or structure change only moved where the error
lands. That's too little variety (14 song pairs), not the loss.

v5 = v3.2 (two independent outputs, same loss, LR, epochs, warm start) +
**cross-song remixes**: one training song's lead track over another training
song's rhythm track, as Demucs itself is trained. 50% of crops are remixes
(75% key-matched by pitch-class profile, 25% random), 40% original pairs, 10%
two-rhythm. Remix lead crops only use windows where the lead track sounds
like lead on its own; rhythm crops avoid lead-sounding moments
(`tools/remix_windows.py`). Lead-vs-rhythm level is drawn from +-6dB.
Validation and test are unchanged, so the numbers compare directly with v3.2.
Listening-approved 2026-10-19.

New per-epoch **`train`** line: the same metrics on training songs (original
pairs). The bet is that the train-vs-test gap narrows.

## v5.1 (2026-10-19): native 7.8s crops, fixed evaluation crops

v5 epoch 12 is the best checkpoint of the project on all three real songs
(by measurement; by ear on The Sentinel and Your Light). What's left is
momentary misrouting at solo handoffs: for a few seconds the solo goes to the
rhythm output and vanishes from the lead output.

- **7.8s crops** (was 4s). HTDemucs sets its window to the input length in
  training, so v5 trained on 4s. Inference with 7.8s chunks removed the
  handoff dropouts but made separation less decisive, plausibly because the
  model had never trained on that much context. 7.8s is the pretrained
  model's native window, so training and inference now match with no
  padding. ~2x memory per crop: batch 8 -> 4.
- **Fixed evaluation crops** (40 test, 40 val, 24 train, 16 two-rhythm),
  drawn once with a fixed seed and reused every epoch. v5's lines used 6-16
  fresh random crops and jumped 3-5dB between epochs.
- **"Never Say Never" out of validation** (labels unreliable by ear).
  Validation is "Darkest Heart" alone; training songs are unchanged.
- Warm start from `htd_v5_epoch12.pt`. Everything else as v5.
- Local real-song tests of v5.1 checkpoints: `tools/realsong_test_stereo.py
  --chunk-sec 7.8`.

## v6 (2026-10-23): train on the real-song input

v5 epoch 12 is the reference (by ear: The Sentinel's rhythm track "much more
clear, no noises", Your Light "best results so far"). v5.1 (7.8s crops) tied
it. The biggest measured loss left is the **input**: real songs reach the
model as a guitar stem that htdemucs_6s extracted from a full mix, but every
run so far trained on clean tracks. On two held-out MoisesDB songs, that input
alone costs 6-10dB of lead/rhythm SI-SDR, more than any training change has
ever moved (input-mismatch test, CLAUDE.md 2026-10-19). v6 = v5.1 plus:

1. **Demucs-extracted inputs.** For each of the 31 pair songs, the full mix
   (all stems) was run through htdemucs_6s locally, as the app does
   (`moisesdb_stems.zip`). Half of the original-pair and two-rhythm crops
   use that as the INPUT; targets stay the true tracks. The Demucs stem holds
   every guitar in the song: accompaniment-like extra tracks (acoustic, clean
   comping) join the rhythm target, and windows where a lead-like extra plays
   are skipped (`tools/extra_guitar_roles.py`).
2. **More remix material.** The 82 MoisesDB songs with one distorted guitar
   (`moisesdb_guitar_single.zip`): confident lead and rhythm windows join the
   remix pool. The test songs' artists are left out (`EXCLUDE_TEST_ARTISTS`).
3. **Other-instrument bleed** on half the remix inputs: real drums / bass /
   vocals / keys stems from the training songs, at the level Demucs's own
   guitar stems leak (measured, `BLEED_DB`). Targets stay clean.

Warm start `htd_v51_epoch5.pt` (7.8s crops). Attach FOUR inputs: the
moisesdb_guitar dataset, moisesdb_guitar_single, moisesdb_stems, and the
checkpoint. Listening-approved before this run, per the hard gate.

**The new line to watch is `tdmx`**: the test songs through the Demucs input.
Download a checkpoint only if `tdmx` clearly beats its INIT and `test` (clean
input) doesn't fall apart.

## v6a (2026-10-27): one change at a time

v6 was negative: in 6 epochs `tdmx` never beat its INIT (lead 1.0 -> -5.5dB),
clean `test` fell (lead 3.8 -> 0.5), and epochs see-sawed between strong
two-rhythm behaviour with a collapsed lead on Demucs input (ep3/ep5) and the
reverse (ep4), ending in copy-shortcut drift (ep6 leak10 -12.7). Three changes
were bundled, so v6a keeps only the central one:

- Demucs input on **half the original-pair crops only** (`DEMUCS_INPUT_FRAC`).
- Two-rhythm crops back to clean input (`DBL_DEMUCS_FRAC = 0`).
- No bleed augmentation (`BLEED_FRAC = 0`), no single-guitar songs
  (`USE_SINGLE_GUITAR = False`): the remix pool is v5.1's again.

Same warm start (`htd_v51_epoch5.pt`), so the INIT lines equal v6's
(test 3.8 / 5.3 / leak10 -17.2, tdmx 1.0 / -1.1). Inputs: moisesdb_guitar,
moisesdb_stems and the checkpoint (moisesdb_guitar_single may stay attached;
it isn't read). Same download bar: `tdmx` clearly above INIT, `test` holding.
If v6a also fails, Demucs-input training doesn't close the gap with 14
training songs, and the approach needs rethinking, not tuning.

## What to watch

- **`sep`** -- both-active separation, mixture-thresholded SNR (capped
  ~30dB). New metric, **not comparable to v1's SI-SDR best_perm**.
- **`role`** -- fraction of both-active crops where out0 is closer to the
  lead than to the rhythm. ~50% = not working.
- **`leak01` / `leak10`** -- SI-SDR of each output against the WRONG
  source, both-active crops only. Lower is better. **`leak10` is the copy-
  shortcut alarm**: drifting toward 0dB means the rhythm output is turning
  into a copy of the input. In v4a the rhythm output is mix - lead, so a
  high `leak10` there means the lead output is missing part of the solo.
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
SEG_SEC = 7.8   # v5.1: HTDemucs's native window (was 4.0). See the v5.1 markdown
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_DP = torch.cuda.device_count() > 1
BATCH = 4 if USE_DP else 2   # v5.1: was 8/4 at 4s crops; 7.8s crops need ~2x memory per crop
N_EPOCHS = 15
LR_PEAK = 1e-4       # v3.1: fine-tuning from htd_v3_epoch5, not learning from scratch (fresh starts used 2e-4)
GRAD_CLIP = 5.0
CKPT_PREFIX = "htd_v6a"
MIX_CONSISTENT = False    # v4a's rhythm = mix - lead wrapper. v4a was negative (CLAUDE.md 2026-10-19), so v5 uses v3.2's two outputs
# v5: cross-song remix augmentation (one song's lead track over another's rhythm
# track), the way Demucs itself is trained. Every run v3-v4a scored far better on
# training songs than held-out ones (lead SI-SDR +1.5 vs ~-8.5dB): too little
# variety, 14 song pairs. Listening-approved 2026-10-19 (key-matched preferred,
# random "will still do").
REMIX_FRAC = 0.5          # share of training crops that are cross-song remixes
REMIX_KEYMATCH_FRAC = 0.75  # of those, rhythm = a key fit (random among the best 8 of 64, 8s context); rest random
REMIX_GAIN_DB = 6.0       # lead-vs-rhythm level drawn uniformly from +-6dB (both sources always playing)
RHY_HINGE_MARGIN = 3.0   # v3.2: rhythm-side hinge margin (dB), see combined_loss
LAMBDA_SIL = 2.0     # v3.1: was 4.0. At x4 the lead output slid back to silence after epoch 5 (CLAUDE.md 2026-10-18)
LAMBDA_LVL = 1.0     # v3.1: was 0.5; level-match term for playing sources (DC excluded), outside a +-3dB deadzone
LVL_DEADZONE = 3.0
DOUBLE_RHYTHM_FRAC = 0.10   # v3.1: was 0.25, which over-taught "two guitars -> all rhythm" (solos leaked into rhythm)
INIT_CKPT = "htd_v51_epoch5.pt"   # v6: trained at 7.8s crops; ties v5 ep12 by ear (CLAUDE.md 2026-10-19). None = fresh
EVAL_SEED = 1234   # v5.1: fixed evaluation crops, drawn once (see the split cell)
EVAL_N = {"test": 40, "val": 40, "train": 24, "dbl": 16}
LAMBDA_HINGE = 3.0   # Stage 2/3 found this is where routing actually gets a real gradient
LAMBDA_LEAK = 1.0    # conservative starting weight, same philosophy as every prior leak-loss run
LAMBDA_BAND = 0.0    # deliberately off this run -- see markdown above
# v6: the real-song input. Real songs reach the model as a guitar stem that
# htdemucs_6s extracted from a full mix; training only ever saw clean tracks, and
# that costs 6-10dB on the same songs (input-mismatch test, CLAUDE.md 2026-10-19).
DEMUCS_INPUT_FRAC = 0.5   # original-pair crops whose INPUT is the Demucs guitar of the full mix (targets stay the true tracks)
# v6a: v6 (all three changes at once) was negative (CLAUDE.md 2026-10-27), so
# one change at a time. Only the Demucs input on the pair crops stays on.
DBL_DEMUCS_FRAC = 0.0     # v6a: was 0.5 in v6. Smeared two-rhythm inputs plausibly fed the lead-vs-two-rhythm seesaw
BLEED_FRAC = 0.0          # v6a: was 0.5 in v6. Remix crops with other-instrument bleed added to the INPUT (targets stay clean)
USE_SINGLE_GUITAR = False # v6a: was True in v6. The 82 single-guitar songs' windows in the remix pool
BLEED_DB = __BLEED_DB__   # bleed level vs the guitar mix, dB, drawn uniformly (measured from Demucs's own residual, see markdown)
EXCLUDE_TEST_ARTISTS = True   # keep the test songs' artists (Firefly, The Unfortunates) out of the NEW single-guitar remix pool

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

# v6: two more datasets. moisesdb_guitar_single.zip (the 82 songs with ONE
# distorted guitar: more lead/rhythm material for remixing) and
# moisesdb_stems.zip (the 31 pair songs' non-guitar stems as category FLACs +
# the htdemucs_6s guitar of each full mix).
SINGLE_DIR, STEMS_DIR = None, None
for root, dirs, files in os.walk("/kaggle/input"):
    if "stems_manifest.json" in files:
        STEMS_DIR = root
    elif "manifest.json" in files and "role_labels.json" not in files:
        SINGLE_DIR = root
assert STEMS_DIR, "attach moisesdb_stems via + Add Input"
assert SINGLE_DIR or not USE_SINGLE_GUITAR, "USE_SINGLE_GUITAR needs moisesdb_guitar_single attached"
print("SINGLE_DIR:", SINGLE_DIR, "| STEMS_DIR:", STEMS_DIR)
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


class MixConsistent(nn.Module):
    \"\"\"v4a: rhythm output = input mixture - lead output (see the markdown at
    the top). Copying the input now requires silencing the lead. lead_sign
    corrects the inherited lead head's inverted polarity; it is measured in
    the smoke-test cell and saved with the checkpoint. Same class as
    tools/htdemucs_leadrhythm.py, which local tools use to load these.\"\"\"
    def __init__(self, base, lead_sign=1.0):
        super().__init__()
        self.base = base
        self.register_buffer("lead_sign", torch.tensor(float(lead_sign)))

    def forward(self, mix):
        lead = self.lead_sign * self.base(mix)[:, 0]
        return torch.stack([lead, mix - lead], dim=1)


model = build_leadrhythm_htdemucs()
if INIT_CKPT:
    # Upload the checkpoint as a private Kaggle Dataset and attach it (+ Add Input);
    # on Colab, put it in /content. Searched for by file name.
    found = [os.path.join(r, INIT_CKPT) for base in ("/kaggle/input", "/content")
             for r, _, fs in os.walk(base) if INIT_CKPT in fs]
    assert found, f"{INIT_CKPT} not found under /kaggle/input or /content -- attach it first"
    model.load_state_dict(torch.load(found[0], map_location="cpu"))
    print("warm start from", found[0])
backbone_params = sum(p.numel() for n, p in model.named_parameters()
                       if "decoder.3.conv_tr" not in n and "tdecoder.3.conv_tr" not in n)
head_params = sum(p.numel() for n, p in model.named_parameters()) - backbone_params
print(f"backbone (pretrained): {backbone_params:,}  new head: {head_params:,}")

model = (MixConsistent(model) if MIX_CONSISTENT else model).to(DEVICE)
core = model   # saved checkpoints = core.state_dict() (v4a: base.* plus lead_sign)
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


def _zm(x):
    return x - x.mean(-1, keepdim=True)   # remove DC per channel


def dc_db(x, ref):
    \"\"\"DC energy of x relative to ref's AC energy, dB.\"\"\"
    dc = x.float().mean(-1, keepdim=True).expand_as(x)
    return level_db(dc, _zm(ref))


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
    # v3.1: level measured with DC removed. v3 run 1 found the loophole: SI-SDR
    # ignores DC, and the level term counted DC as level, so the rhythm output
    # met "right level" with a -0.075 constant offset while its actual guitar
    # content stayed ~20dB quiet (CLAUDE.md 2026-10-18). DC is now also pushed
    # below -30dB of the target directly.
    lvl = F.relu(level_db(_zm(o), _zm(ref)).abs() - LVL_DEADZONE).clamp(max=30.0)
    dc = F.relu(dc_db(o, ref) + 30.0).clamp(max=30.0)
    active = -sisdr_flat(o, ref) + LAMBDA_LVL * (lvl + dc)
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
    # Lead hinge: out0 must resemble the lead >= 1dB more than the rhythm.
    # v3.2 adds the mirror for out1 (>= RHY_HINGE_MARGIN dB more like the
    # rhythm than the lead). Without it nothing directly opposed the "rhythm
    # output = copy of the input" shortcut that v3/v3.1 slid into (out1 vs
    # mixture SI-SDR went 0 -> +13dB). Static test on real crops: the copy-vs-
    # separate gap widens from ~10 to ~27 loss units (CLAUDE.md 2026-10-18).
    hinge = (_mmean(F.relu(margin - (q00 - q01)), both)
             + _mmean(F.relu(RHY_HINGE_MARGIN - (q11 - q10)), both))
    if lambda_leak > 0 and leak_margin is not None:
        # v3.2: one margin per output. v3.1 used min(leak01, leak10) - 2 for BOTH,
        # i.e. the clean lead output's -35dB imposed on the rhythm output too
        # (which started at -11dB), so its leak penalty was large and permanent.
        m01, m10 = leak_margin if isinstance(leak_margin, tuple) else (leak_margin, leak_margin)
        leak = (_mmean(F.relu(q01 - m01), both) + _mmean(F.relu(q10 - m10), both)) / 2
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
        "lead_lvl": _mmean(level_db(_zm(o0), _zm(LEAD)).detach(), both),   # DC excluded (v3.1)
        "rhy_lvl": _mmean(level_db(_zm(o1), _zm(RHYTHM)).detach(), both),
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
        "rhy_lvl_on": _mmean(level_db(_zm(o1), _zm(RHYTHM)).detach(), rhy_on.float()),
    }
    return pit_loss, hinge, leak, m


def leak_baseline(model, dataset, n=20):
    \"\"\"Self-calibrate LEAK_MARGIN from the model's own init leakage on
    both-active crops: (baseline leak01 - 2dB, baseline leak10 - 2dB), one per output.\"\"\"
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
    return (l01 / n - 2.0, l10 / n - 2.0)
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


def read_crop(path, start, seg_n, gain=1.0):
    x, _ = sf.read(path, dtype="float32", always_2d=True, start=start, frames=seg_n)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if x.shape[0] < seg_n:
        x = np.pad(x, ((0, seg_n - x.shape[0]), (0, 0)))
    return x / gain


def normalise(inp, lead, rhythm):
    \"\"\"One gain for input and targets (v2: silence stays silent), set from the
    MODEL INPUT so the model always sees the same level (v6: the input can be
    the Demucs guitar or carry bleed, so it is no longer lead + rhythm).
    Returns (input, lead, rhythm), each (2, T).\"\"\"
    g = MIX_RMS / max(float(np.sqrt(np.mean(inp ** 2))), 1e-8)
    peak = np.abs(inp).max() * g
    if peak > 0.95:
        g *= 0.95 / peak
    return (inp * g).T, (lead * g).T, (rhythm * g).T


class MoisesLeadRhythmDataset:
    def __init__(self, data_dir, track_ids, labels_by_id, seg_sec=SEG_SEC,
                 overlap_bias=0.7, active_thresh=0.01, double_rhythm=False, blocked=None,
                 demucs_frac=0.0):
        # double_rhythm (v3): both tracks are rhythm parts. Target is lead =
        # silence, rhythm = both guitars -- the "two rhythm guitars, nobody
        # soloing" case that dominates The Sentinel's non-solo passages and was
        # absent from v1/v2 training.
        # demucs_frac (v6): share of crops whose INPUT is the htdemucs_6s guitar
        # of the song's full mix (STEMS_DIR). It contains every guitar in the
        # song, so the extra tracks need targets (EXTRA_ROLES, from
        # tools/extra_guitar_roles.py): accompaniment-like extras join the
        # rhythm target; lead-like extras are never labelled -- Demucs-input
        # crops avoid the windows where they play.
        self.seg_sec = seg_sec
        self.overlap_bias = overlap_bias
        self.double_rhythm = double_rhythm
        self.demucs_frac = demucs_frac
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
            song = {
                "blocked": bl,
                "track_id": tid, "artist": r["artist"], "song": r["song"],
                "lead_path": lead_path, "rhythm_path": rhythm_path,
                "n_samples": min(n_lead, n_rhythm), "overlap_windows": both,
                "overlap_frac": len(both) / max(1, n), "dmx": None,
            }
            if demucs_frac > 0 and tid in STEMS_MAN and "demucs_guitar" in STEMS_MAN[tid]:
                extras, dmx_bl = [], set()
                for gid, use in EXTRA_ROLES.get(tid, {}).items():
                    path = os.path.join(data_dir, tid, "guitar", gid + ".wav")
                    if use == "rhythm_extra":
                        extras.append(path)
                    elif use == "block":
                        rms, _, _ = find_active_windows(path)
                        dmx_bl |= {int(i) for i in np.where(rms > active_thresh)[0]}
                song["dmx"] = {"path": os.path.join(STEMS_DIR, tid, "demucs_guitar.flac"),
                               "gain": STEMS_MAN[tid]["demucs_guitar"]["gain"],
                               "extras": extras, "blocked": bl | dmx_bl}
            self.songs.append(song)

    @staticmethod
    def _clear(blocked, start, seg_n):
        if not blocked:
            return True
        w0, w1 = start // (2 * SR), (start + seg_n - 1) // (2 * SR)
        return not (set(range(w0, w1 + 1)) & blocked)

    def _sample_start(self, song, blocked):
        \"\"\"Returns a start clear of `blocked`, or None after 50 tries.\"\"\"
        seg_n = int(self.seg_sec * SR)
        max_start = max(1, song["n_samples"] - seg_n)
        for _try in range(50):
            if len(song["overlap_windows"]) > 0 and random.random() < self.overlap_bias:
                center = int(random.choice(song["overlap_windows"]) * 2.0 * SR) + SR
                start = max(0, min(max_start, center - seg_n // 2))
            else:
                start = random.randint(0, max_start)
            if self._clear(blocked, start, seg_n):
                return start
        return None

    def sample_one(self):
        seg_n = int(self.seg_sec * SR)
        for _try in range(20):
            song = random.choice(self.songs)
            dmx = song["dmx"] if (song["dmx"] and random.random() < self.demucs_frac) else None
            start = self._sample_start(song, dmx["blocked"] if dmx else song["blocked"])
            if start is None and dmx:   # no window clear of lead-like extras: clean input instead
                dmx, start = None, self._sample_start(song, song["blocked"])
            if start is None:
                start = random.randint(0, max(1, song["n_samples"] - seg_n))
            lead = read_crop(song["lead_path"], start, seg_n)
            rhythm = read_crop(song["rhythm_path"], start, seg_n)
            if np.sqrt(np.mean((lead + rhythm) ** 2)) >= SILENT_MIX_RMS:
                break
        if dmx:
            for path in dmx["extras"]:
                rhythm = rhythm + read_crop(path, start, seg_n)
            inp = read_crop(dmx["path"], start, seg_n, dmx["gain"])
        else:
            inp = lead + rhythm
        if self.double_rhythm:
            lead, rhythm = np.zeros_like(lead), lead + rhythm
        return normalise(inp, lead, rhythm)

    def sample_batch(self, batch_size):
        return _stack([self.sample_one() for _ in range(batch_size)])


def _stack(items):
    \"\"\"items: (input, lead, rhythm) triples. v6: the input is no longer always
    lead + rhythm (Demucs-extracted guitar, or bleed added).\"\"\"
    return tuple(torch.from_numpy(np.stack([it[k] for it in items])).float() for k in range(3))


class MixedDataset:
    \"\"\"Each crop comes from `extra` with probability `frac`, else from `main`.
    `extra` can also be a list of (dataset, prob) pairs.\"\"\"
    def __init__(self, main, extra, frac=None):
        self.main = main
        self.parts = extra if isinstance(extra, list) else [(extra, frac)]

    def _pick(self):
        u = random.random()
        for ds, p in self.parts:
            if u < p:
                return ds
            u -= p
        return self.main

    def sample_batch(self, batch_size):
        return _stack([self._pick().sample_one() for _ in range(batch_size)])


class FixedCrops:
    \"\"\"v5.1: evaluation crops drawn ONCE with a fixed seed and reused every
    epoch. v5's per-epoch lines used 6-16 fresh random crops, which made epochs
    jump 3-5dB; its late "+3dB" epochs tied epoch 1 on a fixed 24-crop set
    (CLAUDE.md 2026-10-19).\"\"\"
    def __init__(self, dataset, n, seed):
        rs, ns = random.getstate(), np.random.get_state()
        random.seed(seed); np.random.seed(seed)
        self.items = [dataset.sample_one() for _ in range(n)]
        random.setstate(rs); np.random.set_state(ns)

    def batches(self, batch_size):
        for i in range(0, len(self.items), batch_size):
            yield _stack(self.items[i:i + batch_size])


def chroma12(x):
    \"\"\"Pitch-class energy profile (80-2000Hz), unit norm. x: (T,) mono.\"\"\"
    n = 8192
    fr = np.abs(np.fft.rfft(x[:len(x) // n * n].reshape(-1, n) * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    band = (f > 80) & (f < 2000)
    pc = np.round(12 * np.log2(f[band] / 440)).astype(int) % 12
    c = np.array([fr[:, band][:, pc == k].sum() for k in range(12)])
    return c / (np.linalg.norm(c) + 1e-12)


class RemixDataset:
    \"\"\"v5: one song's lead track over a DIFFERENT song's rhythm track.
    Lead crops: 4s where the lead track is playing and sounds like lead on its
    own (classifier P(lead) > 0.8, REMIX_WINDOWS["lead_ok"]). Rhythm crops: 4s of
    a rhythm track (or a two-rhythm song's track), playing, avoiding lead-
    sounding windows. Both sources always play, so per-source level setting is
    safe here (v1's boosting bug was about silent sources). With probability
    REMIX_KEYMATCH_FRAC the rhythm is a key fit: random among the 8 best of 64
    random candidates, by pitch-class profile over 8s of context around each
    crop. Listening-approved 2026-10-19.
    Two fixes after the first generator samples (user: "the key-matched ones
    don't really have a lead"): (1) a lead crop is used only if the lead is
    sounding (within 12dB of its peak) for >= 75% of it -- the per-2s-window
    activity check let a lead that enters halfway through pass; (2) key fit on
    the 4s crop itself matched NOTES, so a sparse lead got a rhythm part
    playing the same note (unison masking); 8s context matches key/harmony
    instead, and top-8-of-64 avoids the extreme matches. Even so, 32% of
    key-matched pairs had the rhythm hammering the lead's main note, vs 14% in
    the original song pairs and 10% in random remixes, so key-matched picks
    now skip candidates whose crop shares the lead's dominant note (>0.6 of
    the rhythm crop's pitch-class energy).\"\"\"
    def __init__(self, lead_tracks, rhythm_tracks, active_thresh=0.01, bleed=None):
        # *_tracks: lists of (song_key, path, allowed_or_blocked_window_set, mode)
        self.bleed = bleed   # v6: BleedSource or None
        self.leads, self.rhys = [], []
        for dst, tracks in ((self.leads, lead_tracks), (self.rhys, rhythm_tracks)):
            for key, path, wins, mode in tracks:
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                mono = data.mean(axis=1)
                w = int(2 * SR)
                n = len(mono) // w
                rms = np.array([np.sqrt(np.mean(mono[i*w:(i+1)*w] ** 2)) for i in range(n)])
                ok = [(rms[i] > active_thresh) and ((i in wins) if mode == "allow" else (i not in wins))
                      for i in range(n)]
                k = int(np.ceil(SEG_SEC / 2))   # 2s windows per crop (v5.1: 4 for 7.8s crops)
                for i in range(n - k + 1):
                    if all(ok[i:i + k]):
                        a, b = max(0, i - 1), min(n, i + k + 1)   # crop plus ~2s of context each side
                        dst.append((key, path, i, chroma12(mono[a*w:b*w]), chroma12(mono[i*w:(i+k)*w])))
        print(f"remix: {len(self.leads)} lead crops from {len({k for k,*_ in self.leads})} songs, "
              f"{len(self.rhys)} rhythm crops from {len({k for k,*_ in self.rhys})} songs")

    def _read(self, path, i):
        x, _ = sf.read(path, dtype="float32", always_2d=True, start=int(i * 2 * SR), frames=int(SEG_SEC * SR))
        if x.shape[0] < int(SEG_SEC * SR):
            x = np.pad(x, ((0, int(SEG_SEC * SR) - x.shape[0]), (0, 0)))
        return x

    @staticmethod
    def _sounding_frac(x):
        m = x.mean(axis=1)
        q = SR // 4
        e = np.array([np.sqrt(np.mean(m[k*q:(k+1)*q] ** 2)) for k in range(len(m) // q)])
        return float(np.mean(e > e.max() * 10 ** (-12 / 20)))

    def sample_one(self):
        for _try in range(20):
            lk, lp, li, lc, ln = random.choice(self.leads)
            lead = self._read(lp, li)
            if self._sounding_frac(lead) >= 0.75:
                break
        cands = [c for c in random.sample(self.rhys, 64) if c[0] != lk]
        if random.random() < REMIX_KEYMATCH_FRAC:
            note = int(np.argmax(ln))
            ok = [c for c in cands if not (int(np.argmax(c[4])) == note and c[4].max() > 0.6)] or cands
            ok.sort(key=lambda c: float(c[3] @ lc), reverse=True)
            rk, rp, ri = random.choice(ok[:8])[:3]
        else:
            rk, rp, ri = cands[0][:3]
        rhythm = self._read(rp, ri)
        rl, rr = np.sqrt(np.mean(lead ** 2)), np.sqrt(np.mean(rhythm ** 2))
        lead = lead * (rr / max(rl, 1e-8)) * 10 ** (random.uniform(-REMIX_GAIN_DB, REMIX_GAIN_DB) / 20)
        inp = lead + rhythm
        if self.bleed is not None and random.random() < BLEED_FRAC:
            inp = inp + self.bleed.sample(inp)
        return normalise(inp, lead, rhythm)


class BleedSource:
    \"\"\"v6: other-instrument bleed for remix inputs (targets stay clean).
    Real non-guitar stems (drums / bass / vocals / other = keys, strings, ...)
    of the TRAINING pair songs, summed per category in STEMS_DIR. Each crop:
    a random song and position, each category in with its own probability
    (weighted toward what Demucs actually leaks into its guitar stem: keys and
    vocals more than bass), random +-6dB per category, then scaled to BLEED_DB
    below the guitar mix. BLEED_DB comes from the measured residual of
    htdemucs_6s's guitar stem vs the true guitar sum on the training songs.\"\"\"
    P_CAT = {"other": 0.7, "vocals": 0.5, "drums": 0.4, "bass": 0.2}

    def __init__(self, track_ids):
        self.songs = [(t, STEMS_MAN[t]) for t in track_ids if t in STEMS_MAN]
        print(f"bleed: {len(self.songs)} songs' non-guitar stems")

    def sample(self, guitar):
        seg_n = guitar.shape[0]
        for _try in range(20):
            tid, e = random.choice(self.songs)
            cats = [c for c in e["cats"] if random.random() < self.P_CAT.get(c, 0.5)] or [random.choice(list(e["cats"]))]
            n = min(e["cats"][c]["frames"] for c in cats)
            start = random.randint(0, max(1, n - seg_n))
            b = sum(read_crop(os.path.join(STEMS_DIR, tid, f"{c}.flac"), start, seg_n, e["cats"][c]["gain"])
                    * 10 ** (random.uniform(-6, 6) / 20) for c in cats)
            br = float(np.sqrt(np.mean(b ** 2)))
            if br > 1e-4:
                break
        target_db = random.uniform(*BLEED_DB)
        return b * (np.sqrt(np.mean(guitar ** 2)) / max(br, 1e-8)) * 10 ** (target_db / 20)
"""

SPLIT_CODE = """\
# --- Song-level train/val/test split, biased so val/test actually have
# overlap content worth measuring (a song with 0% overlap, like "Is It All
# On Me", can't tell you anything about separation quality).
with open(os.path.join(DATA_DIR, "role_labels.json")) as f:
    labels = json.load(f)
usable = {r["track_id"]: r for r in labels if r["usable"]}
# v6: Demucs-extracted inputs + non-guitar stems (tools/pull_moisesdb_stems.py,
# tools/demucs_guitar_inputs.py), and what to do with each song's EXTRA guitar
# tracks in a Demucs input (tools/extra_guitar_roles.py, embedded):
# "rhythm_extra" joins the rhythm target, "block" = windows where it plays are skipped.
with open(os.path.join(STEMS_DIR, "stems_manifest.json")) as f:
    STEMS_MAN = json.load(f)
EXTRA_ROLES = __EXTRA_ROLES__
print(f"stems for {len(STEMS_MAN)} songs, Demucs guitar for "
      f"{sum('demucs_guitar' in e for e in STEMS_MAN.values())}")

tmp_ds = MoisesLeadRhythmDataset(DATA_DIR, list(usable.keys()), usable, overlap_bias=0.0)
by_overlap = sorted(tmp_ds.songs, key=lambda s: s["overlap_frac"], reverse=True)
for s in by_overlap:
    print(f"  overlap={s['overlap_frac']:.2f}  {s['artist']} - {s['song']}")

test_ids = [s["track_id"] for s in by_overlap[0:2]]
val_ids = [s["track_id"] for s in by_overlap[2:4]]
train_ids = [s["track_id"] for s in by_overlap[4:]]
# v5.1: "Never Say Never" out of validation -- the user found its labels
# unreliable by ear (ambiguous, one control clip heard reversed). It stays out
# of training too, so the training songs are exactly v5's.
val_ids = [t for t in val_ids if usable[t]["song"] != "Never Say Never"]
print(f"\\ntrain={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

val_ds = MoisesLeadRhythmDataset(DATA_DIR, val_ids, usable)
test_ds = MoisesLeadRhythmDataset(DATA_DIR, test_ids, usable)
# v6: the same test songs with the REAL-SONG input (htdemucs_6s guitar of the full
# mix, extras in the rhythm target). This is the line v6 is meant to move.
test_dmx_ds = MoisesLeadRhythmDataset(DATA_DIR, test_ids, usable, demucs_frac=1.0)

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

train_main = MoisesLeadRhythmDataset(DATA_DIR, train_ids, usable, demucs_frac=DEMUCS_INPUT_FRAC)
train_dbl = MoisesLeadRhythmDataset(DATA_DIR, dbl_train_ids, dbl, double_rhythm=True, blocked=DBL_BLOCKED,
                                    demucs_frac=DBL_DEMUCS_FRAC)
train_clean = MoisesLeadRhythmDataset(DATA_DIR, train_ids, usable)   # clean inputs, for the `train` eval line (as v5)
# v5: cross-song remixes from TRAINING songs only (val/test and the two-rhythm
# eval songs never contribute). Window lists computed locally by
# tools/remix_windows.py and embedded here.
REMIX_WINDOWS = __REMIX_WINDOWS__
_lead_tracks = [(usable[t]["song"], os.path.join(DATA_DIR, t, "guitar", usable[t]["lead_id"] + ".wav"),
                 set(REMIX_WINDOWS["lead_ok"].get(t, [])), "allow") for t in train_ids]
_rhy_tracks = [(usable[t]["song"], os.path.join(DATA_DIR, t, "guitar", usable[t]["rhythm_id"] + ".wav"),
                set(REMIX_WINDOWS["rhythm_bad"].get(t, [])), "block") for t in train_ids]
for t in dbl_train_ids:
    for k in ("lead_id", "rhythm_id"):
        _rhy_tracks.append((dbl[t]["song"], os.path.join(DATA_DIR, t, "guitar", dbl[t][k] + ".wav"),
                            set(DBL_BLOCKED.get(t, [])), "block"))
# v6: the 82 MoisesDB songs with ONE distorted guitar (SINGLE_DIR): confident lead
# windows (P(lead) > 0.8) and rhythm windows (< 0.2), tools/single_guitar_windows.py,
# embedded. All are training material. EXCLUDE_TEST_ARTISTS keeps the test songs'
# artists out (Firefly 74 + The Unfortunates 44 of the 181 new lead crops), so the
# test line still measures unseen artists rather than memorised rigs.
SINGLE_WINDOWS = __SINGLE_WINDOWS__
_test_artists = {usable[t]["artist"] for t in test_ids}
_n_single = 0
for t, v in (SINGLE_WINDOWS.items() if USE_SINGLE_GUITAR else []):
    if EXCLUDE_TEST_ARTISTS and v["artist"] in _test_artists:
        continue
    path = os.path.join(SINGLE_DIR, t, "guitar", v["file"] + ".wav")
    _lead_tracks.append(("single:" + t, path, set(v["lead_ok"]), "allow"))
    _rhy_tracks.append(("single:" + t, path, set(v["rhythm_ok"]), "allow"))
    _n_single += 1
print(f"single-guitar songs in the remix pool: {_n_single} (test artists excluded: {EXCLUDE_TEST_ARTISTS})")
train_remix = RemixDataset(_lead_tracks, _rhy_tracks,
                           bleed=BleedSource(train_ids + dbl_train_ids) if BLEED_FRAC > 0 else None)
print(f"Demucs input: pairs {DEMUCS_INPUT_FRAC:.0%}, two-rhythm {DBL_DEMUCS_FRAC:.0%} | bleed {BLEED_FRAC:.0%} "
      f"| single-guitar pool {USE_SINGLE_GUITAR}")
train_ds = MixedDataset(train_main, [(train_remix, REMIX_FRAC), (train_dbl, DOUBLE_RHYTHM_FRAC)])
print(f"training crops: {REMIX_FRAC:.0%} remix, {DOUBLE_RHYTHM_FRAC:.0%} two-rhythm, "
      f"{1 - REMIX_FRAC - DOUBLE_RHYTHM_FRAC:.0%} original song pairs")

dbl_eval_ds = MoisesLeadRhythmDataset(DATA_DIR, dbl_eval_ids, dbl, double_rhythm=True, blocked=DBL_BLOCKED)

# v5.1: fixed evaluation crops (same every epoch, so epochs are comparable)
test_fix = FixedCrops(test_ds, EVAL_N["test"], EVAL_SEED)
val_fix = FixedCrops(val_ds, EVAL_N["val"], EVAL_SEED + 1)
train_fix = FixedCrops(train_clean, EVAL_N["train"], EVAL_SEED + 2)
dbl_fix = FixedCrops(dbl_eval_ds, EVAL_N["dbl"], EVAL_SEED + 3)
test_dmx_fix = FixedCrops(test_dmx_ds, EVAL_N["test"], EVAL_SEED + 4)   # v6
print(f"fixed eval crops: test {EVAL_N['test']} ({', '.join(usable[t]['song'] for t in test_ids)}), "
      f"val {EVAL_N['val']} ({', '.join(usable[t]['song'] for t in val_ids)}), "
      f"train {EVAL_N['train']}, two-rhythm {EVAL_N['dbl']}")
"""

SMOKETEST_CODE = """\
# --- Smoke test: 3 batches with memory monitoring BEFORE committing to a
# real run (standing discipline since Stage 0's OOM). With AMP, the first 3-4
# steps are usually skipped while GradScaler calibrates, so identical-looking
# loss values here are expected (Stage 3 Update 5), not a dead optimizer.
# v3.2: fixed leak margins. Self-calibrating on TRAINING crops gave -31/-35dB
# for v3.1 epoch 3 although its held-out leak is ~-8dB: the model has partly
# memorised the training songs, so the margin came out unrealistically strict
# and the leak term penalised even good separation. -20dB ~ 10% amplitude bleed.
LEAK_MARGIN = (-20.0, -20.0)

# v4a: measure the inherited lead head's polarity on training crops where the
# lead plays, and store it in the wrapper. v3/v3.1 output the lead inverted on
# 100% of measured crops; uncorrected, rhythm = mix - lead would be mix + lead.
def calibrate_lead_sign(n=12):
    core.lead_sign.fill_(1.0)
    core.eval()
    votes = []
    with torch.no_grad():
        while len(votes) < n:
            mix, lead, _ = train_main.sample_batch(1)
            mix, lead = mix.to(DEVICE), lead.to(DEVICE)
            if level_db(lead, mix).item() <= ACTIVE_DB:
                continue
            e, r = _zm(_flat(core(mix)[:, 0])), _zm(_flat(lead))
            votes.append(1.0 if (e * r).sum().item() > 0 else -1.0)
    core.train()
    sign = 1.0 if sum(votes) >= 0 else -1.0
    core.lead_sign.fill_(sign)
    print(f"lead polarity votes: +{votes.count(1.0)} / -{votes.count(-1.0)}  -> lead_sign = {sign:+.0f}")
    if 0 < votes.count(1.0) < n and min(votes.count(1.0), votes.count(-1.0)) > n // 4:
        print("  WARNING: polarity is not consistent across crops -- mix - lead will be poor on some crops")

if MIX_CONSISTENT:
    calibrate_lead_sign()
_cal = leak_baseline(model, train_ds, n=10)
print(f"LEAK_MARGIN (fixed): {LEAK_MARGIN}  | train-crop leak, info only: {_cal[0]+2:.1f}dB, {_cal[1]+2:.1f}dB")

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
        src = (dataset.batches(batch_size) if hasattr(dataset, "batches")
               else (dataset.sample_batch(batch_size) for _ in range(n_batches)))
        for mix, lead, rhythm in src:
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
    res["n_both"] = int(w)
    return res


def fmt(r):
    return (f"sep={r['sep']:6.2f}dB role={r['role']:.2f} leak01={r['leak01']:6.1f} "
            f"leak10={r['leak10']:6.1f} buzz={r['buzz']:6.1f}dB(n={r['buzz_n']})\\n"
            f"         lead: SI-SDR={r['lead_q']:6.1f}dB level={r['lead_lvl']:+6.1f}dB | "
            f"rhythm: SI-SDR={r['rhy_q']:6.1f}dB level={r['rhy_lvl']:+6.1f}dB  [both-active n={r.get('n_both', '?')}]")


def evaluate_dbl(n_batches=6, batch_size=BATCH):
    \"\"\"v3: held-out two-rhythm-guitar songs. Right answer: lead output silent
    (lead_vs_mix -> -30dB or lower), rhythm output = both guitars at the right
    level (SI-SDR high, level ~0dB). This is The Sentinel's non-solo case.\"\"\"
    model.eval()
    lead_db, q, lvl = [], [], []
    with torch.no_grad():
        for mix, lead, rhythm in dbl_fix.batches(batch_size):   # v5.1: fixed crops
            mix, rhythm = mix.to(DEVICE), rhythm.to(DEVICE)
            out = model(mix)
            lead_db.append(level_db(out[:, 0], mix).clamp(min=-60).mean().item())
            q.append(sisdr_flat(out[:, 1], rhythm).mean().item())
            lvl.append(level_db(_zm(out[:, 1]), _zm(rhythm)).mean().item())
    model.train()
    return (f"two-rhythm: lead-out vs mix={np.mean(lead_db):6.1f}dB | "
            f"rhythm: SI-SDR={np.mean(q):6.1f}dB level={np.mean(lvl):+6.1f}dB")
"""

TRAIN_CODE = """\
# --- Main training loop. v3.1 warm-starts from INIT_CKPT (see config); with
# INIT_CKPT = None it starts fresh from pretrained HTDemucs.
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
print("INIT  test", fmt(evaluate(test_fix)))
print("INIT  tdmx", fmt(evaluate(test_dmx_fix)))   # v6: test songs, real-song (Demucs) input
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
    print("   val ", fmt(evaluate(val_fix)))
    print("   test", fmt(evaluate(test_fix)))
    print("   tdmx", fmt(evaluate(test_dmx_fix)))
    print("   " + evaluate_dbl())
    # v5: the same metrics on TRAINING songs (original pairs, not remixes). The
    # remix bet is that this train-vs-test gap narrows.
    print("  train", fmt(evaluate(train_fix)))
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
3. The real question for this run (v6a): does training on the Demucs input
   make the **real songs** cleaner (The Sentinel, Hot Blooded, Your Light at
   7.8s chunks, `tools/realsong_test_stereo.py` + `tools/timeline_outputs.py
   --solo ...`) than v5 epoch 12? `tools/input_mismatch_test.py` re-measures
   the clean-vs-Demucs input gap on the test songs.

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
    with open(os.path.join(os.path.dirname(__file__), "remix_windows.json")) as f:
        remix_windows = json.load(f)
    remix_windows = {k: remix_windows[k] for k in ("lead_ok", "rhythm_bad")}
    with open(os.path.join(os.path.dirname(__file__), "extra_guitar_roles.json")) as f:
        extra_roles = {t: {g: v["use"] for g, v in tr.items() if v["use"] in ("rhythm_extra", "block")}
                       for t, tr in json.load(f).items()}
    with open(os.path.join(os.path.dirname(__file__), "single_guitar_windows.json")) as f:
        single = {t: {k: v[k] for k in ("artist", "file", "lead_ok", "rhythm_ok")}
                  for t, v in json.load(f).items()}
    with open(os.path.join(os.path.dirname(__file__), "v6_bleed_db.json")) as f:
        bleed_db = json.load(f)["bleed_db"]
    split_code = (SPLIT_CODE.replace("__DBL_BLOCKED__", json.dumps(blocked))
                  .replace("__REMIX_WINDOWS__", json.dumps(remix_windows))
                  .replace("__EXTRA_ROLES__", json.dumps(extra_roles))
                  .replace("__SINGLE_WINDOWS__", json.dumps(single)))
    config_code = CONFIG_CODE.replace("__BLEED_DB__", repr(tuple(bleed_db)))
    return [
        cell(MD, INTRO_MD),
        cell(CODE, SETUP_CODE),
        cell(CODE, config_code),
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
# Colab: v6 needs three datasets (~16GB), too big for the upload widget. Put
# moisesdb_guitar.zip, moisesdb_guitar_single.zip and moisesdb_stems.zip (and the
# INIT_CKPT .pt) in Google Drive under MyDrive/stem_splitter/, then run this.
from google.colab import drive
import shutil
import zipfile

drive.mount("/content/drive")
SRC = "/content/drive/MyDrive/stem_splitter"
for z in ("moisesdb_guitar.zip", "moisesdb_guitar_single.zip", "moisesdb_stems.zip"):
    with zipfile.ZipFile(os.path.join(SRC, z)) as zf:
        zf.extractall("/content/data/" + z[:-4])
for f in os.listdir(SRC):
    if f.endswith(".pt"):
        shutil.copy(os.path.join(SRC, f), "/content/")
DATA_DIR = SINGLE_DIR = STEMS_DIR = None
for root, dirs, fs in os.walk("/content/data"):
    if "stems_manifest.json" in fs:
        STEMS_DIR = root
    elif "manifest.json" in fs and "role_labels.json" in fs:
        DATA_DIR = root
    elif "manifest.json" in fs:
        SINGLE_DIR = root
print("DATA_DIR:", DATA_DIR, "| SINGLE_DIR:", SINGLE_DIR, "| STEMS_DIR:", STEMS_DIR)
""")
    data_idx = next(i for i, c in enumerate(colab_cells)
                     if c["cell_type"] == CODE and "/kaggle/input" in "".join(c["source"]))
    colab_cells[data_idx] = colab_data_cell
    write_notebook("output/stage4_kaggle/Stage4_HTDemucs_MoisesDB_Colab.ipynb", colab_cells)


if __name__ == "__main__":
    main()
