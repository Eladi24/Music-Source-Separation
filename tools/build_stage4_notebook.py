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

## What to watch

- **`best-perm`** -- separation quality regardless of routing, same as
  every prior stage. Init will be near 0dB or negative (random head) --
  watch it climb like every previous warm start.
- **`role-acc`** -- fraction where out0 is closer to lead than rhythm.
  ~50% = not working.
- **`leak(out0->rhythm)` / `leak(out1->lead)`** -- bleed, same mechanism
  as Stage 3's leak-loss fix. Self-calibrated margin at init, same as
  before.
- Band-balance loss (the treble-starvation fix) is **deliberately not
  included in this first run** -- this is already a compound change
  (new architecture + new data), and adding a third lever at once would
  make a bad result hard to diagnose. If this run succeeds, band-balance
  is the natural next addition, and it may finally have real signal to
  learn from (MoisesDB has real 2-4kHz content, unlike GuitarSet).
- **Real-song test**: `tools/realsong_test.py` was built for the 16kHz
  mono ConvTasNet and needs adapting for stereo/44.1kHz before it can
  score this checkpoint -- not yet done, flagged as the next concrete
  step once a checkpoint exists to test.
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
N_EPOCHS = 15
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
# --- Loss: PIT-preserving reconstruction + bounded routing hinge + bounded
# leakage penalty -- unchanged mechanism from Stage 2/3 (see CLAUDE.md), just
# shaped for HTDemucs's (B, S=2, C=2, T) stereo output instead of the
# ConvTasNet's mono (B, S, T). sisdr_elem only reduces over the last dim, so
# it already broadcasts correctly over the extra channel dimension -- verified
# locally before this notebook was written.

def sisdr_elem(est, ref, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10((t ** 2).sum(-1) / (((est - t) ** 2).sum(-1) + eps) + eps)


def combined_loss(est, LEAD, RHYTHM, margin=1.0, leak_margin=None, lambda_leak=0.0):
    s00, s11 = sisdr_elem(est[:, 0], LEAD), sisdr_elem(est[:, 1], RHYTHM)
    s01, s10 = sisdr_elem(est[:, 0], RHYTHM), sisdr_elem(est[:, 1], LEAD)
    l_same = -(s00 + s11) / 2
    l_swap = -(s01 + s10) / 2
    pit_loss = torch.minimum(l_same, l_swap).mean()
    hinge = F.relu(margin - (s00 - s01)).mean()
    best_perm = torch.maximum(-l_same, -l_swap)
    if lambda_leak > 0 and leak_margin is not None:
        leak = (F.relu(s01 - leak_margin).mean() + F.relu(s10 - leak_margin).mean()) / 2
    else:
        leak = torch.zeros((), device=est.device)
    role_acc = (s00 > s01).float().mean()
    return pit_loss, hinge, leak, best_perm, role_acc, s01.detach(), s10.detach()


def leak_baseline(model, dataset, n=20):
    \"\"\"Self-calibrate LEAK_MARGIN from the model's own init leakage, same as
    the original leak-loss experiment (margin = min(baseline s01, s10) - 2dB).\"\"\"
    model.eval()
    s01s, s10s = [], []
    with torch.no_grad():
        for _ in range(n):
            mix, lead, rhythm = dataset.sample_batch(1)
            mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
            out = model(mix)
            _, _, _, _, _, s01, s10 = combined_loss(out, lead, rhythm)
            s01s.append(s01.mean().item())
            s10s.append(s10.mean().item())
    model.train()
    baseline = min(sum(s01s) / n, sum(s10s) / n)
    return baseline - 2.0
"""

DATASET_CODE = """\
# --- Streaming dataset: real audio, on-the-fly random crops, biased toward
# measured overlap windows (see markdown above). No synthetic DSP, no
# precompute -- see tools/moisesdb_dataset.py (validated locally) for the
# identical logic this is copied from.

def rms_norm(x, target=0.1, floor=1e-6):
    rms = np.sqrt(np.mean(x ** 2))
    return x if rms < floor else x * (target / rms)


def find_active_windows(path, win_sec=2.0):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    win = int(win_sec * sr)
    n = len(mono) // win
    rms = np.array([np.sqrt(np.mean(mono[i*win:(i+1)*win] ** 2)) for i in range(n)])
    return rms, sr, len(mono)


class MoisesLeadRhythmDataset:
    def __init__(self, data_dir, track_ids, labels_by_id, seg_sec=SEG_SEC,
                 overlap_bias=0.7, active_thresh=0.01):
        self.seg_sec = seg_sec
        self.overlap_bias = overlap_bias
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
            self.songs.append({
                "track_id": tid, "artist": r["artist"], "song": r["song"],
                "lead_path": lead_path, "rhythm_path": rhythm_path,
                "n_samples": min(n_lead, n_rhythm), "overlap_windows": both,
                "overlap_frac": len(both) / max(1, n),
            })

    def _sample_start(self, song):
        seg_n = int(self.seg_sec * SR)
        max_start = max(1, song["n_samples"] - seg_n)
        if len(song["overlap_windows"]) > 0 and random.random() < self.overlap_bias:
            center = int(random.choice(song["overlap_windows"]) * 2.0 * SR) + SR
            start = max(0, min(max_start, center - seg_n // 2))
        else:
            start = random.randint(0, max_start)
        return start

    def sample_batch(self, batch_size):
        seg_n = int(self.seg_sec * SR)
        leads, rhythms = [], []
        for _ in range(batch_size):
            song = random.choice(self.songs)
            start = self._sample_start(song)
            lead, _ = sf.read(song["lead_path"], dtype="float32", always_2d=True, start=start, frames=seg_n)
            rhythm, _ = sf.read(song["rhythm_path"], dtype="float32", always_2d=True, start=start, frames=seg_n)
            if lead.shape[0] < seg_n:
                lead = np.pad(lead, ((0, seg_n - lead.shape[0]), (0, 0)))
            if rhythm.shape[0] < seg_n:
                rhythm = np.pad(rhythm, ((0, seg_n - rhythm.shape[0]), (0, 0)))
            lead, rhythm = rms_norm(lead), rms_norm(rhythm)
            peak = np.abs(lead + rhythm).max()
            if peak > 0.95:
                scale = 0.95 / peak
                lead, rhythm = lead * scale, rhythm * scale
            leads.append(lead.T)
            rhythms.append(rhythm.T)
        lead_t = torch.from_numpy(np.stack(leads)).float()
        rhythm_t = torch.from_numpy(np.stack(rhythms)).float()
        return lead_t + rhythm_t, lead_t, rhythm_t
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

train_ds = MoisesLeadRhythmDataset(DATA_DIR, train_ids, usable)
val_ds = MoisesLeadRhythmDataset(DATA_DIR, val_ids, usable)
test_ds = MoisesLeadRhythmDataset(DATA_DIR, test_ids, usable)
"""

SMOKETEST_CODE = """\
# --- Smoke test: 3 batches with memory monitoring BEFORE committing to a
# real run. Standing discipline since the Stage 0 OOM lesson (hid_chan
# left at Asteroid's default blew memory, caught only by this kind of test).
LEAK_MARGIN = leak_baseline(model, train_ds, n=10)
print(f"self-calibrated LEAK_MARGIN: {LEAK_MARGIN:.2f}dB")

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

for i in range(3):
    mix, lead, rhythm = train_ds.sample_batch(BATCH)
    mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
    opt.zero_grad()
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        out = model(mix)
        pit, hinge, leak, best_perm, role_acc, _, _ = combined_loss(
            out, lead, rhythm, leak_margin=LEAK_MARGIN, lambda_leak=LAMBDA_LEAK)
        loss = pit + LAMBDA_HINGE * hinge + LAMBDA_LEAK * leak
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if DEVICE == "cuda":
        print(f"  batch {i}: loss={loss.item():.3f} best_perm={best_perm.mean().item():.2f}dB "
              f"role_acc={role_acc.item():.2f}  gpu_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    else:
        print(f"  batch {i}: loss={loss.item():.3f} best_perm={best_perm.mean().item():.2f}dB role_acc={role_acc.item():.2f}")

print("\\nSmoke test passed -- note the GPU memory above and adjust BATCH upward if there's headroom "
      "(it's a conservative starting guess) before running the full loop below.")
"""

EVAL_CODE = """\
def evaluate(dataset, n_batches=10, batch_size=BATCH):
    model.eval()
    tot_perm, tot_role, tot_leak01, tot_leak10, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            mix, lead, rhythm = dataset.sample_batch(batch_size)
            mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
            out = model(mix)
            _, _, _, best_perm, role_acc, s01, s10 = combined_loss(
                out, lead, rhythm, leak_margin=LEAK_MARGIN, lambda_leak=LAMBDA_LEAK)
            tot_perm += best_perm.mean().item()
            tot_role += role_acc.item()
            tot_leak01 += s01.mean().item()
            tot_leak10 += s10.mean().item()
            n += 1
    model.train()
    return tot_perm / n, tot_role / n, tot_leak01 / n, tot_leak10 / n
"""

TRAIN_CODE = """\
# --- Main training loop. Checkpoint saved every epoch (lesson from Stage 2's
# save-on-"new best"-only bug, which silently skipped the actual best epoch).
STEPS_PER_EPOCH = 200
os.makedirs("/kaggle/working", exist_ok=True)

init_perm, init_role, init_l01, init_l10 = evaluate(test_ds, n_batches=10)
print(f"INIT  test best_perm={init_perm:.2f}dB role_acc={init_role:.2f} "
      f"leak(out0->rhythm)={init_l01:.1f}dB leak(out1->lead)={init_l10:.1f}dB")
print("(best_perm near 0dB or negative is EXPECTED at init -- the output head is random)")

for epoch in range(N_EPOCHS):
    for step in range(STEPS_PER_EPOCH):
        mix, lead, rhythm = train_ds.sample_batch(BATCH)
        mix, lead, rhythm = mix.to(DEVICE), lead.to(DEVICE), rhythm.to(DEVICE)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            out = model(mix)
            pit, hinge, leak, best_perm, role_acc, _, _ = combined_loss(
                out, lead, rhythm, leak_margin=LEAK_MARGIN, lambda_leak=LAMBDA_LEAK)
            loss = pit + LAMBDA_HINGE * hinge + LAMBDA_LEAK * leak
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    val_perm, val_role, val_l01, val_l10 = evaluate(val_ds, n_batches=10)
    test_perm, test_role, test_l01, test_l10 = evaluate(test_ds, n_batches=10)
    print(f"epoch {epoch+1:2d}  val: perm={val_perm:6.2f}dB role={val_role:.2f}  "
          f"test: perm={test_perm:6.2f}dB role={test_role:.2f} "
          f"leak01={test_l01:.1f}dB leak10={test_l10:.1f}dB")

    ckpt_path = f"/kaggle/working/htd_leadrhythm_epoch{epoch+1}.pt"
    torch.save(core.state_dict(), ckpt_path)
    print(f"  saved {ckpt_path}")
"""

NEXT_MD = """\
## After this run

1. Download the checkpoint from each epoch worth keeping (Kaggle's Output
   panel, same as every prior stage -- don't wait until the end, per the
   standing "download periodically during a run" lesson from Stage 3).
2. `tools/realsong_test.py` needs adapting for stereo/44.1kHz output before
   it can score this checkpoint on The Sentinel etc. -- not yet built.
3. Select the checkpoint to keep by the real-song test, never by these
   in-domain numbers alone -- the project's own repeated, measured lesson
   (in-domain leak inverted the real-song ranking three separate times in
   Stage 3's Update 5/6).
4. If this succeeds: band-balance loss (`LAMBDA_BAND`) is the natural next
   addition, since MoisesDB may finally have real signal for it. Add it as
   its own isolated change, not bundled with anything else.
"""


def build_cells():
    return [
        cell(MD, INTRO_MD),
        cell(CODE, SETUP_CODE),
        cell(CODE, CONFIG_CODE),
        cell(CODE, DATA_LOCATE_CODE),
        cell(CODE, MODEL_CODE),
        cell(CODE, LOSS_CODE),
        cell(CODE, DATASET_CODE),
        cell(CODE, SPLIT_CODE),
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
