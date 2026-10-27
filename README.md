# Stem Splitter

A small local GUI for splitting a music track into stems (vocals, drums,
bass, guitar, piano, other).

## How it works

The app doesn't train its own separation model. Getting a from-scratch
model to generalize to arbitrary songs would need a large multi-track
dataset (e.g. MUSDB18) and days of GPU training, with no guarantee of
beating existing work. Instead, this wraps
[Demucs](https://github.com/facebookresearch/demucs) -- Meta's pretrained,
MIT-licensed, actively maintained source-separation model, currently the
strongest well-supported open option for this.

Two presets are exposed in the GUI:

- **4-stem (best quality)** -- `htdemucs_ft`, Meta's fine-tuned model.
  Produces vocals / drums / bass / other. The best-quality option for these
  four.
- **6-stem** -- `htdemucs_6s`. Same architecture, also splits out guitar and
  piano, at slightly lower quality on the core four.

The first time you run a separation, Demucs downloads the chosen model
(a few hundred MB) and caches it under `~/.cache/torch`.

## Setup

```bash
python3 -m venv ~/pytorch_env
source ~/pytorch_env/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install demucs gradio
```

(CPU-only torch build -- smaller download, and GPU is optional/opt-in in
the app anyway. `requirements.txt` lists the same packages if you'd rather
`pip install -r requirements.txt` with the default GPU-enabled build.)

## Usage

```bash
python app.py
```

Then open the printed local URL (typically http://127.0.0.1:7860) in a
browser, upload a track, pick a mode, and hit Separate. Stems play in the
browser and are also saved to `output/<track name>/<stem>.wav`.

GPU use is optional (there's a checkbox) and auto-falls-back to CPU if the
GPU run fails or runs out of memory. CPU-only works fine, just slower --
budget a few minutes per song.

## Splitting multiple guitars (beta)

**Separating multiple simultaneous performers of the *same* instrument**
(e.g. two overlapping guitar parts into two separate tracks) was the
original ask for this project. There's still no robust pretrained *model*
for true source separation of this kind -- stem separation models key off
*timbral differences between instrument classes*, and two guitars in the
same mix share timbre, register, and often rhythm, which is the hardest
case in the field. Narrow academic research exists (score-informed or
MIDI-conditioned separation, datasets like
[GuitarDuets](https://arxiv.org/pdf/2507.01172)) but needs information the
GUI doesn't have and doesn't generalize to an arbitrary uploaded song.

What *does* work, cheaply and without any model: a lot of dual-guitar mixes
-- especially double-tracked guitars, the standard rock/metal production
technique of recording the same part twice and panning the takes apart for
width -- are distinguishable by stereo position even when nothing else
distinguishes them. The **"Split a stem by stereo position (beta)"**
section (after separating) uses that: pick a stem (usually guitar), and it
splits it into a left-leaning and right-leaning pair. Validated on
synthetic tests and a real track to give a real,
measurable improvement when the source material is panned apart, and
correctly no improvement when it's centered/mono -- that's a hard limit of
using stereo position as the only cue, not a bug.

Full source *identification* (knowing these are "guitar 1" and "guitar 2"
regardless of mix position, or separating two centered sources) is still
unsolved. The research below works on a more tractable version of it.

## Research: separating lead guitar from rhythm guitar (work in progress)

Not part of the app yet. This is an ongoing experiment toward the original
goal, aimed at rock/metal with two guitarists.

### The reframe that made progress possible

Splitting "guitarist A" from "guitarist B" failed repeatedly. That label is
arbitrary: nothing in the audio says which player belongs in which output.
Splitting by **role** (lead/solo vs. rhythm/accompaniment) is learnable,
because register, sustain, note density and phrasing differ. So the model
has two outputs, `lead` and `rhythm`, and should put whoever is currently
soloing into `lead`, whichever guitarist that is.

### Where it stands

The earlier attempts trained a 5M-parameter Conv-TasNet on synthetic
mixtures of acoustic GuitarSet clips run through a distortion chain. Every
fix we tried moved the result along one trade-off between a dull rhythm
track and solo bleed, and nothing broke past it.

The current approach fine-tunes **pretrained HTDemucs** (42M parameters,
44.1kHz, trained on real music) on real multitrack recordings from
[MoisesDB](https://github.com/moises-ai/moises-db):

- **14 training songs** with separate lead and rhythm distorted-guitar
  tracks. The roles were labelled by a small classifier trained on
  GuitarSet's solo/comp recordings (97% held-out accuracy).
- **9 songs with two rhythm guitars and no lead.** Real twin-guitar bands
  spend most of a song like this, and without them the model tried to carve
  a "lead" out of every two-guitar passage. Moments the classifier rates
  lead-like are excluded.
- **Evaluation:** 2 + 2 held-out lead/rhythm songs, 2 held-out two-rhythm
  songs, and listening tests on Judas Priest's "The Sentinel".

Best result so far (v3, epoch 5): on "The Sentinel" the lead output isolates
the solos almost completely with very little rhythm bleed, and stays nearly
silent outside them. The rhythm output is clean in non-solo passages but
still lets the solos bleed through. There's also some buzzing, and its real
guitar content came out too quiet because of a loss bug (see "DC" below).
v3.1, warm-started from that checkpoint with the bug fixed and the loss
rebalanced, measured better at epoch 3: on "The Sentinel" the lead output
is louder than the rhythm output during both solos, and the rhythm output
is about 8dB fuller in non-solo passages. By ear, though, it sounds about
the same as v3 epoch 5. After epoch 3 the rhythm output drifted into
copying the input. v3.2 added a loss term against that, and after 15 epochs
no checkpoint beat its starting point: it was the third run in a row to
drift toward "rhythm output = the whole input, lead output quiet".

Tested on two more songs ("Hot Blooded" by Foreigner, and "Your Light"), the
lead output is mixed: good on "Your Light", especially the outro solo, and
weak on "Hot Blooded". On all three songs the rhythm output carries the
whole solo.

v4a computed the rhythm output as *input minus lead output*. It didn't
help: the rhythm track can only be as clean as the lead estimate, and on
new songs the lead estimate was poor. That pointed at the real problem: the
model scored far better on its 14 training songs than on new ones. Every
loss or structure change had only moved where the error landed.

**v5: cross-song remixing.** Half the training crops pair one song's lead
track with another song's rhythm track (mostly key-matched, listening-
approved), the way Demucs itself is trained. This is the first real
progress on real songs: on "The Sentinel" the rhythm track is "much more
clear, no noises", and on "Your Light" both tracks are the best yet, with
very weak bleed. Running the model on its native 7.8-second window, rather
than 4-second chunks padded with silence, removes most moments where the
solo vanishes from the lead track and sounds best by ear.

v5.1 trained on 7.8-second crops and evaluated on a fixed set of crops
(earlier per-epoch numbers jumped 3-5dB from random crop choice alone). It
tied v5 by ear, so v5 epoch 12 stays the reference model.

**The biggest remaining loss is the input.** Real songs reach the model as a
guitar stem that Demucs extracted from the finished mix: bleed from other
instruments, separation artifacts, extra guitar parts. Training only ever used
clean guitar tracks. On two held-out songs, the same model loses 6-10dB when
fed the Demucs-extracted stem instead of the clean tracks.

**Training on the Demucs input didn't help (v6, v6a).** v6 trained partly on
Demucs-extracted inputs, added other-instrument bleed to remixed crops, and
added lead material from 82 more MoisesDB songs. v6a kept only the Demucs
inputs. Neither improved held-out songs fed the Demucs stem, and both swung
back and forth between "quiet lead, solo in the rhythm track" and the reverse.

An oracle test explains why. The best possible mask applied to the Demucs
stem recovers the lead at 6.6dB and the rhythm at 5.7dB, against about 13dB
from the clean tracks: Demucs has already thrown the rest away. The model
already loses no more than that, so there was little to gain by training,
and some training examples asked for audio the input no longer contains.

**So the gap is in the guitar extraction, not in the lead/rhythm model.**
On the same windows, BS-Roformer SW (a community model, currently the best
on MVSep's guitar benchmark) extracts the guitar much more faithfully than
htdemucs_6s: 7.8 vs 5.4dB to the true guitars, oracle lead 9.0 vs 6.6dB,
better in over 90% of windows. Two caveats: its training data is unknown
and may include MoisesDB, and its weights have no stated licence. **Next:**
listen to the lead/rhythm model fed Roformer stems of The Sentinel, Your
Light and Hot Blooded, songs outside every dataset.

Open problems:

- **Bleed at specific moments:** a few seconds where the solo jumps into the
  rhythm track, often at a handoff between guitarists.
- **Twin leads** (both guitars soloing) bleed into the rhythm track. The
  training data has almost none.
- **"Hot Blooded"** still separates poorly by ear.
- **Real-song input:** the Demucs-extracted guitar stem has lost part of the
  guitar before the model sees it (see above).

Earlier we suspected the role labels (each MoisesDB track is labelled lead
or rhythm as a whole). Blind listening tests mostly ruled that out: heard
together with the other guitar, the labels were mostly right. Whether a part
is lead depends on what plays under it, so a classifier that hears one track
alone can't relabel them.

MoisesDB is licensed CC BY-NC-SA 4.0, so models trained on it are for
non-commercial use only.

### The loss function

Each training example is a 4-second stereo crop:
`mixture = lead track + rhythm track`. Either track may be silent in that
crop: the lead rests during verses, and in two-rhythm songs the lead target
is silent throughout. The loss handles each output differently depending on
whether its guitar is playing. A guitar counts as playing if its target is
louder than -20dB relative to the mixture.

**1. Reconstruction, when the guitar is playing.**

    -SI-SDR(output, target)  +  1.0 * level_error  +  1.0 * dc_penalty

- **SI-SDR** (scale-invariant signal-to-distortion ratio) scores how well
  the output's *waveform* matches the target, ignoring overall loudness.
  Ignoring loudness matters. An earlier version used a loudness-aware score,
  and whenever the model was unsure, its best move under the loss was to
  turn the output down. The lead output went effectively silent for a whole
  20-epoch run.
- **level_error** brings loudness back in, but softly: the difference
  between output and target level in dB, counted only beyond a ±3dB
  deadzone. Without it, an output can drift arbitrarily loud. One earlier
  version came out 11dB too loud and full of noise.
- **dc_penalty** pushes any constant offset (DC) in the output below -30dB
  of the target. Level is measured with DC removed. Otherwise the model can
  meet the level target with an inaudible constant offset while the real
  guitar stays quiet. It found exactly that loophole in v3.

**2. Silence, when the guitar isn't playing.**

    2.0 * -10*log10( (tau*|mixture|^2) / (|output|^2 + tau*|mixture|^2) )      tau = -30dB

This is zero when the output is silent and rises toward 30 as the output gets
as loud as the mixture. It's bounded both ways and stays well-defined when
the target is pure silence; SI-SDR breaks there. This term is what removes
the buzzing that earlier versions produced in passages where one guitar
rests. The weight is a balance. At 4.0 the lead output learned that staying
quiet was the safe bet and slid back to silence; 2.0 is the current
compromise.

**3. Which output is which.**

- **Both guitars playing:** permutation-invariant training (PIT). The loss
  takes the better of the two output-to-target assignments, so a model that
  separates well but has the outputs swapped isn't punished harder than one
  that outputs the mixture twice. An earlier experiment collapsed for
  exactly that reason.
- **Only one guitar playing:** the assignment is fixed (lead → output 0,
  rhythm → output 1). Otherwise "rhythm in the lead slot" would score just
  as well, and nothing would teach routing in non-solo passages.

**4. Routing hinges**, both guitars playing, weight 3.0:

    max(0, 1dB - (SI-SDR(out0, lead)   - SI-SDR(out0, rhythm)))
  + max(0, 3dB - (SI-SDR(out1, rhythm) - SI-SDR(out1, lead)))

The first term teaches "output 0 is the lead": output 0 must resemble the
lead at least 1dB more than the rhythm. The second is the mirror for output
1, added in v3.2. Without it the rhythm output drifted into simply copying
the input. That's the right answer whenever the lead rests (about 40% of
crops), and copying is the easiest thing a network can learn. Both terms are
bounded: once they hold, they contribute nothing.

**5. Leak penalty**, both guitars playing, weight 1.0:

    max(0, SI-SDR(out0, rhythm) - m) + max(0, SI-SDR(out1, lead) - m)

This penalises each output for containing the *other* guitar, which is the
bleed you hear. The margin `m` is fixed at -20dB, roughly 10% amplitude
bleed. Earlier versions calibrated it from the model's leakage on training
crops. The model had partly memorised those, so the margin came out far too
strict (-35dB) and the term punished even good separation.

**Total:** `reconstruction-or-silence (averaged over both outputs) + 3.0 * hinge + 1.0 * leak`

### Hyperparameters (v5.1) and why

| Setting | Value | Why |
|---|---|---|
| Outputs | Two independent outputs | v4a's rhythm = input minus lead didn't help (the rhythm can only be as clean as the lead estimate) |
| Training data | 50% cross-song remixes, 40% original song pairs, 10% two-rhythm songs | The model memorised its 14 songs; remixing turns 14 pairs into hundreds of combinations |
| Remix pairing | 75% key-matched (random among the 8 best of 64 candidates, skipping same-note pairs), 25% random | Matching on exact notes buried the lead under a rhythm part playing the same note |
| Remix level | Lead vs rhythm drawn from ±6dB | Both parts always play, so setting levels per part is safe |
| Sample rate / crop | 44.1kHz stereo, 7.8s | HTDemucs's native rate and window: no resampling, no padding, so training matches inference |
| Batch | 4 (2 per GPU, 2× T4) | 7.8s crops need about twice the memory of 4s |
| Evaluation | Fixed crops: 40 test, 40 validation, 24 training, 16 two-rhythm | Random per-epoch crops made scores jump 3-5dB between epochs |
| Steps / epochs | 200 per epoch, 15 epochs | Data is cropped on the fly, so an "epoch" is just a reporting interval |
| Peak learning rate | 1e-4 | Fine-tuning from the best previous checkpoint. Fresh starts used 2e-4: 1e-4 stalled, 5e-4 diverged |
| LR schedule | 1-epoch linear warmup, cosine decay to 5% | A flat 2e-4 hit NaN twice around epochs 5-8 |
| Gradient clipping | norm 5.0 | Extra insurance against those divergences |
| Silence weight | 2.0 | 4.0 made a quiet lead output the safe bet (see above) |
| Level weight / deadzone | 1.0, ±3dB | Keeps outputs at the right loudness without being a second content score |
| Hinge weight / margins | 3.0; 1dB (lead), 3dB (rhythm) | Weaker weights didn't move routing in earlier experiments; the rhythm hinge counters the copy-the-input shortcut |
| Leak weight / margin | 1.0, -20dB for each output | Bounded, so it can't trade away fidelity indefinitely; fixed because self-calibration was fooled by memorised training crops |
| "Playing" threshold | -20dB vs mixture | Below this, a track counts as resting: bleed or amp hiss, not playing |
| Two-rhythm share | 10% of crops | 25% over-taught "two guitars → all rhythm", so solos leaked into the rhythm output |
| Overlap bias | 70% of original-pair crops centred where both guitars play | Simultaneous playing is the hard case; the other 30% teaches silence and solo-only passages |
| Gain | One shared gain per crop, mixture RMS 0.1 | Normalising each track separately boosted a resting track's amp hiss 40-90dB into a full-loudness target, the root cause of the original buzzing |

### Reproducing

- `tools/build_stage4_notebook.py` generates the training notebook (Kaggle
  and Colab versions) into `output/stage4_kaggle/`.
- `tools/pull_moisesdb_guitar.py` fetches only the guitar stems of the
  qualifying MoisesDB songs (~6GB of the 83GB archive).
- `tools/diagnose_htd_ckpt.py` scores a checkpoint per output: content
  quality and level, on held-out songs.
- `tools/realsong_test_stereo.py` runs a checkpoint on a real song for
  listening.
- `tools/remix_windows.py` lists which windows of each track can be used in
  cross-song remixes (the lead must sound like lead on its own).
- `tools/input_mismatch_test.py` scores a checkpoint on held-out songs fed
  three ways: clean lead + rhythm, all guitar tracks, and the Demucs-extracted
  guitar of the full mix.
- `tools/pull_moisesdb_stems.py` and `tools/demucs_guitar_inputs.py` rebuild
  each pair song's full mix and its htdemucs_6s guitar stem (the v6 input).
- `tools/oracle_demucs_input.py` measures how much of the lead and rhythm can
  still be recovered from an extracted guitar stem (best possible mask).
- `tools/roformer_vs_demucs.py` compares BS-Roformer SW with htdemucs_6s as
  the guitar extractor on those windows; `tools/roformer_guitar_stem.py`
  extracts a whole song's guitar with it (ONNX, CPU, ~7x song length).
- `tools/timeline_outputs.py` scores real-song outputs against solo windows
  marked by ear: how much the lead output rises in the solos, and how much
  of the solo the rhythm output carries.

The most important lesson so far: numbers measured on the training
material have picked the wrong checkpoint many times. Checkpoints are chosen
by listening to real songs.
