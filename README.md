# Stem Splitter

A small local GUI for splitting a music track into stems (vocals, drums,
bass, guitar, piano, other).

## How it works

This project doesn't train its own separation model. Getting a from-scratch
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
synthetic tests and a real track (details in `CLAUDE.md`) to give a real,
measurable improvement when the source material is panned apart, and
correctly no improvement when it's centered/mono -- that's a hard limit of
using stereo position as the only cue, not a bug.

Full source *identification* (knowing these are "guitar 1" and "guitar 2"
regardless of mix position, or separating two centered sources) is still
unsolved and would be a real research project. See `CLAUDE.md` for the
state of that.
