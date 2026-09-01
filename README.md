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
pip install -r requirements.txt
```

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

## Future ideas (not implemented)

**Separating multiple simultaneous performers of the *same* instrument**
(e.g. two overlapping guitar parts into two separate tracks) was the
original ask for this project, but there's currently no robust pretrained
model for it. Stem separation models key off *timbral differences between
instrument classes* (vocals vs. guitar vs. drums); two guitars in the same
mix share timbre, register, and often rhythm, which is the hardest case in
the field. What exists is narrow academic research (e.g. score-informed or
MIDI-conditioned separation, or datasets like
[GuitarDuets](https://arxiv.org/pdf/2507.01172)) that needs extra
information (a score, a second modality) and doesn't generalize to an
arbitrary uploaded song. Revisit this if/when that research matures into
something that runs on unlabeled audio; for now the 6-stem mode's single
combined "guitar" stem is the practical ceiling.
