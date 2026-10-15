"""Real-song lead/rhythm test + cross-chunk routing diagnostic.

Persistent local version of the notebook's cells 15-16. This test has been
recreated from scratch at least three times after /tmp scratchpad clears --
it lives in the repo now. Pure inference, runs fine on CPU (training does
not: see CLAUDE.md's FULL-config memory wall).

Modes:
  naive      -- the notebook's current path: chunk's est[0] always goes to
                out0, no cross-chunk agreement check. Reproduces the
                numbers recorded in CLAUDE.md.
  continuity -- pick each chunk's (A,B) vs (B,A) ordering by agreement with
                the already-reconstructed overlap region.

--diagnose measures how often the two orderings actually disagree, and how
confidently, per chunk -- turning "we hypothesise routing instability" into
a number before any fix is written.
"""
import argparse, wave, sys
import numpy as np
import torch

SR = 16000
SEG = SR * 2
FULL = dict(n_filters=512, n_blocks=8, n_repeats=3,
            bn_chan=128, hid_chan=512, skip_chan=128)   # epoch-50 config, unchanged since Stage 1

TIPTON_SOLO, DOWNING_SOLO = (141.0, 146.0), (147.0, 151.0)
BLEED_SPIKE = (161.0, 167.0)   # ~2:41-2:47, user-reported on lead5_epoch1


def load_wav(path, target_sr=SR):
    with wave.open(str(path), 'rb') as f:
        sr, n, sw, ch = f.getframerate(), f.getnframes(), f.getsampwidth(), f.getnchannels()
        raw = f.readframes(n)
    x = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[sw]).astype(np.float64)
    x /= float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != target_sr:
        m = int(len(x) * target_sr / sr)
        x = np.interp(np.linspace(0, len(x), m, endpoint=False), np.arange(len(x)), x)
    return x.astype(np.float32)


def write_wav(path, x, sr=SR):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def build_model(ckpt_path, device):
    from asteroid.models import ConvTasNet
    model = ConvTasNet(n_src=2, sample_rate=SR, **FULL)
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f'  WARNING load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected', file=sys.stderr)
        if missing:
            print(f'    missing e.g. {missing[:3]}', file=sys.stderr)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'  model: {n:,} params, loaded {len(sd)} keys from {ckpt_path}')
    return model


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _env(x, frame=160):
    """Short-frame RMS envelope -- a coarser agreement cue than raw samples,
    robust to phase differences between two reconstructions of overlapping
    content."""
    n = (len(x) // frame) * frame
    if n == 0:
        return np.zeros(1)
    return np.sqrt((x[:n].reshape(-1, frame) ** 2).mean(1) + 1e-12)


@torch.no_grad()
def separate_real(model, audio, device, mode='naive', diagnose=False, chunk=SEG,
                  gain_fix='none', lam=0.1):
    """Two-pass: run every chunk, then decide orientation and per-chunk gain
    across the whole sequence, then overlap-add. One pass cannot do this --
    both corrections need to see chunks that have not been processed yet."""
    try:        # run as a script: sys.path[0] is tools/
        from stitch import (viterbi_orient, greedy_orient, solve_gains,
                            overlap_add, measure_log_alpha)
    except ImportError:   # imported as tools.realsong_test from the repo root
        from tools.stitch import (viterbi_orient, greedy_orient, solve_gains,
                                  overlap_add, measure_log_alpha)
    hop = chunk // 2
    starts = list(range(0, max(len(audio) - chunk + 1, 1), hop))
    chunks = []
    for s in starts:
        seg = audio[s:s + chunk]
        if len(seg) < chunk:
            seg = np.pad(seg, (0, chunk - len(seg)))
        c = torch.from_numpy(seg).float().unsqueeze(0).to(device)
        e = model(c).squeeze(0).float().cpu().numpy()
        chunks.append((e[0], e[1]))

    if mode == 'continuity':
        swap = viterbi_orient(chunks)
    elif mode == 'greedy':
        swap = greedy_orient(chunks)
    else:
        swap = np.zeros(len(chunks), dtype=bool)
    oriented = [(e1, e0) if sw else (e0, e1) for (e0, e1), sw in zip(chunks, swap)]

    g = solve_gains(oriented, lam=lam) if gain_fix == 'ratio' else np.ones((len(oriented), 2))
    corrected = [(e0 * g[i, 0], e1 * g[i, 1]) for i, (e0, e1) in enumerate(oriented)]

    info = {}
    if diagnose:
        info['swap'] = swap
        info['starts'] = starts
        info['gains'] = g
        for tag, seq in (('before', oriented), ('after', corrected)):
            d0 = np.abs(measure_log_alpha(seq, 0)[1:]) * 20 / np.log(10)
            d1 = np.abs(measure_log_alpha(seq, 1)[1:]) * 20 / np.log(10)
            info[tag] = (d0, d1)
    o0, o1 = overlap_add(corrected, np.zeros(len(corrected), dtype=bool), len(audio), starts)
    return o0, o1, info


def rms_win(x, a, b):
    seg = x[int(a * SR):int(b * SR)]
    return float(np.sqrt(np.mean(seg ** 2))) if len(seg) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--stem', default='output/04 - The Sentinel/guitar.wav')
    ap.add_argument('--mode', default='naive', choices=['naive', 'greedy', 'continuity'])
    ap.add_argument('--diagnose', action='store_true')
    ap.add_argument('--gain-fix', default='none', choices=['none', 'ratio'],
                    help="ratio: least-squares per-chunk gain so overlapping chunks agree "
                         "on level. SI-SDR is scale-invariant, so training never constrained "
                         "how much of the mixture each output claims; measured allocation "
                         "swing between out0/out1 is up to 8x chunk-to-chunk. A mixture-anchored "
                         "correction does NOT work here -- measurement showed overall level is "
                         "already stable (beta sd 0.005), only the split varies.")
    ap.add_argument('--lam', type=float, default=0.1,
                    help='gain-fix regularisation: pulls gains toward the model native output')
    ap.add_argument('--chunk-sec', type=float, default=2.0,
                    help='inference chunk length. The model was TRAINED on 2s but is a '
                         'conv net -- longer chunks are valid and cut the number of '
                         'overlap-add boundaries proportionally.')
    ap.add_argument('--out-prefix', default=None, help='write out0/out1 wavs with this prefix')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}  mode {a.mode}  gain-fix {a.gain_fix}')
    model = build_model(a.ckpt, device)
    audio = load_wav(a.stem)
    print(f'  stem: {a.stem}  {len(audio)/SR:.1f}s')

    chunk = int(round(a.chunk_sec * SR))
    chunk -= chunk % 2
    print(f'  chunk {chunk/SR:.1f}s  hop {chunk/2/SR:.1f}s  '
          f'~{max(len(audio)//(chunk//2)-1,0)} boundaries')
    out0, out1, info = separate_real(model, audio, device, mode=a.mode,
                                     diagnose=a.diagnose, chunk=chunk,
                                     gain_fix=a.gain_fix, lam=a.lam)

    corr = float(np.corrcoef(out0, out1)[0, 1])
    print(f'\ncorr(out0, out1)              : {corr:+.3f}   (lower = more distinct)')
    for name, (lo, hi) in [('Tipton-solo ', TIPTON_SOLO), ('Downing-solo', DOWNING_SOLO)]:
        r0, r1 = rms_win(out0, lo, hi), rms_win(out1, lo, hi)
        print(f'{name} window  out0/out1: {r0:.4f} / {r1:.4f}   ratio {r0/(r1+1e-9):.2f}  (want > 1)')

    if a.diagnose and info:
        print(f"\norientation: {int(info['swap'].sum())}/{len(info['swap'])} chunks swapped")
        g = info['gains']
        print(f"gain correction: out0 {g[:,0].min():.3f}..{g[:,0].max():.3f}   "
              f"out1 {g[:,1].min():.3f}..{g[:,1].max():.3f}")
        for tag in ('before', 'after'):
            d0, d1 = info[tag]
            print(f"  overlap level disagreement {tag:<6}: out0 mean {d0.mean():5.2f}dB "
                  f"p90 {np.percentile(d0,90):5.2f}dB max {d0.max():6.2f}dB  |  "
                  f"out1 mean {d1.mean():5.2f}dB p90 {np.percentile(d1,90):5.2f}dB max {d1.max():6.2f}dB")
            print(f"  {'':>29}  >3dB at out0 {(d0>3).sum()}/{len(d0)}  out1 {(d1>3).sum()}/{len(d1)}")

    if a.out_prefix:
        write_wav(f'{a.out_prefix}_out0_lead.wav', out0)
        write_wav(f'{a.out_prefix}_out1_rhythm.wav', out1)
        print(f'\nwrote {a.out_prefix}_out0_lead.wav / {a.out_prefix}_out1_rhythm.wav')


if __name__ == '__main__':
    main()
