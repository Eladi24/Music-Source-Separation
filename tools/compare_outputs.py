"""Compare lead/rhythm separation outputs by spectral balance and per-band
lead dominance. The two things broadband metrics cannot tell apart.

Why this exists: `corr` and the broadband solo-window RMS ratio are both
dominated by low frequencies, so they are blind to the defect that actually
drew the user's complaint ("treble all the way down" on the rhythm track)
AND they confound two opposite outcomes. When the rhythm output's missing
2-4kHz energy is restored -- the goal -- its level in that band rises, which
MECHANICALLY lowers the out0/out1 ratio exactly as more solo bleed would.
A success and a regression have the same broadband signature.

Splitting by band separates them:
  - band SHARES vs the input stem say whether each output's spectral balance
    is right (is the rhythm track still treble-starved?)
  - per-band DOMINANCE in the solo windows says whether the lead is still
    winning where solos live (has bleed got worse?)

Validated against the 2026-10-14 lead4 -> lead6_epoch5 comparison, where it
correctly predicted an audible change (2-4kHz solo dominance 5.47 -> 2.08)
that corr (0.204 -> 0.182, slightly "better") completely missed. First
metric in this project to predict a listening result rather than trail it.
"""
import argparse, sys
import numpy as np

sys.path.insert(0, '/home/eladubuntu/Music-Source-Separation')
from tools.realsong_test import load_wav, SR, TIPTON_SOLO, DOWNING_SOLO

BANDS = ((20, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6500))
LABELS = ['20-250', '250-500', '500-1k', '1-2k', '2-4k', '4-6.5k']


def band_shares(x, sr=SR, nfft=8192, ceil_hz=6500):
    """Share of sub-ceil energy per band, %. Normalised per signal, so it is
    invariant to overall level (which the model does not control anyway --
    see the beta=-0.5 finding in CLAUDE.md)."""
    n = (len(x) // nfft) * nfft
    if n == 0:
        return None
    X = np.abs(np.fft.rfft(x[:n].reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(0)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    p = X ** 2
    tot = p[f < ceil_hz].sum()
    return np.array([100 * p[(f >= lo) & (f < hi)].sum() / (tot + 1e-20) for lo, hi in BANDS])


def band_rms(x, lo, hi, a, b, sr=SR):
    """Per-bin RMS in a band over a time window.

    MUST normalise by the bin count: windows of different lengths have
    different numbers of FFT bins in the same Hz range, so returning total
    band energy makes any comparison ACROSS window lengths wrong. The
    dominance table compares two signals over the SAME window (the bin count
    cancels), but solo_swell() compares a 4-5s solo window against 10s
    rhythm windows, where it does not. Caught 2026-10-14: the unnormalised
    version reported lead4 out1 swell as 0.77/0.54 instead of 1.09/0.86,
    which would have corrupted the checkpoint-selection metric."""
    seg = x[int(a * sr):int(b * sr)]
    if len(seg) < 16:
        return 0.0
    X = np.fft.rfft(seg * np.hanning(len(seg)))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    m = (f >= lo) & (f < hi)
    if m.sum() == 0:
        return 0.0
    return float(np.sqrt((np.abs(X[m]) ** 2).sum() / m.sum()))


RHYTHM_WINDOWS = [(60.0, 70.0), (100.0, 110.0), (185.0, 195.0)]   # no-solo passages


def solo_swell(x, lo, hi, solo_win, sr=SR):
    """How much louder a signal gets during a solo than during its OWN
    rhythm-only passages, in one band.

    This is the metric that finally separates the two outcomes the dominance
    ratio confounds. Restoring out1's missing treble raises its level
    everywhere, which lowers dominance without any extra bleed; solo content
    leaking in raises it ONLY during solos. Normalising each signal against
    itself makes this immune to that confound.

    out1 swell ~1.0 => the rhythm output does not react to solos (clean).
    Measured 2026-10-14: lead4 1.09/0.86 (clean but treble-starved at 4.11%
    share), lead6_ep14 1.23/1.57 (treble restored to 9.87% but solo leaking
    in). The input mix itself swells 2.15/2.55, so out0 should be up there
    and out1 should not.
    """
    ref = float(np.mean([band_rms(x, lo, hi, a, b) for a, b in RHYTHM_WINDOWS]))
    return band_rms(x, lo, hi, *solo_win) / (ref + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prefixes', nargs='+',
                    help='output prefixes, e.g. output/stage3_local_test/sentinel_lead4_2s')
    ap.add_argument('--stem', default='output/04 - The Sentinel/guitar.wav',
                    help='the input stem, used as the spectral-balance reference')
    ap.add_argument('--suffix0', default='_out0_lead.wav')
    ap.add_argument('--suffix1', default='_out1_rhythm.wav')
    a = ap.parse_args()

    inp = load_wav(a.stem)
    print('Share of sub-6500Hz energy (%) -- compare each output against the INPUT row\n')
    print(f'  {"signal":<34}' + ''.join(l.rjust(9) for l in LABELS))
    print(f'  {"INPUT stem (the target)":<34}' + ''.join(f'{v:8.2f}%' for v in band_shares(inp)))
    loaded = {}
    for p in a.prefixes:
        name = p.split('/')[-1]
        try:
            o0, o1 = load_wav(p + a.suffix0), load_wav(p + a.suffix1)
        except Exception as e:
            print(f'  {name}: could not load ({e})')
            continue
        loaded[name] = (o0, o1)
        print(f'  {name + "  out0 lead":<34}' + ''.join(f'{v:8.2f}%' for v in band_shares(o0)))
        print(f'  {name + "  out1 RHYTHM":<34}' + ''.join(f'{v:8.2f}%' for v in band_shares(o1)))

    print('\n\nPer-band lead dominance in the solo windows (out0/out1; >1 = lead wins)')
    print('The 2-4k column is the one that tracks audible bleed during solos.\n')
    print(f'  {"signal / window":<34}' + ''.join(l.rjust(9) for l in LABELS[:5]))
    for name, (o0, o1) in loaded.items():
        for wn, (lo_t, hi_t) in (('Tipton', TIPTON_SOLO), ('Downing', DOWNING_SOLO)):
            r = [band_rms(o0, lo, hi, lo_t, hi_t) / (band_rms(o1, lo, hi, lo_t, hi_t) + 1e-12)
                 for lo, hi in BANDS[:5]]
            print(f'  {name + " / " + wn:<34}' + ''.join(f'{v:9.2f}' for v in r))

    print('\n\nSolo swell at 2-4kHz (level during solos / level during rhythm-only passages,')
    print('each signal vs ITSELF).  out1 ~1.0 = rhythm output stays clean; higher = solo leaking in.\n')
    print(f'  {"signal":<34}{"out1 Tip":>10}{"out1 KK":>10}{"out0 Tip":>10}{"out0 KK":>10}')
    print(f'  {"INPUT stem (the mix)":<34}'
          + ''.join(f'{solo_swell(inp,2000,4000,w):10.2f}' for w in (TIPTON_SOLO, DOWNING_SOLO))
          + f'{"":>20}')
    for name,(o0,o1) in loaded.items():
        vals=[solo_swell(o1,2000,4000,TIPTON_SOLO), solo_swell(o1,2000,4000,DOWNING_SOLO),
              solo_swell(o0,2000,4000,TIPTON_SOLO), solo_swell(o0,2000,4000,DOWNING_SOLO)]
        print(f'  {name:<34}'+''.join(f'{v:10.2f}' for v in vals))

    print('\n\nSummary per output set:')
    ref = band_shares(inp)
    for name, (o0, o1) in loaded.items():
        s0, s1 = band_shares(o0), band_shares(o1)
        print(f'  {name}:')
        print(f'    out1 rhythm 2-4kHz share {s1[4]:5.2f}%  (input {ref[4]:5.2f}%  -> '
              f'{100*s1[4]/ref[4]:4.0f}% of target)')
        print(f'    out0 lead   2-4kHz share {s0[4]:5.2f}%  (input {ref[4]:5.2f}%  -> '
              f'{100*s0[4]/ref[4]:4.0f}% of target)')
        print(f'    imbalance out0/out1 at 2-4kHz: {s0[4]/(s1[4]+1e-9):.2f}x  (1.0 = balanced)')
        sw=[solo_swell(o1,2000,4000,w) for w in (TIPTON_SOLO,DOWNING_SOLO)]
        print(f'    out1 solo swell {sw[0]:.2f} / {sw[1]:.2f}   (want ~1.0 -- the GOAL is '
              f'full treble share AND swell ~1.0)')


if __name__ == '__main__':
    main()
