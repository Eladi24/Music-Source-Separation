"""Verify Guitar-TECHS against the measured requirement that motivated it.

The bottleneck measured 2026-10-14: GuitarSet training targets hold ~1.5-2.0%
of their energy at 2-4kHz, where real electric guitar in a real mix holds
10-25% (Sentinel 24.5%, Your Light 10.4%) -- a 12.5x mismatch. The model
therefore never learned how to allocate real treble content and starves the
rhythm output of it on real audio (out1 5.1% vs the input's 24.5%).

So the question this dataset has to answer is specific and quantitative:
does its audio -- particularly the real amp-microphone perspective -- have a
band-energy distribution resembling real electric guitar rather than
GuitarSet's? Everything else (license, format, labels) is secondary.
"""
import sys, zipfile, wave, io, collections
import numpy as np

BANDS = ((20, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6500))
# measured references, for comparison
REF = {
    'GuitarSet solo target': [10.15, 65.71, 17.82, 4.67, 1.46, 0.19],
    'GuitarSet comp target': [29.36, 41.98, 20.82, 5.60, 1.96, 0.29],
    'REAL Sentinel stem':    [24.76, 20.52, 12.50, 17.23, 24.51, 0.48],
    'REAL Your Light stem':  [17.55, 33.63, 29.34,  7.58, 10.41, 1.48],
}


def band_shares(x, sr, nfft=8192):
    """Share of total energy per band, %. Computed at the file's own sample
    rate but only over bands below 6500Hz, so it is directly comparable to
    the 16kHz-pipeline numbers above."""
    n = (len(x) // nfft) * nfft
    if n == 0:
        return None
    X = np.abs(np.fft.rfft(x[:n].reshape(-1, nfft) * np.hanning(nfft), axis=1)).mean(0)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    p = X ** 2
    sel = f < 6500
    tot = p[sel].sum()
    return [100 * p[(f >= lo) & (f < hi)].sum() / (tot + 1e-20) for lo, hi in BANDS]


def read_wav_bytes(b):
    with wave.open(io.BytesIO(b), 'rb') as w:
        sr, n, sw, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float64)
    if sw == 1:
        x = x - 128.0
    x /= float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr, sw, ch, n


def main(zip_path, max_per_group=4):
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.lower().endswith(('.wav', '.mid', '.midi'))]
    wavs = [n for n in names if n.lower().endswith('.wav')]
    mids = [n for n in names if n.lower().endswith(('.mid', '.midi'))]
    print(f'{zip_path}: {len(z.namelist())} entries, {len(wavs)} wav, {len(mids)} midi\n')
    print('--- directory layout (first 25 entries) ---')
    for n in z.namelist()[:25]:
        print('  ', n)
    print()

    # group by the perspective token in the path/filename
    groups = collections.defaultdict(list)
    for n in wavs:
        low = n.lower()
        key = ('amp_mic' if 'amp' in low or 'mic' in low else
               'DI' if 'di' in low.replace('midi', '') else
               'egocentric' if 'ego' in low else
               'exocentric' if 'exo' in low else 'other')
        groups[key].append(n)
    print('--- wav files per perspective ---')
    for k, v in sorted(groups.items()):
        print(f'  {k:<12} {len(v):>4} files   e.g. {v[0]}')
    print()

    print('--- format + band-energy distribution (share of <6500Hz energy, %) ---')
    hdr = f'  {"source":<34}' + ''.join(f'{lo//1000 if lo>=1000 else lo}-{hi//1000 if hi>=1000 else hi//1000}k'.rjust(9) for lo, hi in BANDS)
    for k, v in sorted(groups.items()):
        for n in v[:max_per_group]:
            try:
                x, sr, sw, ch, nf = read_wav_bytes(z.read(n))
            except Exception as e:
                print(f'  {k}: {n} -> unreadable ({e})')
                continue
            sh = band_shares(x, sr)
            if sh is None:
                continue
            if 'printed_hdr' not in dir(main):
                print(hdr); main.printed_hdr = True
            label = f'{k}/{n.split("/")[-1][:24]}'
            print(f'  {label:<34}' + ''.join(f'{s:8.2f}%' for s in sh)
                  + f'   [{sr}Hz {sw*8}bit {ch}ch {nf/sr:.1f}s]')
    print()
    print('--- measured references ---')
    print(hdr)
    for k, v in REF.items():
        print(f'  {k:<34}' + ''.join(f'{s:8.2f}%' for s in v))
    print('\nThe question: does the amp_mic row look like the REAL rows (10-25% at 2-4k)')
    print('or like the GuitarSet rows (1.5-2%)? That is what this dataset is for.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else
         '/home/eladubuntu/Music-Source-Separation/output/guitartechs/P3_music.zip')
