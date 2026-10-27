"""
v6 (2026-10-23): how loud is the non-guitar content in htdemucs_6s's guitar
stem? Sets the notebook's BLEED_DB range for remix bleed augmentation, so the
augmentation matches what the real-song input actually carries.

Per 2s window where the true guitar sum g plays, on TRAINING songs only (the
test, validation and two-rhythm eval songs are excluded): least-squares fit of
the Demucs guitar dg on [g, drums, bass, vocals, other] (the song's true
stems). The stems' part = bleed; what's left = Demucs's own artifacts on the
guitar. Both in dB vs the fitted guitar part. BLEED_DB = the bleed part's
10th..90th percentile. (A first version used the whole leftover, -9..0dB: that
counts the artifacts as bleed, which full drums/vocals stems at that level
would badly overstate.) A linear fit can't follow Demucs's frequency-dependent
leakage, so the bleed figure is a lower bound.

    python tools/v6_bleed_db.py      # -> tools/v6_bleed_db.json
"""
import glob
import json
import os

import numpy as np
import soundfile as sf

SR = 44100
HELD_OUT = {"Electric", "False Black Bear", "Darkest Heart", "Never Say Never", "Dear You", "Snake Eyes"}
STEMS = "output/moisesdb_stems"
GUITAR = "output/moisesdb_guitar"


def load(path, gain=1.0):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR
    return x.T / gain


def add(acc, x):
    if acc is None:
        return x.copy()
    n = max(acc.shape[1], x.shape[1])
    out = np.zeros((2, n), np.float32)
    out[:, :acc.shape[1]] += acc
    out[:, :x.shape[1]] += x
    return out


def main():
    man = json.load(open(os.path.join(STEMS, "stems_manifest.json")))
    vals, per_song = [], {}
    for tid, e in man.items():
        if e["song"] in HELD_OUT or "demucs_guitar" not in e:
            continue
        g = None
        for p in glob.glob(os.path.join(GUITAR, tid, "guitar", "*.wav")):
            g = add(g, load(p))
        dg = load(os.path.join(STEMS, tid, "demucs_guitar.flac"), e["demucs_guitar"]["gain"])
        cats = [load(os.path.join(STEMS, tid, f"{c}.flac"), i["gain"]) for c, i in e["cats"].items()]
        n = min([g.shape[1], dg.shape[1]] + [x.shape[1] for x in cats])
        w = 2 * SR
        song = []
        for i in range(0, n - w, w):
            gw, dw = g[:, i:i + w].reshape(-1), dg[:, i:i + w].reshape(-1)
            if np.sqrt(np.mean(gw ** 2)) < 0.01:
                continue
            X = np.stack([gw] + [x[:, i:i + w].reshape(-1) for x in cats], 1)
            coef = np.linalg.lstsq(X, dw, rcond=None)[0]
            gp, bp = coef[0] * gw, X[:, 1:] @ coef[1:]
            rest = dw - gp - bp
            song.append((10 * np.log10(bp @ bp / (gp @ gp) + 1e-12), 10 * np.log10(rest @ rest / (gp @ gp) + 1e-12)))
        sb = [x for x, _ in song]
        per_song[e["song"]] = round(float(np.median(sb)), 1) if song else None
        vals += song
        print(f"{e['artist'][:16]:16s} {e['song'][:26]:26s} windows {len(song):3d}  bleed {per_song[e['song']]}dB  "
              f"artifacts {np.median([y for _, y in song]):.1f}dB", flush=True)
    art = [y for _, y in vals]
    vals = [x for x, _ in vals]
    lo, med, hi = np.percentile(vals, [10, 50, 90])
    print(f"\nbleed, all training windows: n={len(vals)}  p10 {lo:.1f}  median {med:.1f}  p90 {hi:.1f} dB"
          f"  | artifacts median {np.median(art):.1f} dB")
    out = {"bleed_db": [round(float(lo), 1), round(float(hi), 1)], "median": round(float(med), 1),
           "artifacts_median": round(float(np.median(art)), 1),
           "n_windows": len(vals), "per_song_bleed_median": per_song}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "v6_bleed_db.json"), "w"), indent=1)
    print("wrote tools/v6_bleed_db.json")


if __name__ == "__main__":
    main()
