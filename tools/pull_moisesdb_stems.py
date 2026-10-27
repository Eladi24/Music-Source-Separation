"""
Pulls the NON-guitar stems of the 31 MoisesDB lead/rhythm-pair songs for v6
(2026-10-23): needed to build each song's full mix (-> a Demucs-extracted
guitar input, see tools/demucs_guitar_inputs.py) and as other-instrument
bleed for augmentation.

Storage-light by design: each stem file is streamed into memory via HTTP range
reads (remote_zip_metadata.HTTPRangeFile) and never written raw. Stems are
summed per category and saved as 16-bit FLAC, ~16.6GB of raw 24-bit WAV ->
a few GB:

    output/moisesdb_stems/<track_id>/{drums,bass,vocals,other}.flac
    output/moisesdb_stems/<track_id>/data.json
    output/moisesdb_stems/stems_manifest.json   # per song: categories, gains, length

"other" = every stem folder that isn't guitar/drums/bass/vocals (piano, keys,
strings, wind, percussion, ...). Each FLAC is scaled to peak <= 0.99 if needed;
multiply by 1/gain (manifest) to get the true level back. The guitar tracks
stay in output/moisesdb_guitar/ (24-bit, unchanged). Songs already in
output/moisesdb_full/ are read from there instead of the network.

    python tools/pull_moisesdb_stems.py "<presigned URL>"
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from remote_zip_metadata import HTTPRangeFile  # noqa: E402

ROOT_PREFIX = "moisesdb/moisesdb_v0.1/"
SR = 44100
CATS = ("drums", "bass", "vocals")
LOCAL_FULL = "output/moisesdb_full"


def category(stem_folder):
    return stem_folder if stem_folder in CATS else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--pairs", default="output/moisesdb_guitar/manifest.json")
    ap.add_argument("--out", default="output/moisesdb_stems")
    a = ap.parse_args()

    songs = json.load(open(a.pairs))["songs"]
    f = HTTPRangeFile(a.url)
    zf = zipfile.ZipFile(f)
    infos = zf.infolist()
    print(f"archive {f.size/1e9:.1f}GB, {len(infos)} entries; {len(songs)} songs", flush=True)

    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "stems_manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    t0, got = time.time(), 0

    for k, s in enumerate(songs):
        tid = s["track_id"]
        if tid in manifest:
            print(f"[{k+1}/{len(songs)}] {s['artist']} - {s['song']}: done earlier, skipped", flush=True)
            continue
        d = os.path.join(a.out, tid)
        os.makedirs(d, exist_ok=True)
        local = os.path.join(LOCAL_FULL, tid)
        prefix = f"{ROOT_PREFIX}{tid}/"
        members = [i for i in infos if i.filename.startswith(prefix) and i.filename.endswith(".wav")
                   and i.filename[len(prefix):].split("/")[0] != "guitar"]
        if os.path.exists(os.path.join(local, "data.json")):
            data_json = open(os.path.join(local, "data.json"), "rb").read()
        else:
            data_json = zf.read(prefix + "data.json")
        open(os.path.join(d, "data.json"), "wb").write(data_json)

        sums, folders = {}, {}
        for m in members:
            rel = m.filename[len(prefix):]
            folder = rel.split("/")[0]
            lp = os.path.join(local, rel)
            if os.path.exists(lp):
                x, sr = sf.read(lp, dtype="float32", always_2d=True)
            else:
                x, sr = sf.read(io.BytesIO(zf.read(m.filename)), dtype="float32", always_2d=True)
                got += m.file_size
            assert sr == SR, (m.filename, sr)
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            c = category(folder)
            folders.setdefault(c, set()).add(folder)
            if c in sums:
                n = max(len(sums[c]), len(x))
                acc = np.zeros((n, 2), np.float32)
                acc[:len(sums[c])] += sums[c]
                acc[:len(x)] += x
                sums[c] = acc
            else:
                sums[c] = x
            del x

        entry = {"artist": s["artist"], "song": s["song"], "cats": {}}
        for c, x in sums.items():
            peak = float(np.abs(x).max())
            gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
            sf.write(os.path.join(d, f"{c}.flac"), x * gain, SR, subtype="PCM_16")
            entry["cats"][c] = {"gain": gain, "frames": len(x), "folders": sorted(folders[c])}
        manifest[tid] = entry
        json.dump(manifest, open(man_path, "w"), indent=1)
        mb = sum(os.path.getsize(os.path.join(d, fn)) for fn in os.listdir(d)) / 1e6
        el = time.time() - t0
        print(f"[{k+1}/{len(songs)}] {s['artist']} - {s['song']}: {len(members)} stem files -> "
              f"{sorted(sums)} ({mb:.0f}MB on disk); network {got/1e9:.2f}GB, "
              f"{got/1e6/max(el,1):.1f}MB/s", flush=True)
        del sums

    print("done", flush=True)


if __name__ == "__main__":
    main()
