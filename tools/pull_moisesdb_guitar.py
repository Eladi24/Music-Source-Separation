"""
Pulls just the guitar-stem audio for the MoisesDB songs that have >=2
separate Distorted Electric Guitar sources (real isolated lead+rhythm
pairs from real multitrack recordings) -- see CLAUDE.md, "The step
change" and "MoisesDB gating research". Uses HTTP range reads against
the zip's central directory (via remote_zip_metadata.HTTPRangeFile) so
only the ~6.07GB of actually-needed audio is transferred, not the full
82.7GB archive.

Usage:
    python tools/pull_moisesdb_guitar.py "<presigned URL>" --out output/moisesdb_guitar
    # 2026-10-19: the 82 songs with exactly ONE distorted guitar (more lead /
    # rhythm material for cross-song remixing):
    python tools/pull_moisesdb_guitar.py "<presigned URL>" --n-distorted 1 --out output/moisesdb_guitar_single
"""
import argparse
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
from remote_zip_metadata import HTTPRangeFile

ROOT_PREFIX = "moisesdb/moisesdb_v0.1/"


def find_qualifying_tracks(zf, n_distorted="2+"):
    """Returns list of dicts: {track_id, artist, song, genre, tracks: [...]}"""
    names = zf.namelist()
    data_json_names = [n for n in names if n.endswith("/data.json") and n.startswith(ROOT_PREFIX)]
    qualifying = []
    for n in data_json_names:
        track_id = n[len(ROOT_PREFIX):].split("/")[0]
        data = json.loads(zf.read(n))
        guitar_tracks = []
        n_dist = 0
        for stem in data.get("stems", []):
            if stem.get("stemName", "").strip().lower() != "guitar":
                continue
            for t in stem.get("tracks", []):
                guitar_tracks.append(t)
                if t.get("trackType", "").strip().lower() == "distorted_electric_guitar":
                    n_dist += 1
        if (n_dist >= 2) if n_distorted == "2+" else (n_dist == int(n_distorted)):
            qualifying.append({
                "track_id": track_id,
                "artist": data.get("artist"),
                "song": data.get("song"),
                "genre": data.get("genre"),
                "guitar_tracks": guitar_tracks,
                "n_distorted": n_dist,
            })
    return qualifying


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="output/moisesdb_guitar")
    ap.add_argument("--n-distorted", default="2+", help='"2+" (lead/rhythm pairs) or an exact count, e.g. "1"')
    args = ap.parse_args()

    f = HTTPRangeFile(args.url)
    print(f"Remote object size: {f.size/1e9:.1f} GB")
    zf = zipfile.ZipFile(f)

    qualifying = find_qualifying_tracks(zf, args.n_distorted)
    print(f"{len(qualifying)} qualifying songs ({args.n_distorted} distorted electric guitar sources)")

    os.makedirs(args.out, exist_ok=True)
    manifest = {"songs": qualifying}
    with open(os.path.join(args.out, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    infos = {info.filename: info for info in zf.infolist()}
    to_fetch = []
    for song in qualifying:
        prefix = f"{ROOT_PREFIX}{song['track_id']}/guitar/"
        for name, info in infos.items():
            if name.startswith(prefix) and not name.endswith("/"):
                to_fetch.append((song, name, info))

    total_bytes = sum(info.file_size for _, _, info in to_fetch)
    print(f"{len(to_fetch)} audio files, {total_bytes/1e9:.2f} GB to transfer")

    done_bytes = 0
    skipped = 0
    for i, (song, name, info) in enumerate(to_fetch):
        rel = name[len(ROOT_PREFIX):]  # "<track_id>/guitar/<id>.wav"
        dest = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            done_bytes += info.file_size
            skipped += 1
            continue
        data = zf.read(name)
        with open(dest, "wb") as out:
            out.write(data)
        done_bytes += info.file_size
        print(f"  [{i+1}/{len(to_fetch)}] {song['artist']} - {song['song']}: "
              f"{rel} ({info.file_size/1e6:.1f}MB) -- {done_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB")

    print(f"\nDone. {skipped} files already present and skipped. "
          f"Total on disk for this slice: {done_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
