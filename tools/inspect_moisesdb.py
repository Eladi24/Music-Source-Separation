"""
Answers the single gating question for the HTDemucs step-change plan (see
CLAUDE.md, "The step change: what it would actually take"): how many
MoisesDB songs actually have 2+ separate Distorted Electric Guitar sources
(a real isolated lead+rhythm pair), not just a "guitar" stem present.

Reads data.json files directly -- no audio decoding, no moisesdb package
dependency, so it's fast even over the full download.

Usage:
    export MOISESDB_PATH=/path/to/extracted/moisesdb_v0.1
    python tools/inspect_moisesdb.py
"""
import json
import os
import sys
from collections import Counter

TARGET_TYPES = {
    "distorted electric guitar": "distorted",
    "clean electric guitar": "clean",
    "acoustic guitar": "acoustic",
    "lap steel guitar or slide guitar": "lapsteel",
}


def find_data_jsons(root):
    for dirpath, _, filenames in os.walk(root):
        if "data.json" in filenames:
            yield os.path.join(dirpath, "data.json")


def guitar_track_type_counts(data):
    """trackType -> count of individual source files, for the guitar stem only."""
    counts = Counter()
    for stem in data.get("stems", []):
        if stem.get("stemName", "").strip().lower() != "guitar":
            continue
        for track in stem.get("tracks", []):
            tt = track.get("trackType", "").strip().lower()
            counts[tt] += 1
    return counts


def main():
    root = os.environ.get("MOISESDB_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not root or not os.path.isdir(root):
        print("Set MOISESDB_PATH or pass the extracted dataset path as argv[1].")
        sys.exit(1)

    jsons = list(find_data_jsons(root))
    print(f"Found {len(jsons)} data.json files under {root}\n")
    if not jsons:
        print("No data.json found -- check the path points at the extracted "
              "provider/track_id folder structure (moisesdb_v0.1/<provider>/<track_id>/data.json).")
        sys.exit(1)

    multi_distorted = []  # (artist, song, n_distorted)
    has_guitar_stem = 0
    any_guitar_type_multi = []  # >=2 of the SAME guitar sub-type, any sub-type
    unknown_track_types = Counter()

    for jp in jsons:
        try:
            with open(jp) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  WARN: failed to parse {jp}: {e}")
            continue

        counts = guitar_track_type_counts(data)
        if counts:
            has_guitar_stem += 1

        artist = data.get("artist", "?")
        song = data.get("song", "untitled")

        n_distorted = counts.get("distorted electric guitar", 0)
        if n_distorted >= 2:
            multi_distorted.append((artist, song, n_distorted))

        for tt, n in counts.items():
            if tt not in TARGET_TYPES and n > 0:
                unknown_track_types[tt] += 1
            if n >= 2:
                any_guitar_type_multi.append((artist, song, tt, n))

    print(f"Songs with a guitar stem present at all: {has_guitar_stem} / {len(jsons)}")
    print(f"\nSongs with >=2 separate Distorted Electric Guitar sources "
          f"(the thing we actually need): {len(multi_distorted)}")
    for artist, song, n in multi_distorted:
        print(f"  {n}x  {artist} - {song}")

    if any_guitar_type_multi:
        print(f"\nSongs with >=2 sources of ANY single guitar sub-type "
              f"(looser criterion, includes the above): {len(any_guitar_type_multi)}")
        for artist, song, tt, n in any_guitar_type_multi:
            print(f"  {n}x {tt}: {artist} - {song}")

    if unknown_track_types:
        print(f"\nUnrecognized guitar trackType strings seen (schema may differ "
              f"from what track.py's Table 2 names suggest): {dict(unknown_track_types)}")


if __name__ == "__main__":
    main()
