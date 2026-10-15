"""
Extracts short excerpts from full MoisesDB tracks for the standing
listening-approval gate (CLAUDE.md "Conventions" -- before training on
any newly-acquired audio data). Picks a window centered on peak energy
for the lead track (likeliest to catch an actual solo) and the song's
midpoint for the rhythm track.
"""
import os
import sys

import numpy as np
import soundfile as sf

SONGS = [
    ("holymagick_crystalobservatory", "d4df499c-e394-4753-b459-e167e6a58bad",
     "13a8dfe2-8580-4df9-93e8-4112bd81752d", "c4dacf6e-e04c-4706-8623-a4d91a7a83ee"),
    ("prorata_fightthefeeling", "9653a690-c28c-4e8f-962e-ff7ed18b8ee9",
     "1d835071-b5bc-41e4-ba73-c5f737441b2c", "2cec450a-1dca-409f-8dc9-efca2a289bce"),
]

DATA_DIR = "output/moisesdb_guitar"
OUT_DIR = "output/stage3_local_test"
EXCERPT_SEC = 45


def energy_peak_window(path, sec):
    data, sr = sf.read(path, dtype="float32")
    mono = data.mean(axis=1) if data.ndim == 2 else data
    win = sec * sr
    hop = sr  # 1s hop
    best_start, best_rms = 0, -1
    for start in range(0, max(1, len(mono) - win), hop):
        rms = np.sqrt(np.mean(mono[start:start + win] ** 2))
        if rms > best_rms:
            best_rms, best_start = rms, start
    return best_start, sr, data


def midpoint_window(path, sec):
    data, sr = sf.read(path, dtype="float32")
    win = sec * sr
    start = max(0, len(data) // 2 - win // 2)
    return start, sr, data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, track_id, lead_id, rhythm_id in SONGS:
        lead_path = os.path.join(DATA_DIR, track_id, "guitar", f"{lead_id}.wav")
        rhythm_path = os.path.join(DATA_DIR, track_id, "guitar", f"{rhythm_id}.wav")

        start, sr, data = energy_peak_window(lead_path, EXCERPT_SEC)
        excerpt = data[start:start + EXCERPT_SEC * sr]
        out_path = os.path.join(OUT_DIR, f"moisesdb_{name}_LEAD.wav")
        sf.write(out_path, excerpt, sr)
        print(f"{out_path}  ({start/sr:.0f}s-{(start+EXCERPT_SEC*sr)/sr:.0f}s of {lead_path})")

        start, sr, data = midpoint_window(rhythm_path, EXCERPT_SEC)
        excerpt = data[start:start + EXCERPT_SEC * sr]
        out_path = os.path.join(OUT_DIR, f"moisesdb_{name}_RHYTHM.wav")
        sf.write(out_path, excerpt, sr)
        print(f"{out_path}  ({start/sr:.0f}s-{(start+EXCERPT_SEC*sr)/sr:.0f}s of {rhythm_path})")


if __name__ == "__main__":
    main()
