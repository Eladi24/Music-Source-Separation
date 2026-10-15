"""
Applies the GuitarSet-trained role classifier (tools/train_role_classifier.py)
to each MoisesDB distorted-electric-guitar track to auto-label it lead or
rhythm -- MoisesDB has no native role label the way GuitarSet's
_solo_mix/_comp_mix filenames did (see CLAUDE.md, "The step change" /
role-labeling decision, 2026-10-15).

For each qualifying song (>=2 distorted_electric_guitar sources, from
manifest.json), classifies every 2s window of every distorted-guitar track
and reports the per-track lead-fraction (how much of the track reads as
"lead" to the classifier) -- real playing isn't one role for the whole
clip, so this is a distribution, not a single label. A song is usable as
a training pair only if its tracks separate into a clear lead-leaning one
and a clear rhythm-leaning one.

Usage:
    source ~/pytorch_env/bin/activate
    python tools/label_moisesdb_guitar.py --data output/moisesdb_guitar \
        --classifier output/role_classifier.pt
"""
import argparse
import json
import os
import sys

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(__file__))
from train_role_classifier import RoleNet, SR, SEG


def classify_track(model, path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    n = wav.shape[0]
    if n < SEG:
        return None
    n_segs = n // SEG
    segs = torch.stack([wav[i * SEG:(i + 1) * SEG] for i in range(n_segs)])
    peaks = segs.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    active = (peaks.squeeze(1) > 0.01)  # skip near-silent windows
    segs = segs / peaks * 0.8
    with torch.no_grad():
        probs = torch.softmax(model(segs), dim=1)[:, 1]  # P(lead)
    probs = probs[active]
    if len(probs) == 0:
        return None
    return {
        "n_windows": int(len(probs)),
        "lead_fraction": float((probs > 0.5).float().mean()),
        "mean_p_lead": float(probs.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/moisesdb_guitar")
    ap.add_argument("--classifier", default="output/role_classifier.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = RoleNet()
    model.load_state_dict(torch.load(args.classifier, map_location="cpu"))
    model.eval()

    with open(os.path.join(args.data, "manifest.json")) as f:
        manifest = json.load(f)

    results = []
    for song in manifest["songs"]:
        track_id = song["track_id"]
        dist_tracks = [t for t in song["guitar_tracks"]
                       if t.get("trackType", "").strip().lower() == "distorted_electric_guitar"]
        track_results = []
        for t in dist_tracks:
            path = os.path.join(args.data, track_id, "guitar", f"{t['id']}.wav")
            if not os.path.exists(path):
                continue
            stats = classify_track(model, path)
            if stats:
                track_results.append({**stats, "id": t["id"], "has_bleed": t.get("has_bleed")})

        if len(track_results) < 2:
            continue
        track_results.sort(key=lambda r: r["mean_p_lead"], reverse=True)
        lead, rhythm = track_results[0], track_results[-1]
        gap = lead["mean_p_lead"] - rhythm["mean_p_lead"]
        results.append({
            "artist": song["artist"], "song": song["song"], "genre": song["genre"],
            "track_id": track_id, "n_tracks": len(track_results),
            "lead_id": lead["id"], "lead_p": lead["mean_p_lead"],
            "rhythm_id": rhythm["id"], "rhythm_p": rhythm["mean_p_lead"],
            "gap": gap, "usable": gap > 0.3,
        })

    results.sort(key=lambda r: r["gap"], reverse=True)
    print(f"{len(results)} songs with >=2 classifiable distorted-guitar tracks\n")
    for r in results:
        flag = "OK " if r["usable"] else "low"
        print(f"  [{flag}] gap={r['gap']:.2f}  {r['artist']} - {r['song']}  "
              f"({r['n_tracks']} tracks, lead_p={r['lead_p']:.2f} rhythm_p={r['rhythm_p']:.2f})")

    usable = [r for r in results if r["usable"]]
    print(f"\n{len(usable)}/{len(results)} songs have a clear lead/rhythm split (gap > 0.3)")

    out_path = args.out or os.path.join(args.data, "role_labels.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
