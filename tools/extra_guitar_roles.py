"""
v6 (2026-10-23): role of the EXTRA guitar tracks in the 31 MoisesDB pair songs
(everything in the guitar folder except the labelled lead and rhythm track:
clean electric, acoustic, a third/fourth distorted part).

A Demucs-extracted guitar input contains every guitar in the song, so each
extra needs a target. Rule: an extra whose mean P(lead) over its playing 2s
windows is < 0.3 is accompaniment -> added to the RHYTHM target ("rhythm" =
every guitar except the lead). An extra rated >= 0.3 is "block": Demucs-input
crops avoid every window where it plays (no target is guessed for it). The
role classifier rates tracks heard alone, which can misjudge role in context
(blind tests, CLAUDE.md 2026-10-19), hence blocking rather than relabelling.

Writes tools/extra_guitar_roles.json: {track_id: {guitar_id: {role, type, active, p_lead, use}}}
with use in {"lead", "rhythm", "rhythm_extra", "block"}.

    python tools/extra_guitar_roles.py
"""
import json, sys, os, numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, "tools")
from train_role_classifier import RoleNet, SR, SEG
net = RoleNet(); net.load_state_dict(torch.load("output/role_classifier.pt", map_location="cpu")); net.eval()
man = json.load(open("output/moisesdb_guitar/manifest.json"))["songs"]
lab = {r["track_id"]: r for r in json.load(open("output/moisesdb_guitar/role_labels.json"))}
out = {}
for s in man:
    r = lab[s["track_id"]]
    rows = []
    for t in s["guitar_tracks"]:
        d, sr = sf.read(f"output/moisesdb_guitar/{s['track_id']}/guitar/{t['id']}.wav", dtype="float32", always_2d=True)
        mono = d.mean(1); w = 2*sr; n = len(mono)//w
        rms = np.array([np.sqrt(np.mean(mono[i*w:(i+1)*w]**2)) for i in range(n)])
        x = torchaudio.functional.resample(torch.from_numpy(mono), sr, SR)
        segs = torch.stack([x[i*SEG:(i+1)*SEG] for i in range(min(n, x.shape[0]//SEG))])
        pk = segs.abs().amax(1, keepdim=True).clamp(min=1e-6)
        with torch.no_grad(): p = torch.softmax(net(segs/pk*0.8), 1)[:,1].numpy()
        act = rms[:len(p)] > 0.01
        role = "LEAD" if t["id"]==r["lead_id"] else "RHY" if t["id"]==r["rhythm_id"] else "extra"
        rows.append((role, t["trackType"][:12], act.mean(), p[act].mean() if act.any() else float("nan"), (p[act]>0.8).mean() if act.any() else 0))
        out.setdefault(s["track_id"], {})[t["id"]] = {"role": role, "type": t["trackType"], "active": float(act.mean()), "p_lead": float(p[act].mean()) if act.any() else None}
    print(f"{s['artist'][:12]:12s} {s['song'][:20]:20s} {'usable' if r['usable'] else '      '} " + " | ".join(f"{a} {b} act{c:.2f} P{d:.2f} hi{e:.2f}" for a,b,c,d,e in rows), flush=True)
for tid, tr in out.items():
    for g in tr.values():
        g["use"] = ("lead" if g["role"] == "LEAD" else "rhythm" if g["role"] == "RHY"
                    else "rhythm_extra" if (g["p_lead"] is None or g["p_lead"] < 0.3) else "block")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "extra_guitar_roles.json"), "w"), indent=1)
