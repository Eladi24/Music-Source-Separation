"""Is BS-Roformer SW a better guitar extractor than htdemucs_6s on real songs?

The oracle check (tools/oracle_demucs_input.py) showed the real-song gap lives
in the guitar-extraction step: the htdemucs_6s guitar stem loses 4.6 / 6.8dB of
lead / rhythm even under a perfect mask, and our lead/rhythm model already loses
no more than that. This scores a second extractor on the SAME windows of the
same rebuilt full mixes (all guitar tracks + drums/bass/vocals/other stems, as
tools/demucs_guitar_inputs.py built them).

Model: BS-Roformer SW 6-stem, ONNX export (elicwhite/bs-roformer-sw-6stem-onnx,
fp16 weights; the weights themselves have NO stated licence and unknown
provenance -- research measurement only). STFT 2048/512 outside the graph,
fixed 4s chunks; validated against the repo's reference.npz (77-129dB SNR).
Too slow on this CPU for whole songs (~14s per 4s chunk), so each 4s eval
window is extracted from 2s-hop, Hann-windowed overlapping chunks covering
[w-2s, w+6s] (every sample gets two chunks of context); chunks are cached so
neighbouring windows share them.

Per window: faith (input vs true guitar sum) and oracle-mask lead / rhythm
SI-SDR (od_L / od_R) for both inputs. Results -> tools/roformer_vs_demucs_<shard>.json.

    python tools/roformer_vs_demucs.py --shard 0/2 & python tools/roformer_vs_demucs.py --shard 1/2
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
from oracle_demucs_input import ACTIVE, SR, WIN, load, oracle, rms2s, sisdr  # noqa: E402

CH = 4 * SR          # model chunk (fixed by the ONNX trace)
HOP = 2 * SR
GUITAR = 4           # stem order: bass drums other vocals guitar piano
_w = torch.hann_window(2048)
_syn = np.hanning(CH + 1)[:CH].astype(np.float32)   # periodic Hann: 50% overlap sums to 1


def add(acc, x):
    if acc is None:
        return x.copy()
    n = max(acc.shape[1], x.shape[1])
    out = np.zeros((2, n), np.float32)
    out[:, :acc.shape[1]] += acc
    out[:, :x.shape[1]] += x
    return out


class Roformer:
    def __init__(self, path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.enable_cpu_mem_arena = False   # the arena pushed peak RSS 3.8 -> 6.3GB and got OOM-killed (7.6GB machine)
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def guitar(self, chunk):   # (2, CH) -> (2, CH)
        S = torch.stft(torch.from_numpy(chunk), 2048, 512, 2048, window=_w, center=True, return_complex=True)
        re, im = self.sess.run(None, {"spec_real": S.real[None].numpy(), "spec_imag": S.imag[None].numpy()})
        Y = torch.complex(torch.from_numpy(re[0, GUITAR]), torch.from_numpy(im[0, GUITAR]))
        return torch.istft(Y, 2048, 512, 2048, window=_w, center=True, length=CH).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guitar-dir", default="output/moisesdb_guitar")
    ap.add_argument("--stems-dir", default="output/moisesdb_stems")
    ap.add_argument("--extras", default="tools/extra_guitar_roles.json")
    ap.add_argument("--onnx", default="output/models/bs_roformer_sw/bs_roformer_sw_6stem_fp16.onnx")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    si, sn = map(int, args.shard.split("/"))

    labels = [r for r in json.load(open(os.path.join(args.guitar_dir, "role_labels.json"))) if r["usable"]]
    stems = json.load(open(os.path.join(args.stems_dir, "stems_manifest.json")))
    extras = json.load(open(args.extras))
    model = Roformer(args.onnx, args.threads)
    out_path = f"tools/roformer_vs_demucs_{si}.json"
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t0 = time.time()
    for k, r in enumerate(labels):
        tid = r["track_id"]
        if k % sn != si or tid in results:
            continue
        gdir = os.path.join(args.guitar_dir, tid, "guitar")
        L = load(os.path.join(gdir, r["lead_id"] + ".wav"))
        R = load(os.path.join(gdir, r["rhythm_id"] + ".wav"))
        D = load(os.path.join(args.stems_dir, tid, "demucs_guitar.flac"), stems[tid]["demucs_guitar"]["gain"])
        n = min(L.shape[1], R.shape[1], D.shape[1])
        L, R, D = L[:, :n], R[:, :n], D[:, :n]
        R_ext, blocked = R.copy(), set()
        for gid, e in extras.get(tid, {}).items():
            if e["use"] == "rhythm_extra":
                R_ext += load(os.path.join(gdir, gid + ".wav"))[:, :n]
            elif e["use"] == "block":
                act = np.where(rms2s(load(os.path.join(gdir, gid + ".wav"))[:, :n]) > ACTIVE)[0]
                blocked |= {int(i) for i in act}
        lr, rr = rms2s(L), rms2s(R)
        wins = [w for w in range(n // WIN)
                if all(lr[i] > ACTIVE and rr[i] > ACTIVE for i in (2 * w, 2 * w + 1))
                and not ({2 * w - 1, 2 * w, 2 * w + 1, 2 * w + 2} & blocked)]
        if not wins:
            results[tid] = {"song": r["song"], "artist": r["artist"], "n": 0, "windows": []}
            json.dump(results, open(out_path, "w"))
            continue
        # the full mix, exactly as tools/demucs_guitar_inputs.py built it
        mix = None
        for p in sorted(glob.glob(os.path.join(gdir, "*.wav"))):
            mix = add(mix, load(p))
        for c, info in stems[tid]["cats"].items():
            mix = add(mix, load(os.path.join(args.stems_dir, tid, f"{c}.flac"), info["gain"]))
        mix = np.pad(mix, ((0, 0), (HOP, HOP + CH)))   # room for the first/last context chunks
        cache = {}

        def chunk_out(s):   # s: chunk start in ORIGINAL samples (may be negative)
            if s not in cache:
                cache[s] = model.guitar(np.ascontiguousarray(mix[:, s + HOP:s + HOP + CH])) * _syn
            return cache[s]

        rows = []
        for w in wins:
            a = w * WIN
            seg = np.zeros((2, 2 * CH), np.float32)   # covers [a - 2s, a + 6s]
            for s in (a - HOP, a, a + HOP):
                seg[:, s - (a - HOP):s - (a - HOP) + CH] += chunk_out(s)
            g = seg[:, HOP:HOP + WIN]   # [a, a + 4s]: every sample summed from two chunks
            sl = slice(a, a + WIN)
            l, re, d = L[:, sl], R_ext[:, sl], D[:, sl]
            rows.append([float(v) for v in (w * 4,
                         sisdr(d, l + re), sisdr(oracle(d, l), l), sisdr(oracle(d, re), re),
                         sisdr(g, l + re), sisdr(oracle(g, l), l), sisdr(oracle(g, re), re))])
        results[tid] = {"song": r["song"], "artist": r["artist"], "n": len(rows), "windows": rows,
                        "cols": ["t", "dm_faith", "dm_L", "dm_R", "rf_faith", "rf_L", "rf_R"]}
        json.dump(results, open(out_path, "w"))
        med = np.median(np.array(rows)[:, 1:], axis=0)
        print(f"[{si}] {r['artist'][:14]} - {r['song'][:22]:22s} n={len(rows):2d} chunks={len(cache):3d} | "
              f"Demucs faith {med[0]:5.1f} L {med[1]:5.1f} R {med[2]:5.1f} | "
              f"Roformer faith {med[3]:5.1f} L {med[4]:5.1f} R {med[5]:5.1f} | {(time.time()-t0)/60:.0f} min",
              flush=True)
        del mix, cache
    print(f"[{si}] done", flush=True)


if __name__ == "__main__":
    main()
