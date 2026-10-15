"""
Trains a small solo-vs-comp (lead vs rhythm) classifier on GuitarSet's
native _solo_mix/_comp_mix labels. This re-derives the diagnostic from the
2026-09-14 "Pivot" session (recorded as ~90% test accuracy, not saved to
the repo at the time) so it can be reused as an auto-labeler for MoisesDB's
unlabeled distorted-electric-guitar tracks (see CLAUDE.md, "The step
change" / role-labeling decision).

Content-disjoint split: clips are split by base recording (not by random
2s-window), so train/test never share audio from the same performance.

Usage:
    source ~/pytorch_env/bin/activate
    python tools/train_role_classifier.py \
        --data output/guitarset/audio_mono-pickup_mix \
        --out output/role_classifier.pt
"""
import argparse
import glob
import os
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

SR = 16000
SEG = SR * 2  # 2s segments, matches the rest of this project's pipeline


class RoleNet(nn.Module):
    """Small CNN over a log-mel spectrogram -- plenty for a 2-class problem."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=256, n_mels=64)
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 2)

    def forward(self, wav):
        x = self.db(self.mel(wav))  # (B, n_mels, T)
        x = x.unsqueeze(1)
        x = self.net(x).flatten(1)
        return self.head(x)


def load_segments(path, label, segs_per_clip=8):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))  # mono
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    n = wav.shape[0]
    out = []
    if n < SEG:
        return out
    starts = torch.linspace(0, n - SEG, segs_per_clip).long().tolist()
    for s in starts:
        seg = wav[s:s + SEG]
        peak = seg.abs().max()
        if peak > 1e-4:
            seg = seg / peak * 0.8
        out.append((seg, label))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="output/role_classifier.pt")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    solo_files = sorted(glob.glob(os.path.join(args.data, "*_solo_mix.wav")))
    comp_files = sorted(glob.glob(os.path.join(args.data, "*_comp_mix.wav")))
    print(f"{len(solo_files)} solo clips, {len(comp_files)} comp clips")

    # content-disjoint split by base recording id (e.g. "00_BN1-129-Eb")
    def base_id(fp):
        name = os.path.basename(fp)
        return name.replace("_solo_mix.wav", "").replace("_comp_mix.wav", "")

    all_ids = sorted(set(base_id(f) for f in solo_files + comp_files))
    rng = random.Random(0)
    rng.shuffle(all_ids)
    n_test = max(1, int(0.2 * len(all_ids)))
    test_ids = set(all_ids[:n_test])
    train_ids = set(all_ids[n_test:])
    print(f"{len(train_ids)} train recordings, {len(test_ids)} test recordings (content-disjoint)")

    def build(ids):
        data = []
        for f in solo_files:
            if base_id(f) in ids:
                data += load_segments(f, 1)
        for f in comp_files:
            if base_id(f) in ids:
                data += load_segments(f, 0)
        return data

    train_data = build(train_ids)
    test_data = build(test_ids)
    print(f"{len(train_data)} train segments, {len(test_data)} test segments")

    model = RoleNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def batches(data, bs=16, shuffle=True):
        idx = list(range(len(data)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [data[j] for j in idx[i:i + bs]]
            wavs = torch.stack([c[0] for c in chunk])
            labels = torch.tensor([c[1] for c in chunk])
            yield wavs, labels

    def evaluate(data):
        model.eval()
        correct = 0
        solo_correct = solo_total = comp_correct = comp_total = 0
        with torch.no_grad():
            for wavs, labels in batches(data, shuffle=False):
                preds = model(wavs).argmax(1)
                correct += (preds == labels).sum().item()
                for p, l in zip(preds.tolist(), labels.tolist()):
                    if l == 1:
                        solo_total += 1
                        solo_correct += int(p == l)
                    else:
                        comp_total += 1
                        comp_correct += int(p == l)
        model.train()
        acc = correct / max(1, len(data))
        return acc, solo_correct / max(1, solo_total), comp_correct / max(1, comp_total)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for wavs, labels in batches(train_data):
            opt.zero_grad()
            logits = model(wavs)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        test_acc, solo_recall, comp_recall = evaluate(test_data)
        print(f"epoch {epoch+1:2d}  train_loss {total_loss/n_batches:.3f}  "
              f"test_acc {test_acc*100:.1f}%  solo_recall {solo_recall*100:.1f}%  "
              f"comp_recall {comp_recall*100:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
