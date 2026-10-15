"""
Adapts pretrained HTDemucs (4 sources: drums/bass/other/vocals) for a
2-source lead/rhythm guitar separation task -- see CLAUDE.md, "The step
change". Only the two final output layers encode the source count
(everything else -- encoder, cross-transformer, the rest of the decoder --
is source-count-independent, confirmed by reading demucs/htdemucs.py:
`chin_z = audio_channels * len(sources) * (2 if cac else 1)` at the very
last decoder layer only). Those two layers are replaced with freshly
sized, randomly initialized ones; everything upstream keeps its pretrained
weights, warm-starting from real-music-trained representations the same
way the lead/rhythm pivot warm-started from checkpoint_epoch50.pt.
"""
import torch
import torch.nn as nn

from demucs.pretrained import get_model


def build_leadrhythm_htdemucs(sources=("lead", "rhythm")):
    bag = get_model("htdemucs")
    model = bag.models[0]  # htdemucs (not _ft) is a bag of exactly 1 model

    n_src = len(sources)
    audio_ch = model.audio_channels
    cac_mult = 2 if model.cac else 1

    old_freq = model.decoder[-1].conv_tr
    new_freq_out = audio_ch * n_src * cac_mult
    new_freq = nn.ConvTranspose2d(
        old_freq.in_channels, new_freq_out,
        kernel_size=old_freq.kernel_size, stride=old_freq.stride)
    model.decoder[-1].conv_tr = new_freq

    old_time = model.tdecoder[-1].conv_tr
    new_time_out = audio_ch * n_src
    new_time = nn.ConvTranspose1d(
        old_time.in_channels, new_time_out,
        kernel_size=old_time.kernel_size, stride=old_time.stride)
    model.tdecoder[-1].conv_tr = new_time

    model.sources = list(sources)

    return model, bag


def count_params(model):
    backbone = sum(p.numel() for n, p in model.named_parameters()
                   if "decoder.3.conv_tr" not in n and "tdecoder.3.conv_tr" not in n)
    new_head = sum(p.numel() for n, p in model.named_parameters()
                   if "decoder.3.conv_tr" in n or "tdecoder.3.conv_tr" in n)
    return backbone, new_head


if __name__ == "__main__":
    model, bag = build_leadrhythm_htdemucs()
    backbone, new_head = count_params(model)
    print(f"backbone (pretrained, carried over): {backbone:,} params")
    print(f"new output head (randomly initialized): {new_head:,} params")
    print(f"total: {backbone + new_head:,}")

    model.eval()
    sr = model.samplerate
    mix = torch.randn(1, 2, sr * 2)  # 1 batch, stereo, 2 seconds @ 44.1kHz

    import resource
    with torch.no_grad():
        out = model(mix)
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\noutput shape: {out.shape}  (expect [1, {len(model.sources)}, 2, {sr*2}])")
    print(f"output finite: {torch.isfinite(out).all().item()}")
    print(f"peak RSS so far: {peak_rss_mb:.0f} MB")
