"""Band-balance loss, targeting the measured treble-starvation defect.

THE DEFECT (measured 2026-10-14 on lead4_epoch5.pt's real-song output):
out0 (lead) sits +5.0dB vs the input stem at 3-5kHz while out1 (rhythm) sits
-6.6dB -- an 11.6dB swing. The model uses brightness as a ROUTING cue and
dumps high frequencies into the lead output, starving the rhythm output.
That is what "treble is all the way down"/"underwater" describes.

WHY TRAINING ALLOWS IT: SI-SDR is energy-weighted and the targets put 93%
(solo) / 96.5% (comp) of their energy below 1kHz. The 3-5kHz band holds
0.86% / 0.13%, so the destroyed band is well under 1% of the objective.
Comp targets have 6.6x less relative treble than solo targets -- hence the
damage lands on the rhythm branch specifically.

A REJECTED DESIGN, worth recording: per-band SI-SDR, averaged over bands.
It fails structurally -- SI-SDR is scale-invariant, so computing it PER BAND
makes each band's level unconstrained, which is the very thing being fixed.
Validation showed it scoring an estimate with +12dB excess treble HIGHER
than a perfect one. Do not revisit that formulation.

WHAT THIS DOES INSTEAD: compare the per-band energy DISTRIBUTION of estimate
and reference, in dB, after normalising each by its own total energy. That
is invariant to overall level (as SI-SDR is, and as the model's output
genuinely is -- see the beta=-0.5 finding) but directly sensitive to getting
the band balance wrong, in either direction. A deadzone keeps it from
fighting target fidelity over inaudibly small errors, the same bounding
philosophy as the routing hinge and the leak term.
"""
import torch

BANDS = ((20, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6500))
PRESENCE_FLOOR_DB = -60.0   # a reference band quieter than this is ignored, not optimised
DEADZONE_DB = 1.0           # per-band error below this is free


def band_split(x, sr, bands=BANDS):
    """Zero all bins outside each band, transform back. Exact, differentiable,
    and free of the filter phase distortion that confounded an earlier test.
    Returns (n_bands, *x.shape)."""
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(n, 1.0 / sr, device=x.device)
    return torch.stack([torch.fft.irfft(X * ((f >= lo) & (f < hi)).to(X.dtype), n=n, dim=-1)
                        for lo, hi in bands], 0)


def band_energy(x, sr, bands=BANDS, eps=1e-12):
    """Per-band energy as a fraction of the signal's own total. (n_bands, batch)"""
    b = band_split(x, sr, bands)
    E = (b ** 2).sum(-1)
    return E / E.sum(0, keepdim=True).clamp_min(eps)


def band_balance_error(est, ref, sr, bands=BANDS, floor_db=PRESENCE_FLOOR_DB,
                       deadzone_db=DEADZONE_DB, eps=1e-12):
    """Mean |dB| error of the band-energy distribution, over present bands.

    est/ref: (batch, time). Returns (batch,) in dB -- directly interpretable
    as "average per-band dB of spectral-balance error".
    """
    Pe = band_energy(est, sr, bands, eps)
    Pr = band_energy(ref, sr, bands, eps)
    d = 10 * torch.log10(Pe.clamp_min(eps) / Pr.clamp_min(eps))
    w = (10 * torch.log10(Pr.clamp_min(eps)) > floor_db).to(d.dtype)
    w = torch.where(w.sum(0, keepdim=True) > 0, w, torch.ones_like(w))
    pen = torch.relu(d.abs() - deadzone_db)
    return (pen * w).sum(0) / w.sum(0)


def band_loss(est, S, C, sr, bands=BANDS):
    """For the routing-solved outputs: out0 -> solo, out1 -> comp. No PIT --
    role-acc is at ceiling (99%), the same reasoning the STFT-loss experiment
    used. Returns (scalar loss to ADD, out0 error dB, out1 error dB)."""
    e0 = band_balance_error(est[:, 0], S, sr, bands)
    e1 = band_balance_error(est[:, 1], C, sr, bands)
    return (e0 + e1).mean() / 2, e0.detach().mean(), e1.detach().mean()
