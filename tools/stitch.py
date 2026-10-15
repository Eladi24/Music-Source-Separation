"""Cross-chunk routing continuity for role-separation inference.

The problem (CLAUDE.md, leak-loss + amp-realism real-song tests): the real-song
inference path processes the song as independent overlapping 2s chunks and
assigns est[0]->out0 unconditionally. The model is role-trained (99% in-domain
role-acc) so its native ordering is usually right, but on locally ambiguous
real audio the per-chunk decision can flip. Overlap-adding two chunks that
disagree sums a lead-leaning signal onto a rhythm-leaning one -> destructive
interference ("white noise/static") with brief peaks where consecutive chunks
happen to agree.

Stage 1's blind model used greedy correlation chaining against the previous
chunk. That fixes local flapping but is irrevocable: one bad early decision
propagates forever, and it has no notion of global polarity (the whole song
can end up lead-in-out1).

This module instead solves the whole chunk sequence at once as a 2-state
Viterbi (keep / swap per chunk):
  transition cost -- disagreement between consecutive chunks in their overlap
  emission cost   -- a small prior toward the model's native ordering, which
                     is what supplies global polarity and lets a locally
                     ambiguous stretch be overridden by the confident
                     majority around it
Cheap (n_chunks x 2 states) and globally optimal for this cost, unlike greedy.
"""
import numpy as np


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def pair_agreement(prev0, prev1, cur0, cur1):
    """Agreement of (cur0,cur1) with (prev0,prev1) in a shared overlap region,
    for both orderings. Returns (same, swap) in [-2, 2]."""
    same = _corr(cur0, prev0) + _corr(cur1, prev1)
    swap = _corr(cur1, prev0) + _corr(cur0, prev1)
    return same, swap


def viterbi_orient(chunks, lambda_prior=0.15):
    """chunks: list of (e0, e1) float arrays, uniform length, consecutive
    chunks overlapping by hop = len/2 (second half of i overlaps first half
    of i+1).

    Returns a bool array `swap`, where swap[i] means chunk i's outputs should
    be exchanged before overlap-add.
    """
    n = len(chunks)
    if n == 0:
        return np.zeros(0, dtype=bool)
    hop = len(chunks[0][0]) // 2
    NEG = 1e9
    # cost[i][s] = best total cost of orienting chunks 0..i with chunk i in state s
    cost = np.full((n, 2), NEG)
    back = np.zeros((n, 2), dtype=int)
    cost[0, 0] = 0.0
    cost[0, 1] = lambda_prior            # deviating from the model's native ordering
    for i in range(1, n):
        p0, p1 = chunks[i - 1]
        c0, c1 = chunks[i]
        # previous chunk's tail vs current chunk's head
        pt0, pt1 = p0[hop:], p1[hop:]
        ch0, ch1 = c0[:hop], c1[:hop]
        same, swap = pair_agreement(pt0, pt1, ch0, ch1)
        # transition cost: higher agreement = lower cost. prev state s_prev
        # already fixes which of pt0/pt1 is "out0", so a prev swap exchanges
        # the roles of same/swap.
        for s_prev in (0, 1):
            for s_cur in (0, 1):
                aligned = (s_prev == s_cur)
                agree = same if aligned else swap
                t = -agree + (lambda_prior if s_cur == 1 else 0.0)
                v = cost[i - 1, s_prev] + t
                if v < cost[i, s_cur]:
                    cost[i, s_cur] = v
                    back[i, s_cur] = s_prev
    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states.astype(bool)


def greedy_orient(chunks):
    """Stage 1's approach, kept for comparison: chain each chunk against the
    previous one's already-chosen orientation. Irrevocable per decision."""
    n = len(chunks)
    swap = np.zeros(n, dtype=bool)
    if n == 0:
        return swap
    hop = len(chunks[0][0]) // 2
    for i in range(1, n):
        p0, p1 = chunks[i - 1]
        if swap[i - 1]:
            p0, p1 = p1, p0
        c0, c1 = chunks[i]
        same, sw = pair_agreement(p0[hop:], p1[hop:], c0[:hop], c1[:hop])
        swap[i] = sw > same
    return swap


def overlap_add(chunks, swap, total_len, starts):
    """Hann-window overlap-add with per-chunk orientation applied."""
    chunk = len(chunks[0][0])
    win = np.hanning(chunk)
    o0 = np.zeros(total_len); o1 = np.zeros(total_len); ws = np.zeros(total_len)
    for (e0, e1), sw, s in zip(chunks, swap, starts):
        if sw:
            e0, e1 = e1, e0
        m = min(chunk, total_len - s)
        o0[s:s + m] += e0[:m] * win[:m]
        o1[s:s + m] += e1[:m] * win[:m]
        ws[s:s + m] += win[:m]
    ws[ws < 1e-8] = 1e-8
    return o0 / ws, o1 / ws


def measure_log_alpha(chunks, idx):
    """log of the scale mapping chunk i-1's estimate onto chunk i's, in their
    shared overlap, for output `idx`. la[0] is unused."""
    n = len(chunks)
    hop = len(chunks[0][0]) // 2
    la = np.zeros(n)
    for i in range(1, n):
        prv, cur = chunks[i - 1][idx][hop:], chunks[i][idx][:hop]
        den = float((prv * prv).sum()) + 1e-12
        a = float((cur * prv).sum() / den)
        la[i] = np.log(min(max(abs(a), 1e-3), 1e3))
    return la


def solve_gains(chunks, lam=0.1):
    """Per-chunk, per-output gain correction.

    Measured on the real song: overall level is stable (mixture-consistency
    beta sd 0.005) but the ALLOCATION between out0 and out1 swings up to 8x
    chunk-to-chunk -- because SI-SDR is scale-invariant, so training never
    constrained how much of the mixture each output claims. Hann overlap-add
    turns that disagreement into level modulation at the hop rate, which is
    the long-standing "prominence fluctuates within a passage" symptom.

    Fix: choose a log-gain x_i per chunk so consecutive chunks agree in their
    overlap, minimising
        sum_i (x_i - x_{i-1} + log alpha_i)^2  +  lam * sum_i x_i^2
    The lam term pulls gains back toward the model's native output (x=0),
    which both regularises the solution and stops error accumulating into
    drift over a few hundred chunks -- the failure mode of chaining each
    chunk off its predecessor. Closed-form linear solve, not iterative.
    """
    n = len(chunks)
    g = np.ones((n, 2))
    if n < 2:
        return g
    for idx in (0, 1):
        la = measure_log_alpha(chunks, idx)
        D = np.zeros((n - 1, n))
        for i in range(1, n):
            D[i - 1, i] = 1.0
            D[i - 1, i - 1] = -1.0
        A = D.T @ D + lam * np.eye(n)
        b = -D.T @ la[1:]
        x = np.linalg.solve(A, b)
        g[:, idx] = np.exp(x)
    return g
