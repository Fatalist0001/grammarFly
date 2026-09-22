import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

PAUSES_MS = [0, 25, 50, 100, 200]
PAIRS = [('AB', 'BA'), ('AAB', 'ABB'), ('AABB', 'BBAA')]


def build(seed=0):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)
    return pipe, acc, rej


def window_counts(pipe, word, pause_ms, bin_ms=10):
    """Return MBON spike counts per time bin after word end (during pause)."""
    pipe.reset_state()
    t0, t1 = pipe.present_word(word, delay=pause_ms * ms)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return np.zeros(0), np.zeros(0)
    nbins = max(1, int(pause_ms / bin_ms))
    acc, rej = np.zeros(nbins), np.zeros(nbins)
    for b in range(nbins):
        lo = t1 / ms - pause_ms + b * bin_ms
        hi = lo + bin_ms
        sel = (t >= lo) & (t < hi)
        acc[b] = np.sum(sel & np.isin(i, pipe.out_acc))
        rej[b] = np.sum(sel & np.isin(i, pipe.out_rej))
    return acc, rej


def main():
    print(f"{'pair':10s} {'pause':>6s} | {'MBON_acc bin-sum (word1 vs word2)':<38s} {'MBON_rej':<38s} "
          f"{'acc>rej?':>10s}")
    for w1, w2 in PAIRS:
        for pause_ms in PAUSES_MS:
            diffs = []
            for seed in range(3):
                pipe, _, _ = build(seed)
                a1, r1 = window_counts(pipe, w1, pause_ms)
                a2, r2 = window_counts(pipe, w2, pause_ms)
                # pool over pause bins
                d = (a1.sum() / max(len(a1), 1) - a2.sum() / max(len(a2), 1))
                diffs.append(d)
            md = np.mean(diffs)
            print(f"{w1+'/'+w2:10s} {pause_ms:5d}ms | ... diff acc={md:+.3f}")
    print()
    print("Detailed per 25-ms bins for 200ms pause (seed 0):")
    pipe, acc, rej = build(0)
    for w1, w2 in PAIRS:
        a1, r1 = window_counts(pipe, w1, 200)
        a2, r2 = window_counts(pipe, w2, 200)
        line_a = ' '.join(f"{a1[i]:>3.0f}/{a2[i]:<3.0f}" for i in range(len(a1)))
        line_r = ' '.join(f"{r1[i]:>3.0f}/{r2[i]:<3.0f}" for i in range(len(r1)))
        print(f"  {w1+'/'+w2:10s} acc bins: {line_a}")
        print(f"  {w1+'/'+w2:10s} rej bins: {line_r}")


if __name__ == "__main__":
    main()