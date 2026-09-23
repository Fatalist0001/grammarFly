import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope, amp

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.lif import make_izhikevich

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
PAUSE = 300 * ms
BINS = [(0, 50), (50, 100), (100, 200), (200, 300)]  # ms within pause
K = 6
W_IN = 0.8 * nA
w_scale = 0.002 * nA
SEEDS = [0, 1, 2]


def iz_reset(pipe, c, b):
    pipe.brain.v = c
    pipe.brain.u = b * c
    pipe.brain.I = 0 * amp


def run_bins(pipe, mbon, c, b):
    """Return fp[bin][word] = list of K per-neuron count vectors."""
    pause_ms = PAUSE / ms
    out = {bl: {w: [] for w in WORDS} for bl in BINS}
    tots = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            iz_reset(pipe, c, b)
            t0, t1 = pipe.present_word(word, delay=PAUSE)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            end = t1 / ms
            tots[word].append(len(t))
            for (a0, a1) in BINS:
                w0, w1 = end - pause_ms + a0, end - pause_ms + a1
                sel = (t >= w0) & (t <= w1)
                it = i[sel]
                out[(a0, a1)][word].append(
                    np.array([np.isin(it, nid).sum() for nid in mbon]))
    return out, tots


def norm_dist(x, y):
    return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y) + 1e-12)


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]

    configs = [
        ("slow-u a.01 d16", dict(preset=None, a=0.01, b=0.2, d=16.0 * mV), {}),
        ("ch",              dict(preset="ch"), {}),
        ("b=0.1",           dict(preset=None, a=0.02, b=0.1), {}),
        ("a.02 d16",        dict(preset=None, a=0.02, b=0.2, d=16.0 * mV), {}),
        ("a.01 d8",         dict(preset=None, a=0.01, b=0.2, d=8.0 * mV), {}),
        ("a.005 d16",       dict(preset=None, a=0.005, b=0.2, d=16.0 * mV), {}),
    ]

    for label, nz, extra in configs:
        agg = {bl: {p: [] for p in pairs} for bl in BINS}
        tots_all = []
        for seed in SEEDS:
            start_scope()
            np.random.seed(seed)
            ia, ib = graph.kc_indices(100, seed=seed)
            org_a = SensoryOrgan("A", 4, 100 * Hz)
            org_b = SensoryOrgan("B", 4, 100 * Hz)
            c = nz.get("c", -65 * mV)
            b = nz.get("b", 0.2)
            pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                            w_in=W_IN, w_scale=w_scale,
                            plastic=False, neuron_fn=make_izhikevich,
                            **nz, **extra)
            fp, tots = run_bins(pipe, mbon, c, b)
            tots_all.append(tots)
            for bl in BINS:
                for (wa, wb) in pairs:
                    A = np.array(fp[bl][wa])   # K x n
                    B = np.array(fp[bl][wb])
                    mA, mB = A.mean(0), B.mean(0)
                    intra = np.mean([norm_dist(Ai, Bi) for i in range(K)
                                     for j in range(i + 1, K)
                                     for Ai, Bi in [(A[i], A[j])]]
                                    ) if K > 1 else 0
                    intraB = np.mean([norm_dist(B[i], B[j]) for i in range(K)
                                      for j in range(i + 1, K)]) if K > 1 else 0
                    inter = norm_dist(mA, mB)
                    ratio = (intra + intraB) / ((intra + intraB) + inter + 1e-12)
                    agg[bl][(wa, wb)].append(ratio)
        print(f"\n=== {label} (3 seeds, 6 runs each) ===")
        tot_mean = {w: np.mean([tots_all[s][w] for s in range(len(SEEDS))]) for w in WORDS}
        print("  pause tot (whole brain): " +
              ", ".join(f"{w}:{tot_mean[w]:.0f}" for w in WORDS))
        for bl in BINS:
            row = []
            for p in pairs:
                r = np.mean(agg[bl][p])
                row.append(f"{p[0]}/{p[1]}={r:.3f}")
            print(f"  bin {bl[0]:>3d}-{bl[1]:>3d}ms: " + "  ".join(row))


if __name__ == "__main__":
    main()