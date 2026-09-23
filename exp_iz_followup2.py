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
WINDOW_BIN = (50, 150)  # ms within pause
K = 8
W_IN = 0.8 * nA
w_scale = 0.002 * nA
BASE_SEED = 1000


def iz_reset(pipe, c, b):
    pipe.brain.v = c
    pipe.brain.u = b * c
    pipe.brain.I = 0 * amp


def run_once(pipe, mbon, c, b, word):
    iz_reset(pipe, c, b)
    t0, t1 = pipe.present_word(word, delay=PAUSE)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    end = t1 / ms
    w0, w1 = end - PAUSE / ms + WINDOW_BIN[0], end - PAUSE / ms + WINDOW_BIN[1]
    sel = (t >= w0) & (t <= w1)
    it = i[sel]
    return np.array([np.isin(it, nid).sum() for nid in mbon]), len(t)


def norm_dist(x, y):
    return np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y) + 1e-12)


def build(seed, nz, extra, graph, acc, rej):
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
    return pipe, c, b


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])

    configs = [
        ("slow-u a.01 d16", dict(preset=None, a=0.01, b=0.2, d=16.0 * mV), {}),
        ("ch",              dict(preset="ch"), {}),
        ("a.01 d8",         dict(preset=None, a=0.01, b=0.2, d=8.0 * mV), {}),
    ]
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]

    for label, nz, extra in configs:
        prof = {w: [] for w in WORDS}
        tot = {w: [] for w in WORDS}
        for k in range(K):
            pipe, c, b = build(BASE_SEED + k, nz, extra, graph, acc, rej)
            for word in WORDS:
                v, t = run_once(pipe, mbon, c, b, word)
                prof[word].append(v)
                tot[word].append(t)
        print(f"\n=== {label} window {WINDOW_BIN[0]}-{WINDOW_BIN[1]}ms, K={K} ===")
        for (wa, wb) in pairs:
            A = np.array(prof[wa]); B = np.array(prof[wb])
            intraA = np.mean([norm_dist(A[i], A[j]) for i in range(K)
                              for j in range(i + 1, K)])
            intraB = np.mean([norm_dist(B[i], B[j]) for i in range(K)
                              for j in range(i + 1, K)])
            interAB = np.mean([norm_dist(A[i], B[j]) for i in range(K)
                               for j in range(K)])
            inter_same = np.mean([norm_dist(A[i], B[i]) for i in range(K)])
            print(f"  {wa}/{wb}: intraA={intraA:.3f} intraB={intraB:.3f} "
                  f"inter(all)={interAB:.3f} inter(same-k)={inter_same:.3f} "
                  f"totA={np.mean(tot[wa]):.0f} totB={np.mean(tot[wb]):.0f}")


if __name__ == "__main__":
    main()