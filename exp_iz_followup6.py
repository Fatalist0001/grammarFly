import sys
sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.lif import make_izhikevich

PAUSE = 300 * ms
BINS = [(0, 50), (50, 100), (100, 200), (200, 300)]
W_IN = 0.8 * nA
w_scale = 0.002 * nA


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


def run_window(pipe, mbon, c, b, word):
    pipe.brain.v = c
    pipe.brain.u = b * c
    pipe.brain.I = 0 * 1e-12 * nA
    t0, t1 = pipe.present_word(word, delay=PAUSE)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    end = t1 / ms
    t1s = end - PAUSE / ms
    out = {}
    for (a0, a1) in BINS:
        sel = (t >= t1s + a0) & (t <= t1s + a1)
        it = i[sel]
        out[(a0, a1)] = np.array([np.isin(it, nid).sum() for nid in mbon])
    return out


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    configs = [
        ("rs-base",      dict(preset="rs"), {}),
        ("a=0.01 d=16",  dict(preset=None, a=0.01, b=0.2, d=16.0 * mV), {}),
    ]
    for label, nz, extra in configs:
        print(f"\n=== {label}: MBON spike-count norm per bin, AB vs BA ===")
        for sd in [0, 5]:
            pipe, c, b = build(3000 + sd, nz, extra, graph, acc, rej)
            for word in ["AB", "BA"]:
                r = run_window(pipe, mbon, c, b, word)
                norms = {bl: float(r[bl].sum()) for bl in BINS}
                print(f"  seed{sd} {word}: " + "  ".join(
                    f"{bl[0]:>3}-{bl[1]:<3}ms:{v:6.0f}" for bl, v in norms.items()))


if __name__ == "__main__":
    main()