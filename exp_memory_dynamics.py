import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
PAUSE = 150 * ms
COMBOS = [
    {"tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"tau_m": 10 * ms, "tau_syn": 5 * ms},
    {"tau_m": 10 * ms, "tau_syn": 10 * ms},
    {"tau_m": 20 * ms, "tau_syn": 2 * ms},
    {"tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"tau_m": 20 * ms, "tau_syn": 10 * ms},
    {"tau_m": 30 * ms, "tau_syn": 5 * ms},
    {"tau_m": 30 * ms, "tau_syn": 10 * ms},
    {"tau_m": 30 * ms, "tau_syn": 20 * ms},
    {"tau_m": 40 * ms, "tau_syn": 10 * ms},
    {"tau_m": 40 * ms, "tau_syn": 20 * ms},
]


def run_combo(graph, ia, ib, acc, rej, seed, **lif_kwargs):
    start_scope()
    np.random.seed(seed)
    org_a = SensoryOrgan("A", 4, 100 * Hz)
    org_b = SensoryOrgan("B", 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False, **lif_kwargs)
    mbon = np.concatenate([acc, rej])
    pause_ms = PAUSE / ms
    fp = {}
    res = {}
    for word in WORDS:
        pipe.reset_state()
        t0, t1 = pipe.present_word(word, delay=PAUSE)
        t = np.asarray(pipe.mon_brain.t / ms)
        i = pipe.mon_brain.i
        w0, w1 = t1 / ms - pause_ms, t1 / ms
        sel = (t >= w0) & (t <= w1)
        it = i[sel]
        cnt = np.zeros(len(mbon))
        for k, nid in enumerate(mbon):
            cnt[k] = np.isin(it, nid).sum()
        fp[word] = cnt
        res[word] = (sel.sum(), np.isin(it, acc).sum(), np.isin(it, rej).sum())
    return fp, res


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print(f"{'tau_m/tau_syn':16s} | {'tot(ab/ba/aab/aabbb/aabb/bbaa)':40s} | {'d(ab,ba)':>9s} d(aab,abb) d(aabb,bbaa)")
    for c in COMBOS:
        fp, res = run_combo(graph, ia, ib, acc, rej, 0, **c)
        tots = [res[w][0] for w in WORDS]
        ds = [np.abs(fp[a] - fp[b]).sum() for a, b in pairs]
        label = f"{c['tau_m']/ms:.0f}/{c['tau_syn']/ms:.0f}"
        print(f"{label:16s} | {' '.join(f'{t:5d}' for t in tots)} | "
              f"{ds[0]:9.1f} {ds[1]:7.1f} {ds[2]:10.1f}")


if __name__ == "__main__":
    main()