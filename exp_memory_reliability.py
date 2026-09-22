import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
PAUSE = 150 * ms
K = 5
COMBOS = [
    {"tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"tau_m": 30 * ms, "tau_syn": 5 * ms},
    {"tau_m": 40 * ms, "tau_syn": 10 * ms},
    {"tau_m": 30 * ms, "tau_syn": 10 * ms},
]


def fingerprints(pipe, mbon, seed):
    pause_ms = PAUSE / ms
    fp = {w: [] for w in WORDS}
    tots = {w: [] for w in WORDS}
    for trial in range(K):
        for word in WORDS:
            pipe.reset_state()
            t0, t1 = pipe.present_word(word, delay=PAUSE)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            sel = (t >= w0) & (t <= w1)
            it = i[sel]
            tots[word].append(sel.sum())
            cnt = np.array([np.isin(it, nid).sum() for nid in mbon])
            fp[word].append(cnt)
    return fp, tots


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print(f"{'tau':8s} {'word':6s} {'tot_mean':>8s} | "
          f"{'d_intra':>8s} {'d_AB_BA':>8s} {'d_AAB_ABB':>10s} {'d_AABB_BBAA':>11s} | {'ratio':>6s}")
    for c in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        0.8 * nA, 0.001 * nA, plastic=False, **c)
        fp, tots = fingerprints(pipe, mbon, 0)
        means = {w: np.mean(fp[w], 0) for w in WORDS}
        intra = {}
        for w in WORDS:
            m = means[w]
            intra[w] = np.mean([np.abs(f - m).sum() for f in fp[w]])
        for w in WORDS:
            d_intra = intra[w]
            ds = []
            for a, b in pairs:
                if w == a:
                    ds.append(np.abs(means[a] - means[b]).sum())
                elif w == b:
                    ds.append(np.abs(means[b] - means[a]).sum())
                else:
                    ds.append(float("nan"))
            ratio = d_intra / (d_intra + min(d for d in ds if d == d))
            core = f"{c['tau_m']/ms:.0f}/{c['tau_syn']/ms:.0f}"
            print(f"{core:8s} {w:6s} {np.mean(tots[w]):8.1f} | "
                  f"{d_intra:8.1f} "
                  + " ".join(f"{d:8.1f}" if d == d else "      -- " for d in ds)
                  + f" | {ratio:6.2f}")
        print()


if __name__ == "__main__":
    main()