import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
PAUSE = 150 * ms
K = 5
COMBOS = [
    {"tau_m": 10 * ms, "tau_syn": 2 * ms, "tau_ahp": 100 * ms, "w_ahp": 1 * mV},
    {"tau_m": 10 * ms, "tau_syn": 2 * ms, "tau_ahp": 100 * ms, "w_ahp": 2 * mV},
    {"tau_m": 10 * ms, "tau_syn": 5 * ms, "tau_ahp": 100 * ms, "w_ahp": 1 * mV},
    {"tau_m": 20 * ms, "tau_syn": 5 * ms, "tau_ahp": 100 * ms, "w_ahp": 1 * mV},
    {"tau_m": 20 * ms, "tau_syn": 5 * ms, "tau_ahp": 200 * ms, "w_ahp": 2 * mV},
    {"tau_m": 30 * ms, "tau_syn": 5 * ms, "tau_ahp": 100 * ms, "w_ahp": 0.5 * mV},
    {"tau_m": 30 * ms, "tau_syn": 5 * ms, "tau_ahp": 200 * ms, "w_ahp": 1 * mV},
]


def run(pipe, mbon):
    pause_ms = PAUSE / ms
    fp = {w: [] for w in WORDS}
    tots = {w: [] for w in WORDS}
    acc_c = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            pipe.reset_state()
            t0, t1 = pipe.present_word(word, delay=PAUSE)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            sel = (t >= w0) & (t <= w1)
            it = i[sel]
            tots[word].append(sel.sum())
            acc_c[word].append(np.isin(it, pipe.out_acc).sum())
            fp[word].append(np.array([np.isin(it, nid).sum() for nid in mbon]))
    return fp, tots, acc_c


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print(f"{'combo':22s} {'word':6s} {'tot':>7s} {'acc':>5s} | "
          f"{'d_intra':>8s} {'d_ABBA':>7s} {'d_AABABB':>8s} {'d_AABB_BA':>9s} | {'ratio':>6s}")
    for c in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        0.8 * nA, 0.001 * nA, plastic=False, **c)
        fp, tots, acc_c = run(pipe, mbon)
        means = {w: np.mean(fp[w], 0) for w in WORDS}
        label = f"{c['tau_m']/ms:.0f}/{c['tau_syn']/ms:.0f} ahp={c.get('tau_ahp', 0*ms)/ms:.0f}/{c.get('w_ahp', 0*mV)/mV:.1f}"
        for w in WORDS:
            intra = np.mean([np.abs(f - means[w]).sum() for f in fp[w]])
            vals = []
            for a, b in pairs:
                if w == a:
                    vals.append(np.abs(means[a] - means[b]).sum())
                else:
                    vals.append(float("nan"))
            nonnan = [v for v in vals if v == v] or [intra]
            ratio = intra / (intra + min(nonnan))
            print(f"{label:22s} {w:6s} {np.mean(tots[w]):7.1f} {np.mean(acc_c[w]):5.1f} | "
                  f"{intra:8.1f} "
                  + " ".join(f"{v:7.1f}" if v == v else "     -- " for v in vals)
                  + f" | {ratio:6.2f}")
        print()


if __name__ == "__main__":
    main()