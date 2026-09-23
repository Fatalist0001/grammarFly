import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope, amp, volt

from organs.sensory import SensoryOrgan
from brain.malecns import MaleCNSGraph
from brain.pipeline import Pipeline
from brain.lif import make_izhikevich

WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
PAUSE = 150 * ms
K = 5
W_IN = 0.8 * nA
w_scale = 0.002 * nA  # working point found in calibration (RS region)

# (label, preset-or-dict, extra kwargs)
CONFIGS = [
    ("rs-base",     dict(preset="rs"), {}),
    ("b=0.1",       dict(preset=None, a=0.02, b=0.1), {}),
    ("b=0.3",       dict(preset=None, a=0.02, b=0.3), {}),
    ("d=2",         dict(preset=None, a=0.02, b=0.2, d=2.0 * mV), {}),
    ("d=16",        dict(preset=None, a=0.02, b=0.2, d=16.0 * mV), {}),
    ("a=0.01",      dict(preset=None, a=0.01, b=0.2), {}),
    ("a=0.005 d=8", dict(preset=None, a=0.005, b=0.2, d=8.0 * mV), {}),
    ("ch",          dict(preset="ch"), {}),
    ("ib",          dict(preset="ib"), {}),
    ("slow-u a01d16", dict(preset=None, a=0.01, b=0.2, d=16.0 * mV), {}),
    ("rs tau_syn32",  dict(preset="rs"), {"tau_syn": 32 * ms}),
    ("b.1 tau_syn32", dict(preset=None, a=0.02, b=0.1), {"tau_syn": 32 * ms}),
]


def iz_reset(pipe, c, b):
    pipe.brain.v = c
    pipe.brain.u = b * c
    pipe.brain.I = 0 * amp


def run(pipe, mbon, c, b):
    pause_ms = PAUSE / ms
    fp = {w: [] for w in WORDS}
    tots = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            iz_reset(pipe, c, b)
            t0, t1 = pipe.present_word(word, delay=PAUSE)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            sel = (t >= w0) & (t <= w1)
            it = i[sel]
            tots[word].append(sel.sum())
            fp[word].append(np.array([np.isin(it, nid).sum() for nid in mbon]))
    return fp, tots


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    mbon = np.concatenate([acc, rej])
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print(f"{'config':18s} {'word':6s} {'tot':>7s} | {'d_intra':>8s} "
          f"{'d_ABBA':>7s} {'d_AABAB':>8s} {'d_AABB':>7s} | {'ratio':>6s} "
          f"{'pos%':>5s}")
    for label, nz, extra in CONFIGS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        c = nz.get("c", -65 * mV)
        b = nz.get("b", 0.2)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        w_in=W_IN, w_scale=w_scale,
                        plastic=False, neuron_fn=make_izhikevich,
                        **nz, **extra)
        fp, tots = run(pipe, mbon, c, b)
        means = {w: np.mean(fp[w], 0) for w in WORDS}
        cond_min_ratio = 1e9
        for w in WORDS:
            intra = np.mean([np.abs(f - means[w]).sum() for f in fp[w]])
            vals = []
            for a, wb in pairs:
                if w == a:
                    vals.append(np.abs(means[a] - means[wb]).sum())
                else:
                    vals.append(float("nan"))
            nonnan = [v for v in vals if v == v] or [intra]
            ratio = intra / (intra + min(nonnan)) if min(nonnan) > 0 else 1.0
            cond_min_ratio = min(cond_min_ratio, ratio)
            # pos%: readout in pause window
            tot_acc = np.mean([np.isin(np.unique([np.isin(u, acc) for u in fp[w]]), [True]).mean()]) if False else None
            print(f"{label:18s} {w:6s} {np.mean(tots[w]):7.1f} | {intra:8.1f} "
                  + " ".join(f"{v:7.1f}" if v == v else "     -- " for v in vals)
                  + f" | {ratio:6.3f}")
        print(f"  -> best ratio across words for {label}: {cond_min_ratio:.3f}")
        print()


if __name__ == "__main__":
    main()