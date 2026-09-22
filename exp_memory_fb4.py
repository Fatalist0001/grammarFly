import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

PA = 0.001 * nA
WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
K = 6
PAUSE_MS = 150
NBIN = 3
COMBOS = [
    # fb x inh balance
    ("fb16 inh2 kk.3 w3", dict(w_scale=3 * PA, fb_boost=16.0, kc_kk_scale=0.30, inh_scale=2.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb16 inh4 kk.3 w3", dict(w_scale=3 * PA, fb_boost=16.0, kc_kk_scale=0.30, inh_scale=4.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb32 inh4 kk.3 w3", dict(w_scale=3 * PA, fb_boost=32.0, kc_kk_scale=0.30, inh_scale=4.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb64 inh4 kk.3 w3", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30, inh_scale=4.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb64 inh8 kk.3 w3", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30, inh_scale=8.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb16 inh4 kk.3 w2", dict(w_scale=2 * PA, fb_boost=16.0, kc_kk_scale=0.30, inh_scale=4.0,
                               tau_m=10 * ms, tau_syn=2 * ms)),
    ("fb32 inh4 kk.3 w2 slow", dict(w_scale=2 * PA, fb_boost=32.0, kc_kk_scale=0.50, inh_scale=4.0,
                                    tau_m=20 * ms, tau_syn=5 * ms)),
    ("fb16 inh2 kk.5 w2 slow", dict(w_scale=2 * PA, fb_boost=16.0, kc_kk_scale=0.50, inh_scale=2.0,
                                    tau_m=20 * ms, tau_syn=5 * ms)),
]


def run(pipe, pause_ms):
    mb = np.concatenate([pipe.out_acc, pipe.out_rej])
    fp = {w: [] for w in WORDS}
    tots_bin = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            pipe.reset_state()
            t0, t1 = pipe.present_word(word, delay=pause_ms * ms)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            bins_v = []
            for k in range(NBIN):
                a = w0 + k * (w1 - w0) / NBIN
                b = w0 + (k + 1) * (w1 - w0) / NBIN
                selk = (t >= a) & (t <= b)
                ik = i[selk]
                bins_v.append(np.array([np.isin(ik, nid).sum() for nid in mb]))
            fp[word].append(np.concatenate(bins_v))
            tots_bin[word].append([sel.shape[0] for sel in
                                   [(t >= w0 + k * (w1 - w0) / NBIN) & (t <= w0 + (k + 1) * (w1 - w0) / NBIN)
                                    for k in range(NBIN)]])
    return fp, tots_bin


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    print(f"{'config':30s} {'AB/BA':>6s} {'AAB/ABB':>8s} {'AABB/BBAA':>10s} "
          f"{'b1':>6s} {'b2':>6s} {'b3':>6s} (avg brain spikes per 50ms bin)", flush=True)
    for label, c in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        w_in=0.8 * nA, w_scale=c.pop("w_scale"),
                        plastic=False, **c)
        fp, tb = run(pipe, PAUSE_MS)
        means = {w: np.mean(fp[w], 0) for w in WORDS}
        r = []
        for a, b in [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]:
            da = np.mean([np.abs(f - means[a]).sum() for f in fp[a]])
            db = np.mean([np.abs(f - means[b]).sum() for f in fp[b]])
            di = (da + db) / 2
            ds = np.abs(means[a] - means[b]).sum()
            r.append(di / (di + ds) if (di + ds) > 0 else float("nan"))
        per_bin = np.mean([[np.mean([tb[w][k][b] for w in WORDS]) for k in range(K)] for b in range(NBIN)], axis=1)
        print(f"{label:30s} {' | '.join(f'{x:5.2f}' for x in r)}    "
              f"{per_bin[0]:6.1f} {per_bin[1]:6.1f} {per_bin[2]:6.1f}", flush=True)


if __name__ == "__main__":
    main()