import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, mV, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

PA = 0.001 * nA
COMBOS = [
    {"w_scale": 3 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.30,
     "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.30,
     "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 2 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.30,
     "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 32.0, "kc_kk_scale": 0.30,
     "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 128.0, "kc_kk_scale": 0.30,
     "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.10,
     "tau_m": 20 * ms, "tau_syn": 5 * ms},
]
PAUSES = [0, 50, 150, 300, 600]  # ms
WORDS = ["AB", "BA", "AAB", "ABB", "AABB", "BBAA"]
K = 8


def run(pipe, pause_ms):
    fp = {w: [] for w in WORDS}
    tots = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            pipe.reset_state()
            t0, t1 = pipe.present_word(word, delay=pause_ms * ms)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            sel = (t >= w0) & (t <= w1)
            it = i[sel]
            tots[word].append(sel.sum())
            fp[word].append(np.array([np.isin(it, nid).sum() for nid in np.concatenate([pipe.out_acc, pipe.out_rej])]))
    return fp, tots


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    print(f"{'config':30s} {'pause':>6s} | "
          + "  ".join("AB/BA AAB/ABB AABB/BBAA".split())
          + "  | mean_tot")
    print(f"{'':30s} {'(ms)':>6s} |   ratio (lower = more separable)", flush=True)
    for c in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        w_in=0.8 * nA, w_scale=c.pop("w_scale"),
                        plastic=False, **c)
        wscale_pa = pipe.w_scale / nA * 1000
        fb = c.get("fb_boost", 1.0)
        kk = c.get("kc_kk_scale", 1.0)
        tm = c.get("tau_m", 10 * ms) / ms
        ts = c.get("tau_syn", 2 * ms) / ms
        label = f"w={wscale_pa:.0f}pa fb={fb:.0f} kk={kk:.2f} t={tm:.0f}/{ts:.0f}"
        for pause_ms in PAUSES:
            fp, tots = run(pipe, pause_ms)
            means = {w: np.mean(fp[w], 0) for w in WORDS}
            rat = []
            for a, b in [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]:
                da = np.mean([np.abs(f - means[a]).sum() for f in fp[a]])
                db = np.mean([np.abs(f - means[b]).sum() for f in fp[b]])
                di = (da + db) / 2
                ds = np.abs(means[a] - means[b]).sum()
                rat.append(di / (di + ds))
            m_tot = np.mean([np.mean(tots[w]) for w in WORDS])
            print(f"{label:30s} {pause_ms:6d} |   "
                  + "  ".join(f"{r:6.2f}" for r in rat)
                  + f"  | {m_tot:8.1f}", flush=True)
        print()


if __name__ == "__main__":
    main()