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
PA = 0.001 * nA
COMBOS = [
    # baseline (no fb) vs fb_boost sweeps
    {"w_scale": 1 * PA, "fb_boost": 1.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 1 * PA, "fb_boost": 4.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 1 * PA, "fb_boost": 16.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 1 * PA, "fb_boost": 64.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 2 * PA, "fb_boost": 4.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 2 * PA, "fb_boost": 16.0, "kc_kk_scale": 1.0, "tau_m": 10 * ms, "tau_syn": 2 * ms},
    {"w_scale": 2 * PA, "fb_boost": 64.0, "kc_kk_scale": 1.0, "tau_m": 20 * ms, "tau_syn": 5 * ms},
    # weaken isotropic KC->KC bonfire while boosting the feedback leg
    {"w_scale": 2 * PA, "fb_boost": 16.0, "kc_kk_scale": 0.5, "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 2 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.5, "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 16.0, "kc_kk_scale": 0.3, "tau_m": 20 * ms, "tau_syn": 5 * ms},
    {"w_scale": 3 * PA, "fb_boost": 64.0, "kc_kk_scale": 0.3, "tau_m": 20 * ms, "tau_syn": 5 * ms},
    # slow dynamics + fb
    {"w_scale": 2 * PA, "fb_boost": 64.0, "kc_kk_scale": 1.0, "tau_m": 30 * ms, "tau_syn": 8 * ms},
    {"w_scale": 4 * PA, "fb_boost": 16.0, "kc_kk_scale": 0.25, "tau_m": 30 * ms, "tau_syn": 8 * ms},
]


def run(pipe):
    pause_ms = PAUSE / ms
    fp = {w: [] for w in WORDS}
    dec = {w: [] for w in WORDS}
    tots = {w: [] for w in WORDS}
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
            a = np.isin(it, pipe.out_acc).sum() / len(pipe.out_acc)
            r = np.isin(it, pipe.out_rej).sum() / len(pipe.out_rej)
            dec[word].append(1 if a > r else 0)
            fp[word].append(np.array([np.isin(it, nid).sum() for nid in np.concatenate([pipe.out_acc, pipe.out_rej])]))
    return fp, dec, tots


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    pairs = [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]
    print(f"{'config':30s} {'word':6s} {'tot':>7s} {'pos%':>5s} | "
          f"{'d_intra':>8s} {'d_ABBA':>7s} {'d_AABABB':>8s} {'d_AABBBA':>8s} | {'ratio':>6s}")
    for c in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        w_in=0.8 * nA, w_scale=c.pop("w_scale"),
                        plastic=False, **c)
        fp, dec, tots = run(pipe)
        means = {w: np.mean(fp[w], 0) for w in WORDS}
        wscale_pa = pipe.w_scale / nA * 1000
        fb = c.get("fb_boost", 1.0)
        kk = c.get("kc_kk_scale", 1.0)
        tm = c.get("tau_m", 10 * ms) / ms
        ts = c.get("tau_syn", 2 * ms) / ms
        label = f"w={wscale_pa:.0f}pa fb={fb:.0f} kk={kk:.2f} t={tm:.0f}/{ts:.0f}"
        for w in WORDS:
            intra = np.mean([np.abs(f - means[w]).sum() for f in fp[w]])
            vals = []
            for a, b in pairs:
                vals.append(np.abs(means[a] - means[b]).sum() if w == a else float("nan"))
            nonnan = [v for v in vals if v == v] or [intra]
            ratio = intra / (intra + min(nonnan))
            pos = 100 * np.mean(dec[w])
            print(f"{label:30s} {w:6s} {np.mean(tots[w]):7.1f} {pos:5.0f} | {intra:8.1f} "
                  + " ".join(f"{v:7.1f}" if v == v else "     -- " for v in vals)
                  + f" | {ratio:6.2f}")
        print()


if __name__ == "__main__":
    main()