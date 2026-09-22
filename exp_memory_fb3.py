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
# (label, params, fb_nt_filter)
COMBOS = [
    ("w3 fb64 kk.30 t10/2 all", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30,
                                      tau_m=10 * ms, tau_syn=2 * ms), None),
    ("w3 fb64 kk.30 t10/2 exc", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30,
                                      tau_m=10 * ms, tau_syn=2 * ms), ("glutamate", "acetylcholine")),
    ("w3 fb64 kk.30 t10/2 gaba", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30,
                                       tau_m=10 * ms, tau_syn=2 * ms), ("gaba",)),
    ("w3 fb64 kk.30 t20/5 all", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.30,
                                      tau_m=20 * ms, tau_syn=5 * ms), None),
    ("w3 fb64 kk.10 t10/2 all", dict(w_scale=3 * PA, fb_boost=64.0, kc_kk_scale=0.10,
                                      tau_m=10 * ms, tau_syn=2 * ms), None),
    ("w2 fb64 kk.50 t10/2 all", dict(w_scale=2 * PA, fb_boost=64.0, kc_kk_scale=0.50,
                                      tau_m=10 * ms, tau_syn=2 * ms), None),
    ("w2 fb64 kk.50 t10/2 exc", dict(w_scale=2 * PA, fb_boost=64.0, kc_kk_scale=0.50,
                                      tau_m=10 * ms, tau_syn=2 * ms), ("glutamate", "acetylcholine")),
]
PAUSES_MS = [50, 150]
NBIN = 3


def fb_filter_mask(graph, nts):
    "bool over self.pairs rows: MBON->(KC|MBON) with pre NT in nts"
    nt = graph.nt
    pre_body = graph.pairs["bodyId_pre"].to_numpy()
    fb = graph.fb_mask()
    if nts is None:
        return fb
    pre_ids = [int(b) for b in pre_body]
    m = np.zeros(len(graph.pairs), dtype=bool)
    for k in np.nonzero(fb)[0]:
        if nt.get(pre_ids[k], "") in nts:
            m[k] = True
    return m


def mask_weights(pipe, fb_filter):
    if fb_filter is None:
        return
    m = fb_filter_mask(pipe.graph, fb_filter)
    w = pipe.graph.pairs["weight"].to_numpy() * pipe.w_scale
    w[m] *= pipe.fb_boost
    pipe.mcns_syn.w = w


def run(pipe, graph, fb_filter, pause_ms):
    mb = np.concatenate([pipe.out_acc, pipe.out_rej])
    fp = {w: [] for w in WORDS}
    for _ in range(K):
        for word in WORDS:
            pipe.reset_state()
            t0, t1 = pipe.present_word(word, delay=pause_ms * ms)
            t = np.asarray(pipe.mon_brain.t / ms)
            i = pipe.mon_brain.i
            w0, w1 = t1 / ms - pause_ms, t1 / ms
            if fb_filter is not None or pause_ms == 0:
                pass
            sel = (t >= w0) & (t <= w1)
            it = i[sel]
            bins = []
            for k in range(NBIN):
                a = w0 + k * (w1 - w0) / NBIN
                b = w0 + (k + 1) * (w1 - w0) / NBIN
                selk = (t >= a) & (t <= b)
                ik = i[selk]
                bins.append(np.array([np.isin(ik, nid).sum() for nid in mb]))
            fp[word].append(np.concatenate(bins))
    return fp


def main():
    graph = MaleCNSGraph("data/malecns_mbRlobes")
    ia, ib = graph.kc_indices(100, seed=0)
    acc, rej = graph.mbon_indices()
    print(f"{'config':30s} {'pause':>6s} | AB/BA | AAB/ABB | AABB/BBAA", flush=True)
    for label, c, fb_filter in COMBOS:
        start_scope()
        np.random.seed(0)
        org_a = SensoryOrgan("A", 4, 100 * Hz)
        org_b = SensoryOrgan("B", 4, 100 * Hz)
        pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                        w_in=0.8 * nA, w_scale=c.pop("w_scale"),
                        plastic=False, **c)
        mask_weights(pipe, fb_filter)
        lbl = label if fb_filter is None else label + f"[{'-'.join(fb_filter)}]"
        for pause_ms in PAUSES_MS:
            fp = run(pipe, graph, fb_filter, pause_ms)
            means = {w: np.mean(fp[w], 0) for w in WORDS}
            r = []
            for a, b in [("AB", "BA"), ("AAB", "ABB"), ("AABB", "BBAA")]:
                da = np.mean([np.abs(f - means[a]).sum() for f in fp[a]])
                db = np.mean([np.abs(f - means[b]).sum() for f in fp[b]])
                di = (da + db) / 2
                ds = np.abs(means[a] - means[b]).sum()
                r.append(di / (di + ds) if (di + ds) > 0 else float("nan"))
            print(f"{lbl:30s} {pause_ms:6d} | " +
                  " |    ".join(f"{x:6.2f}" for x in r), flush=True)
        print()


if __name__ == "__main__":
    main()