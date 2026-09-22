import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, nA, ms, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

graph = MaleCNSGraph("data/malecns_mbRlobes")
ia, ib = graph.kc_indices(100, seed=0)
acc, rej = graph.mbon_indices()

for ih in [0.0, 0.5, 1.0, 4.0, 16.0]:
    start_scope()
    np.random.seed(0)
    org_a = SensoryOrgan("A", 4, 100 * Hz)
    org_b = SensoryOrgan("B", 4, 100 * Hz)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    w_in=0.8 * nA, w_scale=0.001 * nA, plastic=False,
                    inh_scale=ih, tau_m=20 * ms, tau_syn=5 * ms)
    n_sp = {}
    for w in ["A", "B", "AB", "AABB"]:
        pipe.reset_state()
        t0, t1 = pipe.present_word(w)
        t = np.asarray(pipe.mon_brain.t / ms)
        i = pipe.mon_brain.i
        sel = (t >= t0 / ms) & (t <= t1 / ms)
        n_sp[w] = sel.sum()
        mb_a = np.isin(i[sel], acc).sum()
        mb_r = np.isin(i[sel], rej).sum()
        n_sp[w + "_mbon"] = (mb_a, mb_r)
    am = n_sp["A_mbon"][0] - n_sp["A_mbon"][1]
    bm = n_sp["B_mbon"][0] - n_sp["B_mbon"][1]
    abm = n_sp["AB_mbon"][0] - n_sp["AB_mbon"][1]
    aabm = n_sp["AABB_mbon"][0] - n_sp["AABB_mbon"][1]
    print(f"inh_scale={ih:5.1f}  brain A={n_sp['A']:6d} B={n_sp['B']:6d} "
          f"AB={n_sp['AB']:7d} AABB={n_sp['AABB']:7d}  "
          f"mbon(acc-rej) A/B/AB/AABB: {am:+d} {bm:+d} {abm:+d} {aabm:+d}")