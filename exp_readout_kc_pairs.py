import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, nA, ms, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

SEED = 1

ORDER_PAIRS = [
    ('AB', ['BA', 'AA']),
    ('AABB', ['ABAB', 'BBAA', 'BAAB']),
    ('AAABBB', ['AABBAA', 'BAAABB', 'ABABAB', 'AAABAB']),
]


def make_probe(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    return graph, ia, ib, acc, rej, org_a, org_b


def kc_contrast(pipe, word):
    t0, t1 = pipe.present_word(word)
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return 0.0
    mid = (t0 / ms + t1 / ms) / 2
    sel_e = (t >= t0 / ms) & (t < mid)
    sel_l = (t >= mid) & (t <= t1 / ms)
    n_a = len(pipe.in_a)
    aE = np.isin(i[sel_e], pipe.in_a).sum() / n_a
    aL = np.isin(i[sel_l], pipe.in_a).sum() / n_a
    bE = np.isin(i[sel_e], pipe.in_b).sum() / n_a
    bL = np.isin(i[sel_l], pipe.in_b).sum() / n_a
    return (aE - aL) + (bL - bE), (aE + bL) - (aL + bE)


def main():
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)
    print("KC contrast (aE-aL)+(bL-bE) and (aE+bL)-(aL+bE) on order pairs:")
    for pos, negs in ORDER_PAIRS:
        for w in [pos] + negs:
            pipe.reset_state()
            c1, c2 = kc_contrast(pipe, w)
            flag = ""
            if w == pos:
                flag = " <- POS"
            print(f"  {w:7s} {c1:+7.1f} {c2:+7.1f}{flag}", flush=True)
        print()


if __name__ == "__main__":
    main()