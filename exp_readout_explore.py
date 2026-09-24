import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, nA, ms, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline

TRAIN = [('A', 0), ('B', 0), ('AB', 1), ('AAB', 0), ('ABB', 0),
         ('ABAB', 0), ('AAABB', 0), ('AABB', 1), ('AAABBB', 1)]
TEST = [('AAAB', 1), ('BAA', 0), ('AABBB', 0), ('AAAABBBB', 1),
        ('ABBA', 0), ('AAAA', 0), ('BBBB', 0), ('AAAAB', 0),
        ('ABBB', 0), ('AAAAABBBBB', 1), ('BA', 0), ('AABBAA', 0)]

SEED = 1


def make_probe(seed):
    start_scope()
    np.random.seed(seed)
    graph = MaleCNSGraph('data/malecns_mbRlobes')
    ia, ib = graph.kc_indices(100, seed=seed)
    acc, rej = graph.mbon_indices()
    org_a = SensoryOrgan('A', 4, 100 * Hz)
    org_b = SensoryOrgan('B', 4, 100 * Hz)
    return graph, ia, ib, acc, rej, org_a, org_b


def half_features(pipe, t0, t1):
    t = np.asarray(pipe.mon_brain.t / ms)
    i = pipe.mon_brain.i
    if len(t) == 0:
        return np.zeros(8)
    mid = (t0 / ms + t1 / ms) / 2
    sel_early = (t >= t0 / ms) & (t < mid)
    sel_late = (t >= mid) & (t <= t1 / ms)

    def h(sel, idx):
        if len(t) == 0:
            return 0.0
        return np.isin(i[sel], idx).sum()


    n_acc, n_rej = len(pipe.out_acc), len(pipe.out_rej)
    n_a, n_b = len(pipe.in_a), len(pipe.in_b)
    return np.array([
        h(sel_early, pipe.out_acc) / n_acc, h(sel_late, pipe.out_acc) / n_acc,
        h(sel_early, pipe.out_rej) / n_rej, h(sel_late, pipe.out_rej) / n_rej,
        h(sel_early, pipe.in_a) / n_a, h(sel_late, pipe.in_a) / n_a,
        h(sel_early, pipe.in_b) / n_b, h(sel_late, pipe.in_b) / n_b,
    ])


def collect(samples):
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)
    F, y, words = [], [], []
    for w, l in samples:
        pipe.reset_state()
        t0, t1 = pipe.present_word(w)
        F.append(half_features(pipe, t0, t1))
        y.append(l)
        words.append(w)
    return np.array(F), np.array(y), words


def main():
    print(f"seed={SEED}")
    print("fields: accE accL rejE rejL  aKE aKL  bKE bKL")
    print("=== TRAIN ===")
    F, y, words = collect(TRAIN)
    for w, f, yy in zip(words, F, y):
        print(f"  {w:6s} y={yy}  " + " ".join(f"{v:6.1f}" for v in f))
    print("=== TEST (new n + rules) ===")
    F2, y2, words2 = collect(TEST)
    for w, f, yy in zip(words2, F2, y2):
        print(f"  {w:8s} y={yy}  " + " ".join(f"{v:6.1f}" for v in f))


if __name__ == "__main__":
    main()