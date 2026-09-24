import sys

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, nA, ms, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from grammar.ab import ABGrammar

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
        return np.isin(i[sel], idx).sum()

    n_acc, n_rej = len(pipe.out_acc), len(pipe.out_rej)
    n_a, n_b = len(pipe.in_a), len(pipe.in_b)
    tt = np.unique(t)
    t0s, t1s = t0 / ms, t1 / ms
    early_span = max(tt[(tt >= t0s) & (tt < mid)].max(initial=t0s) - t0s, 0.1) / 1e3
    late_span = max(t1s - tt[(tt >= mid) & (tt <= t1s)].min(initial=t1s), 0.1) / 1e3
    return np.array([
        h(sel_early, pipe.out_acc) / n_acc / early_span,
        h(sel_late, pipe.out_acc) / n_acc / late_span,
        h(sel_early, pipe.out_rej) / n_rej / early_span,
        h(sel_late, pipe.out_rej) / n_rej / late_span,
        h(sel_early, pipe.in_a) / n_a / early_span,
        h(sel_late, pipe.in_a) / n_a / late_span,
        h(sel_early, pipe.in_b) / n_b / early_span,
        h(sel_late, pipe.in_b) / n_b / late_span,
    ])


def collect(samples):
    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    pipe = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)
    F, y = [], []
    for w, l in samples:
        pipe.reset_state()
        t0, t1 = pipe.present_word(w)
        F.append(half_features(pipe, t0, t1))
        y.append(l)
    return np.array(F), np.array(y)


def auroc(score, y):
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    s = score.copy()
    order = np.argsort(s)
    srt = y[order]
    n_pos = (srt == 1).sum()
    n_neg = (srt == 0).sum()
    ranks = np.arange(1, len(srt) + 1)
    pos_ranks = ranks[srt == 1]
    return (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def candidate_readouts():
    # features: 0 accE 1 accL 2 rejE 3 rejL 4 aKE 5 aKL 6 bKE 7 bKL
    return {
        "mean_rate: acc-rej": lambda F: (F[:, 0] + F[:, 1]) - (F[:, 2] + F[:, 3]),
        "acc_L-acc_E": lambda F: F[:, 1] - F[:, 0],
        "rej_E-rej_L": lambda F: F[:, 2] - F[:, 3],
        "rej_L-rej_E": lambda F: F[:, 3] - F[:, 2],
        "acc+rej_L minus E": lambda F: (F[:, 1] + F[:, 3]) - (F[:, 0] + F[:, 2]),
        "KC a_E-a_L": lambda F: F[:, 4] - F[:, 5],
        "KC b_L-b_E": lambda F: F[:, 7] - F[:, 6],
        "KC (aE-aL)+(bL-bE)": lambda F: (F[:, 4] - F[:, 5]) + (F[:, 7] - F[:, 6]),
        "KC |bal| (neg)": lambda F: -np.abs(F[:, 4] + F[:, 5] - F[:, 6] - F[:, 7]),
    }


def build_sets():
    g = ABGrammar()
    # full train: 3 positives (n=1,2,3) + all negatives lengths 1-6 without BA
    train = g.train()
    # full official test: 3 positives (n=4,5,6) + sampled negatives lengths 7-13
    test_pos = [(g.positive(n), True) for n in g.TEST_NS]
    negs = g._negatives(g.TEST_NEG_LENGTHS, "all")
    rng = np.random.RandomState(0)
    by_len = {}
    for w, l in negs:
        by_len.setdefault(len(w), []).append((w, l))
    test_neg = []
    for L in g.TEST_NEG_LENGTHS:
        pool = by_len[L]
        idx = rng.choice(len(pool), min(120, len(pool)), replace=False)
        test_neg.extend(pool[int(i)] for i in idx)
    test = test_pos + test_neg
    # auxiliary pooled set with ALL positives n=1..6 for statistical power
    pool_pos = [(g.positive(n), True) for n in g.TRAIN_NS + g.TEST_NS]
    pool_neg = [(w, False) for w in g.all_negatives((1, 2, 3, 4, 5, 6, 7, 8))]
    rng = np.random.RandomState(1)
    by_len = {}
    for w, l in pool_neg:
        by_len.setdefault(len(w), []).append((w, l))
    pool_neg_s = []
    for L in sorted(by_len):
        pool = by_len[L]
        idx = rng.choice(len(pool), min(100, len(pool)), replace=False)
        pool_neg_s.extend(pool[int(i)] for i in idx)
    return train, test, pool_pos + pool_neg_s


def main():
    train, test, pool = build_sets()
    print(f"train n={len(train)} (pos={sum(l for _, l in train)}), "
          f"test n={len(test)} (pos={sum(l for _, l in test)}), "
          f"pool n={len(pool)} (pos={sum(l for _, l in pool)})")
    Ftr, ytr = collect(train)
    Fte, yte = collect(test)
    Fpo, ypo = collect(pool)
    print("\n--- AUROC of label-free fixed readout scores (score predefined, no fit) ---")
    print(f"{'score':28s}  train   test   pool")
    for name, fn in candidate_readouts().items():
        print(f"  {name:26s} {auroc(fn(Ftr), ytr):.3f}  "
              f"{auroc(fn(Fte), yte):.3f}  "
              f"{auroc(fn(Fpo), ypo):.3f}")


if __name__ == "__main__":
    main()