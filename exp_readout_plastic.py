import sys
import time

sys.path.insert(0, ".")

import numpy as np
from brian2 import Hz, ms, nA, start_scope

from brain.malecns import MaleCNSGraph
from organs.sensory import SensoryOrgan
from brain.pipeline import Pipeline
from brain.training import Trainer
from grammar.ab import ABGrammar

SEED = 1
EK = 50 * ms


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


def collect_features(pipe, samples, reset_state=True):
    F, y = [], []
    for w, l in samples:
        if reset_state:
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
    order = np.argsort(score)
    srt = y[order]
    n_pos = (srt == 1).sum()
    n_neg = (srt == 0).sum()
    ranks = np.arange(1, len(srt) + 1)
    pos_ranks = ranks[srt == 1]
    return (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


CHANNEL_SCORES = {
    # features: 0 accE 1 accL 2 rejE 3 rejL
    "acc_L-acc_E": lambda F: F[:, 1] - F[:, 0],
    "rej_L-rej_E": lambda F: F[:, 3] - F[:, 2],
    "mean_rate acc-rej": lambda F: (F[:, 0] + F[:, 1]) - (F[:, 2] + F[:, 3]),
    "acc+rej_L-E": lambda F: (F[:, 1] + F[:, 3]) - (F[:, 0] + F[:, 2]),
}


def build_sets():
    g = ABGrammar()
    train = g.train()
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


def report(pipe, tag, sets, yacc=None):
    line = [f"{tag:8s}"]
    for sname, (F, y) in sets.items():
        aurs = []
        for nm, fn in CHANNEL_SCORES.items():
            aurs.append(auroc(fn(F), y))
        line.append(f"{sname}: " + " ".join(f"{a:.2f}" for a in aurs))
    if yacc:
        accs = []
        for sname, (F, y) in sets.items():
            a = (F[:, 0] + F[:, 1]) / 2
            r = (F[:, 2] + F[:, 3]) / 2
            pred = a > r
            accs.append((pred == y).mean())
        line.append(f"chacc: " + " ".join(f"{c:.2f}" for c in accs))
    print("  " + " | ".join(line), flush=True)


def main():
    train, test, pool = build_sets()
    npos = {"train": sum(l for _, l in train),
            "test": sum(l for _, l in test),
            "pool": sum(l for _, l in pool)}
    print(f"train={len(train)} (pos={npos['train']}), "
          f"test={len(test)} (pos={npos['test']}), "
          f"pool={len(pool)} (pos={npos['pool']})")
    print("scores order per set: acc_L-acc_E, rej_L-rej_E, "
          "mean_rate acc-rej, acc+rej_L-E; then chacc")

    graph, ia, ib, acc, rej, org_a, org_b = make_probe(SEED)
    base = Pipeline(graph, org_a, org_b, ia, ib, acc, rej,
                    0.8 * nA, 0.001 * nA, plastic=False)
    sets = {"train": collect_features(base, train),
            "test": collect_features(base, test),
            "pool": collect_features(base, pool)}
    report(base, "base", sets)

    start_scope()
    np.random.seed(SEED)
    graph2, ia2, ib2, acc2, rej2, org_a2, org_b2 = make_probe(SEED)
    pipe = Pipeline(graph2, org_a2, org_b2, ia2, ib2, acc2, rej2,
                    0.8 * nA, 0.001 * nA, plastic=True,
                    kc_mbon_plastic=True, kc_mbon_stdp='single',
                    kc_scope='kc_out', tau_el=50 * ms,
                    a_plus=0.02, a_minus=0.01)
    trainer = Trainer(pipe, eta=0.01, w_max_scale=1.5)

    for ep in range(1, 31):
        t0 = time.time()
        trainer.fit(train, epochs=1)
        nsp = len(trainer.pipe.mon_brain.t)
        dt = time.time() - t0
        if ep % 5 == 0:
            psets = {"train": collect_features(pipe, train),
                     "test": collect_features(pipe, test),
                     "pool": collect_features(pipe, pool)}
            report(pipe, f"ep{ep:02d}", psets, yacc=True)
        else:
            print(f"  ep {ep:3d} dt={dt:5.1f}s spikes={nsp}", flush=True)
        if nsp > 6_000_000 or dt > 400:
            print(f"  ABORT at ep {ep}: pathological activity", flush=True)
            break


if __name__ == "__main__":
    main()